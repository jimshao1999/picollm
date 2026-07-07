import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.distributed as dist
from dataloader import DataLoaderLite
from hellaswag import get_most_likely_row, iterate_examples, render_example
from model import GPT, GPTConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class TrainConfig:
    B: int = 64
    T: int = 1024
    total_batch_size: int = 524288  # 2**19, ~0.5M # tokens
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
    learning_rate: float = 6e-4
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 250
    val_steps: int = 20
    sample_every: int = 250
    hella_every: int = 250
    ckpt_every: int = 5000
    log_dir: str = "log"
    vocab_size: int = 50304
    seed: int = 1337
    resume_path: str | None = None


class BaseTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.setup_ddp()

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        torch.set_float32_matmul_precision("high")

        assert (
            self.config.total_batch_size
            % (self.config.B * self.config.T * self.ddp_world_size)
            == 0
        ), "total_batch_size not divisible by B*T * # GPUs"
        self.grad_accum_steps = self.config.total_batch_size // (
            self.config.B * self.config.T * self.ddp_world_size
        )
        if self.master_process:
            print(f"total desired batch size: {self.config.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {self.grad_accum_steps}")

        self.setup_data()
        self.setup_model()
        self.setup_logging()

    def train(self):
        for step in range(self.start_step, self.config.max_steps):
            t0 = time.time()
            last_step = step == self.config.max_steps - 1

            if step % 250 == 0 or last_step:
                val_loss_accum = self._run_val(step)

                if self.master_process and step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(
                        self.config.log_dir, f"model_step_{step}.pt"
                    )
                    checkpoint = {
                        "model": self.raw_model.state_dict(),
                        "config": self.raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        # only required for resume training
                        "optimizer": self.optimizer.state_dict(),
                        "rng": torch.get_rng_state(),
                        "cuda_rng": (
                            torch.cuda.get_rng_state()
                            if torch.cuda.is_available()
                            else None
                        ),
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"saved checkpoint to {checkpoint_path}")

            if step % self.config.hella_every == 0 or last_step:
                self._run_hellaswag(step)

            if self.master_process and (step % 250 == 0 or last_step):
                self._sanity_generate()

            self.model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / self.grad_accum_steps  # since cross_entropy is doing mean
                loss_accum += loss.detach()
                if self.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.grad_accum_steps - 1
                    )
                loss.backward()
            if self.ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )  # prevent model from exploding due to data issues
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()
            if self.device_type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000  # miliseconds
            tokens_per_sec = (
                self.train_loader.B
                * self.train_loader.T
                * self.grad_accum_steps
                * self.ddp_world_size
            ) / (t1 - t0)
            if self.master_process:
                print(
                    f"step {step:4d}| loss: {loss_accum.item():.6f}| lr: {lr:.4e} | norm: {norm:.4f}| dt: {dt:.2f}ms| tok/sec: {tokens_per_sec:.2f}"
                )  # convert tensor to float live on cpu
                with open(self.log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")

        if self.ddp:
            destroy_process_group()

    def setup_ddp(self):
        # set up DDP (distributed data parallel)
        self.ddp = int(os.environ.get("RANK", -1) != -1)
        if self.ddp:
            assert torch.cuda.is_available(), "cuda is required for DDP"
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = (
                self.ddp_rank == 0
            )  # use RNAK=0 to do logging, checkpointing etc
        else:
            # vanilla, non-DDP
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            # mps device has weird fluctuations lead to training not converge
        # device_type ("cuda"/"mps"/"cpu") is what autocast expects, not "cuda:0"
        self.device_type = (
            "cuda" if str(self.device).startswith("cuda") else self.device
        )
        if self.master_process:
            print("using device: ", self.device)

    def setup_data(self):
        self.train_loader = DataLoaderLite(
            B=self.config.B,
            T=self.config.T,
            proc_rank=self.ddp_rank,
            num_procs=self.ddp_world_size,
            split="train",
        )
        self.val_loader = DataLoaderLite(
            B=self.config.B,
            T=self.config.T,
            proc_rank=self.ddp_rank,
            num_procs=self.ddp_world_size,
            split="val",
        )

    def setup_model(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.start_step = 0
        self.model = GPT(GPTConfig(vocab_size=50304))

        if self.config.resume_path is not None:
            ckpt = torch.load(
                self.config.resume_path, map_location=self.device, weights_only=False
            )
            self.model = GPT(ckpt["config"])
            # older checkpoints were saved from the compiled/DDP-wrapped model, so
            # keys may carry a "_orig_mod." (torch.compile) and/or "module." (DDP)
            # prefix. strip them so they match a fresh, unwrapped GPT.
            state_dict = {
                k.replace("_orig_mod.", "").replace("module.", ""): v
                for k, v in ckpt["model"].items()
            }
            self.model.load_state_dict(state_dict)
            self.start_step = ckpt["step"] + 1
            torch.set_rng_state(ckpt["rng"])
            if ckpt["cuda_rng"] is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng"])

        self.model.to(self.device)
        # raw_model = the bare, uncompiled/unwrapped GPT; grab it BEFORE compile/DDP.
        # it shares parameter tensors with self.model, so it always has the live
        # weights. use it for eval/generation (avoids torch.compile recompiles on
        # variable-length inputs) and for checkpoint/optimizer (clean state_dict
        # keys, no _orig_mod./module. prefixes, correct under DDP too).
        self.raw_model = self.model
        self.model = torch.compile(self.model)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            device=self.device,
        )
        if self.config.resume_path is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])

    def setup_logging(self):
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.config.log_dir, f"log.txt")
        with open(self.log_file, "w") as f:
            pass

    def _get_lr(self, it):
        if it < self.config.warmup_steps:
            return self.config.max_lr * (it + 1) / self.config.warmup_steps
        if it > self.config.max_steps:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)

    def _run_val(self, step: int):
        self.model.eval()
        self.val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = self.config.val_steps
            for _ in range(val_loss_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        return val_loss_accum

    @torch.no_grad()
    def _run_hellaswag(self, step: int):
        self.model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # shard examples across processes: each rank handles every Nth one
            if i % self.ddp_world_size != self.ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            # use the uncompiled model: inputs change shape every example
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                logits, _ = self.raw_model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # sum the per-rank counts across all processes
        if self.ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=self.device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if self.master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    def _sanity_generate(self):
        self.model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = self.enc.encode("Hello, I am a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (4, 8)
        xgen = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, _ = self.model(xgen)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                # top-k sampling of 50
                topk_probs, topk_indices = torch.topk(
                    probs, 50, dim=-1
                )  # becomes (B, 50)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=-1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = self.enc.decode(tokens)
            print(f"rank {self.ddp_rank} sample {i}: {decoded}")


if __name__ == "__main__":
    BaseTrainer(TrainConfig()).train()

from dataclasses import dataclass

import torch
from base_train import BaseTrainer, TrainConfig
from chat_cli import generate_reply
from model import GPT
from sft_data import SFTDataLoader
from tokenizer import ChatTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class SFTConfig(TrainConfig):
    max_lr: float = 3e-5
    min_lr: float = 0.0
    learning_rate: float = 3e-5
    warmup_steps: int = 50
    max_steps: int = 2000
    ckpt_every: int = 1000  # SFT is short/fast; save every 1000 (must be multiple of 250)
    log_dir: str = "log_sft"
    base_checkpoint: str = "log/model_step_19072.pt"
    mix_gsm8k: bool = False  # blend GSM8K into SFT (teaches math + #### format for RL)
    gsm8k_epochs: int = 4


class SFTTrainer(BaseTrainer):
    def setup_data(self):
        c = self.config
        self.tok = ChatTokenizer()  # shared by the loaders and _sanity_generate
        self.train_loader = SFTDataLoader(
            c.B,
            c.T,
            self.ddp_rank,
            self.ddp_world_size,
            split="train",
            tokenizer=self.tok,
            mix_gsm8k=c.mix_gsm8k,
            gsm8k_epochs=c.gsm8k_epochs,
        )
        self.val_loader = SFTDataLoader(
            c.B,
            c.T,
            self.ddp_rank,
            self.ddp_world_size,
            split="test",
            tokenizer=self.tok,
            mix_gsm8k=c.mix_gsm8k,
            gsm8k_epochs=c.gsm8k_epochs,
        )

    def setup_model(self):
        # resume from a mid-SFT checkpoint, else start fresh from the base model
        resume = self.config.resume_path is not None
        ckpt_path = self.config.resume_path if resume else self.config.base_checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = GPT(ckpt["config"])
        sd = {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in ckpt["model"].items()
        }
        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device)
        self.raw_model = self.model
        self.model = torch.compile(self.model)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            device=self.device,
        )
        if resume:
            # continue the SFT run: restore step, optimizer, and rng
            self.start_step = ckpt["step"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            torch.set_rng_state(ckpt["rng"].cpu())
            if ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
            if self.master_process:
                print(f"resumed SFT from {ckpt_path} at step {self.start_step}")
        else:
            self.start_step = 0

    def _run_hellaswag(self, step):
        pass

    SAMPLE_PROMPTS = [
        "What is the capital of France?",
        "Give me one tip for studying.",
        "Write a short poem about the moon.",
    ]

    def _sanity_generate(self):
        # in-training qualitative check: generate replies on the in-memory model
        # (raw_model = uncompiled, so variable-length generation won't recompile)
        self.model.eval()
        for p in self.SAMPLE_PROMPTS:
            reply = generate_reply(
                self.raw_model,
                self.tok,
                [{"role": "user", "content": p}],
                device=self.device,
                device_type=self.device_type,
                max_new_tokens=64,
                temperature=0.8,
            )
            print(f"[sample] Q: {p!r}")
            print(f"[sample] A: {reply!r}")
        self.model.train()


def overfit_one_batch(steps=200, lr=1e-3):
    """Sanity check: memorize a single batch. Loss should fall to ~0.
    If it can't, the data/mask/loss pipeline has a bug."""
    t = SFTTrainer(SFTConfig(max_steps=1))
    x, y = t.train_loader.next_batch()
    x, y = x.to(t.device), y.to(t.device)
    for g in t.optimizer.param_groups:
        g["lr"] = lr            # bump LR so it memorizes fast
        g["weight_decay"] = 0.0  # don't let decay fight memorization
    t.model.train()
    loss = None
    for i in range(steps):
        t.optimizer.zero_grad()
        with torch.autocast(device_type=t.device_type, dtype=torch.bfloat16):
            _, loss = t.model(x, y)
        loss.backward()
        t.optimizer.step()
        if i % 20 == 0:
            print(f"step {i:3d} | loss {loss.item():.4f}")
    print(f"final loss {loss.item():.4f}  (should be well under 0.5)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="overfit a single batch to sanity-check the SFT pipeline",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="base model checkpoint to SFT from (overrides SFTConfig default)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="dir for SFT logs + checkpoints (use a distinct name per base model)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="mid-SFT checkpoint to resume from (restores step+optimizer+rng)",
    )
    parser.add_argument(
        "--mix-gsm8k",
        action="store_true",
        help="blend GSM8K into the SFT mixture (teaches math + #### format for RL)",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=None,
        help="per-GPU micro-batch B (lower for big models, e.g. 16 for the 1B d26)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.base_checkpoint is not None:
        overrides["base_checkpoint"] = args.base_checkpoint
    if args.log_dir is not None:
        overrides["log_dir"] = args.log_dir
    if args.resume is not None:
        overrides["resume_path"] = args.resume
    if args.mix_gsm8k:
        overrides["mix_gsm8k"] = True
    if args.device_batch_size is not None:
        overrides["B"] = args.device_batch_size

    if args.overfit:
        overfit_one_batch()
    else:
        SFTTrainer(SFTConfig(**overrides)).train()

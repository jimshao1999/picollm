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
    log_dir: str = "log_sft"
    base_checkpoint: str = "log/model_step_19072.pt"


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
        )
        self.val_loader = SFTDataLoader(
            c.B,
            c.T,
            self.ddp_rank,
            self.ddp_world_size,
            split="test",
            tokenizer=self.tok,
        )

    def setup_model(self):
        self.start_step = 0
        # load the base model from the base model checkpoint
        ckpt = torch.load(
            self.config.base_checkpoint, map_location=self.device, weights_only=False
        )
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
    args = parser.parse_args()

    if args.overfit:
        overfit_one_batch()
    else:
        SFTTrainer(SFTConfig()).train()

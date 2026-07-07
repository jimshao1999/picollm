import torch
from datasets import load_dataset
from tokenizer import ChatTokenizer


class SFTDataLoader:
    """
    One conversation per row. Matches the DataloaderLite's interface.
    """

    def __init__(self, B, T, proc_rank, num_procs, split="train", tokenizer=None):
        self.B = B
        self.T = T
        self.proc_rank = proc_rank
        self.num_procs = num_procs
        assert split in {"train", "test"}
        self.dataset = SmolTalk(split=split)
        self.n = len(self.dataset)
        self.tokenizer = tokenizer or ChatTokenizer()
        self.pad_id = self.tokenizer.encode_special("<|endoftext|>")
        self.reset()

    def reset(self):
        self.cursor = self.proc_rank

    def _next_row(self):
        while True:
            convo = self.dataset[self.cursor]
            self.cursor = (self.cursor + self.num_procs) % self.n
            ids, mask = self.tokenizer.render_conversation(convo, max_tokens=self.T + 1)
            if sum(mask) > 0:
                return ids, mask

    def next_batch(self):
        B, T = self.B, self.T
        buf_ids = torch.full((B, T + 1), self.pad_id, dtype=torch.long)
        buf_mask = torch.zeros((B, T + 1), dtype=torch.long)
        for b in range(B):
            ids, mask = self._next_row()
            L: int = len(ids)
            buf_ids[b, :L] = torch.tensor(ids, dtype=torch.long)
            buf_mask[b, :L] = torch.tensor(mask, dtype=torch.long)

        x = buf_ids[:, :-1].contiguous()
        y = buf_ids[:, 1:].clone()
        y[buf_mask[:, 1:] == 0] = -1
        return x, y


class SmolTalk:
    def __init__(self, split="train"):
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(
            seed=42
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        messages = self.ds[idx]["messages"]
        return {"messages": messages}


def main():
    tok = ChatTokenizer()
    B, T = 2, 128
    loader = SFTDataLoader(B=B, T=T, proc_rank=0, num_procs=1, tokenizer=tok)
    x, y = loader.next_batch()

    # --- shape / dtype invariants ---
    print("shapes:", tuple(x.shape), tuple(y.shape))
    assert x.shape == (B, T) and y.shape == (B, T), "wrong batch shape"

    # --- inputs are valid ids; targets carry the -1 ignore mask ---
    assert (x >= 0).all(), "inputs must be valid token ids (no -1)"
    assert (y != -1).any(), "no supervised targets at all?"

    # --- every row has >=1 supervised target (else cross_entropy would nan) ---
    per_row = (y != -1).sum(dim=1)
    print("supervised targets per row:", per_row.tolist())
    assert (per_row > 0).all(), "a row has 0 supervised targets (would nan)"

    # --- shift alignment: y[t] must equal the next input x[t+1] wherever y != -1
    #     (both are buf[:, t+1]); this proves the mask/shift line up ---
    aligned = True
    for b in range(B):
        for t in range(T - 1):
            if y[b, t] != -1 and x[b, t + 1].item() != y[b, t].item():
                aligned = False
    print("shift alignment ok:", aligned)
    assert aligned, "target/input shift is misaligned"

    # --- decode the supervised targets of row 0: should be assistant text
    #     ending in <|assistant_end|>, with no user text / BOS ---
    sup = tok.tokenizer.decode([t for t in y[0].tolist() if t != -1])
    print("supervised (row 0):", repr(sup[:200]))

    # Across several batches: user tokens must NEVER be supervised (correctness),
    # and assistant_end must appear at least once (the model learns to stop).
    # A single long conversation can be truncated before its assistant_end, so we
    # check over many rows instead of one.
    user_start = tok.encode_special("<|user_start|>")
    assistant_end = tok.encode_special("<|assistant_end|>")
    seen = set()
    for _ in range(20):
        _, yb = loader.next_batch()
        seen.update(t for row in yb.tolist() for t in row if t != -1)
    assert user_start not in seen, "user tokens leaked into supervised targets"
    assert assistant_end in seen, "assistant_end never supervised (check truncation / T)"

    print("\nphase 3 checks passed")


if __name__ == "__main__":
    main()

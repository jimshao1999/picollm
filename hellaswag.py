"""
HellaSwag eval, adapted from the https://github.com/karpathy/nanoGPT.git

Each example is rendered into 4 rows (one per candidate ending). We score each
completion by the average cross-entropy over the completion tokens (mask == 1)
and pick the lowest-loss row as the prediction (length-normalized accuracy).

Standalone GPT-2 baseline:
    python hellaswag.py -m gpt2 -d cuda
"""

import json
import os

import requests
import tiktoken
import torch
from torch.nn import functional as F
from tqdm import tqdm

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size: int = 1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}")
        download_file(data_url, data_filename)


def render_example(example):
    """Return (data, tokens, mask, label).

    tokens: (4, max_len) token ids for the 4 ctx+ending candidates
    mask:   (4, max_len) 1s over the ending region (where loss is measured)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "endings_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        ending_tokens = enc.encode(" " + end)  # leading space for GPT-2 tokenization
        tok_rows.append(ctx_tokens + ending_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(ending_tokens))
        data["endings_tokens"].append(ending_tokens)

    max_len = max(len(t) for t in tok_rows)
    tokens = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
    mask = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    download(split)
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(data_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def get_most_likely_row(tokens, mask, logits):
    """Given logits (B, T, V), return the index of the completion with the
    lowest average (length-normalized) loss over its masked region."""
    # autoregressive loss at every position
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # average the loss over the completion region only (mask shifted too)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # lowest length-normalized loss = most likely completion
    pred_norm = avg_loss.argmin().item()
    return pred_norm


@torch.no_grad()
def evaluate(model_type, device):
    """Standalone baseline: score HuggingFace GPT-2 on HellaSwag val."""
    from transformers import GPT2LMHeadModel  # heavy import, only needed here

    torch.set_float32_matmul_precision("high")  # tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits
        pred_norm = get_most_likely_row(tokens, mask, logits)

        # non-normalized prediction (sum of loss) for reference
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        shift_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
            reduction="none",
        ).view(tokens.size(0), -1)
        masked = shift_losses * mask[..., 1:].contiguous()
        pred = masked.sum(dim=1).argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(
            f"{num_total} acc_norm: {num_correct_norm}/{num_total}="
            f"{num_correct_norm/num_total:.4f}"
        )

    print(
        f"final: acc {num_correct/num_total:.4f} | "
        f"acc_norm {num_correct_norm/num_total:.4f} over {num_total} examples"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)

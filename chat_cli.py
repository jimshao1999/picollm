import argparse

import torch
import torch.nn.functional as F
from model import GPT
from tokenizer import ChatTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config)
    sd = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate_reply(
    model,
    tok,
    messages,
    device,
    device_type="cuda",
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
):
    """Generate one assistant reply given a list of message dicts.

    Model-agnostic: works on any in-memory model (training-time sampling) or a
    disk-loaded one (CLI). Does not touch disk / stdin.
    """
    assistant_start = tok.encode_special("<|assistant_start|>")
    assistant_end = tok.encode_special("<|assistant_end|>")
    # render the conversation (ending on the user turn) then prime with
    # <|assistant_start|>. big max_tokens so it never truncates the latest turn;
    # we keep the context window in check via idx_cond below.
    ids, _ = tok.render_conversation({"messages": messages}, max_tokens=10**9)
    ids = ids + [assistant_start]
    x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, L)

    out = []
    for _ in range(max_new_tokens):
        idx_cond = x[:, -model.config.block_size :]  # crop to context window
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # (1, 1)
        tid = nxt.item()
        if tid == assistant_end:  # model learned to stop
            break
        out.append(tid)
        x = torch.cat([x, nxt], dim=-1)
    return tok.tokenizer.decode([t for t in out if t < 50257])  # drop special tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="log_sft/model_step_01999.pt")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    model = load_model(args.ckpt)
    tok = ChatTokenizer()
    device_type = "cuda" if device.startswith("cuda") else device
    messages = []
    print("chat — Ctrl-C to exit")
    while True:
        try:
            user = input("\nyou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        messages.append({"role": "user", "content": user})
        reply = generate_reply(
            model,
            tok,
            messages,
            device=device,
            device_type=device_type,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("assistant:", reply)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()


"""
Did SFT break arithmetic, or did the base never have it?

Probe the PRE-SFT base checkpoint (the raw autocompleter) with few-shot raw
completion — no chat format, greedy. If the base can't add either, the
capability was never there and SFT is exonerated.
"""
import torch
from chat_cli import device, load_model
from tokenizer import ChatTokenizer

BASE_CKPT = "log_d26/model_step_37999.pt"  # pretrained-only, before any SFT

model = load_model(BASE_CKPT)
enc = ChatTokenizer().tokenizer  # underlying tiktoken (raw text, no chat tokens)


@torch.no_grad()
def complete(prompt, n=8):
    ids = enc.encode(prompt)
    x = torch.tensor([ids], device=device)
    for _ in range(n):
        logits, _ = model(x)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)  # greedy
        x = torch.cat([x, nxt], dim=1)
    return enc.decode(x[0].tolist()[len(ids):])


# few-shot so the base continues in "a + b = c" style instead of prose
fewshot = "2 + 3 = 5\n7 + 8 = 15\n4 + 5 = 9\n10 + 6 = 16\n"
tests = [("6 + 3", 9), ("9 + 9", 18), ("13 + 24", 37),
         ("47 + 85", 132), ("50 + 25", 75), ("12 + 7", 19)]

correct = 0
for q, gold in tests:
    out = complete(fewshot + f"{q} = ", n=6).strip()
    first_line = out.splitlines()[0] if out else ""
    ok = str(gold) in first_line.replace(" ", "")
    correct += ok
    print(f"{q} = ?  gold={gold:<4} base->{first_line!r:<20} {'OK' if ok else 'x'}")
print(f"\nbase got {correct}/{len(tests)} correct")
print("=> if ~0, the base never learned arithmetic; SFT did not break it.")

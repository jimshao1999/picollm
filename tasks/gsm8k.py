"""
GSM8K task — grade-school math word problems with verifiable numeric answers.
https://huggingface.co/datasets/openai/gsm8k

Used for:
- the RL pass@k gate (does the SFT model ever solve GSM8K? -> is there signal?)
- the RL loop (reward = did the final answer match?)
- optionally, the SFT mixture (teach the #### answer format)

v1 is intentionally simple: no calculator tool-call parsing (that's the `<<...>>`
tags nanochat handles). We just strip those annotations and score the final number.
"""

import re

from datasets import load_dataset

# the official GSM8K final-answer marker, e.g. "#### 42"
GSM_RE = re.compile(r"####\s*(-?[0-9\.,]+)")
# calculator annotations inside the reference solution, e.g. "<<12/60=0.2>>"
CALC_RE = re.compile(r"<<[^>]*>>")
# any number, for the lenient fallback
NUM_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")


def extract_answer(text):
    """Strict: the number after the '####' marker (official format)."""
    if not text:
        return None
    m = GSM_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().replace(",", "").rstrip(".")


def extract_last_number(text):
    """Lenient fallback: the last number anywhere in the text.
    Lets us score a model that reasons correctly but doesn't emit '#### N'
    (e.g. a SmolTalk-only SFT model that never learned the format)."""
    if not text:
        return None
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


class GSM8K:
    def __init__(self, subset="main", split="train"):
        assert subset in ("main", "socratic"), "subset must be main|socratic"
        assert split in ("train", "test"), "split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        question = row["question"]
        gold = extract_answer(row["answer"])  # ground-truth final number
        answer = CALC_RE.sub("", row["answer"])  # solution text, calculator tags stripped
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "answer": gold,  # the number we grade against
        }

    def evaluate(self, example, generated_text, lenient=True):
        """0/1 correctness: does the generated final number match the gold?"""
        gold = example["answer"]
        pred = extract_answer(generated_text)
        if pred is None and lenient:
            pred = extract_last_number(generated_text)
        return int(pred is not None and gold is not None and pred == gold)

    def reward(self, example, generated_text):
        """RL reward = correctness as a float (0.0 / 1.0)."""
        return float(self.evaluate(example, generated_text))


def run_gate(ckpt_path, n=100, k=8, temperature=1.0, max_new_tokens=256, lenient=True):
    """RL signal gate: sample k completions per question and report pass@1 / pass@k.
    pass@k > 0 => the model sometimes succeeds => RL has something to amplify."""
    from chat_cli import device, generate_reply, load_model
    from tokenizer import ChatTokenizer

    model = load_model(ckpt_path)
    tok = ChatTokenizer()
    device_type = "cuda" if device.startswith("cuda") else device
    task = GSM8K(subset="main", split="test")
    n = min(n, len(task))
    print(f"gate: ckpt={ckpt_path} | n={n} questions | k={k} samples | "
          f"temp={temperature} | lenient={lenient}\n")

    correct_samples, total_samples, solved_at_k = 0, 0, 0
    for i in range(n):
        ex = task[i]
        question = ex["messages"][0]["content"]
        flags = []
        for _ in range(k):
            out = generate_reply(
                model, tok, [{"role": "user", "content": question}],
                device=device, device_type=device_type,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            flags.append(task.evaluate(ex, out, lenient=lenient))
        correct_samples += sum(flags)
        total_samples += k
        solved_at_k += 1 if any(flags) else 0
        print(f"[{i+1:3d}/{n}] pass@1~{correct_samples/total_samples:.3f} "
              f"pass@{k}~{solved_at_k/(i+1):.3f}")

    pass1 = correct_samples / total_samples
    passk = solved_at_k / n
    print(f"\n=== GATE ({n} q, k={k}, temp={temperature}, lenient={lenient}) ===")
    print(f"pass@1  = {pass1:.4f}  (avg single-sample accuracy)")
    print(f"pass@{k} = {passk:.4f}  (solved by >=1 of {k} samples)")
    if passk > 0:
        print(f"\n=> SIGNAL: pass@{k} > 0. RL has room to lift pass@1 toward pass@{k}.")
    else:
        print(f"\n=> NO SIGNAL: pass@{k} = 0. Fix the base first "
              "(blend GSM8K into SFT, or scale up) before RL.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gate", action="store_true", help="run the pass@k RL-signal gate on a checkpoint")
    p.add_argument("--ckpt", type=str, default="log-sft-modern/model_step_1000.pt")
    p.add_argument("--n", type=int, default=100, help="number of test questions")
    p.add_argument("--k", type=int, default=8, help="samples per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--strict", action="store_true", help="require '#### N' (default: lenient)")
    args = p.parse_args()

    if args.gate:
        run_gate(args.ckpt, n=args.n, k=args.k, temperature=args.temperature,
                 max_new_tokens=args.max_new_tokens, lenient=not args.strict)
    else:
        # smoke test: load, show one example, sanity-check the reward
        task = GSM8K(subset="main", split="train")
        print(f"loaded {len(task)} train examples")
        ex = task[0]
        print("\n--- question ---\n", ex["messages"][0]["content"])
        print("\n--- gold answer ---", ex["answer"])
        ref = ex["messages"][1]["content"]
        print("\nreward(gold solution) =", task.reward(ex, ref), "(expect 1.0)")
        print("reward('the answer is 999999') =", task.reward(ex, "the answer is 999999"), "(expect 0.0)")
        print(f"reward('#### {ex['answer']}') =", task.reward(ex, f"#### {ex['answer']}"), "(expect 1.0)")

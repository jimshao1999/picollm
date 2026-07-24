"""
Synthetic arithmetic task — a controllable, verifiable math task for the
SFT-drill -> RL pipeline.

The base 1B can't compute (it rambles / copies few-shot answers), so the plan is
to TEACH arithmetic in SFT first, then RL to push accuracy higher. To that end
each example ships a worked chain-of-thought solution (place-value decomposition)
ending in the '#### N' format — that's how a small model actually learns to
compute, and it matches GSM8K's reward format (strict '#### N').

Difficulty is tunable via `max_int` (operand size) and `ops`. Same interface as
tasks/gsm8k.py: __getitem__ -> {"messages", "answer"}, plus reward()/evaluate().
Problems are generated deterministically per index (seeded); train/test disjoint.
"""

import random

from tasks.gsm8k import extract_answer, extract_last_number  # reuse the numeric parsers

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def _solution(a, op, b, ans):
    """A short worked scratchpad ending in '#### <ans>'. The decomposition splits
    off the tens of the second operand, so every intermediate step is a value we
    compute directly (no carry/borrow logic to get wrong) — always correct, and
    it teaches a real place-value strategy the model can generalize."""
    if op == "+":
        s1 = a + (b // 10) * 10
        return (f"{a} + {b}. First add the tens: {a} + {(b // 10) * 10} = {s1}. "
                f"Then add the ones: {s1} + {b % 10} = {ans}.\n#### {ans}")
    if op == "-":
        s1 = a - (b // 10) * 10
        return (f"{a} - {b}. First subtract the tens: {a} - {(b // 10) * 10} = {s1}. "
                f"Then subtract the ones: {s1} - {b % 10} = {ans}.\n#### {ans}")
    return f"{a} * {b} = {ans}.\n#### {ans}"


class Arithmetic:
    def __init__(self, split="train", n=100_000, max_int=99, ops="+-", seed=0):
        assert split in ("train", "test")
        self.split = split
        self.n = n if split == "train" else max(1, n // 100)
        self.max_int = max_int
        self.ops = list(ops)
        # disjoint train/test by offsetting the seed space
        self.base = seed + (0 if split == "train" else 10_000_000)

    def __len__(self):
        return self.n

    def _problem(self, idx):
        rng = random.Random(self.base + idx)
        op = rng.choice(self.ops)
        a, b = rng.randint(0, self.max_int), rng.randint(0, self.max_int)
        if op == "-" and b > a:  # keep subtraction non-negative
            a, b = b, a
        return a, op, b, _OPS[op](a, b)

    def __getitem__(self, idx):
        a, op, b, ans = self._problem(idx % self.n)
        return {
            "messages": [
                {"role": "user", "content": f"What is {a} {op} {b}?"},
                {"role": "assistant", "content": _solution(a, op, b, ans)},
            ],
            "answer": str(ans),
        }

    def evaluate(self, example, generated_text, lenient=True):
        gold = example["answer"]
        pred = extract_answer(generated_text)
        if pred is None and lenient:
            pred = extract_last_number(generated_text)
        return int(pred is not None and gold is not None and pred == gold)

    def reward(self, example, generated_text):
        return float(self.evaluate(example, generated_text))


if __name__ == "__main__":
    t = Arithmetic(split="train")
    print("len:", len(t))
    for i in range(3):
        ex = t[i]
        print(ex["messages"][0]["content"], "->", ex["answer"])
    ex = t[0]
    print("reward(gold) =", t.reward(ex, ex["messages"][1]["content"]), "(expect 1.0)")
    print("reward('the answer is 99999') =", t.reward(ex, "the answer is 99999"), "(expect 0.0)")

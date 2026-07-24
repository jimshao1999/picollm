"""
Easier real math word-problem sets (SVAMP, MultiArith) — genuinely easier than
GSM8K (1–2 reasoning steps vs GSM8K's 2–8), with the same verifiable numeric
answer. A middle ground between synthetic arithmetic and GSM8K when GSM8K is too
hard for the base model to give RL any signal.

Same interface as tasks/gsm8k.py: __getitem__ -> {"messages", "answer"},
plus reward()/evaluate().

PRESETS maps a short name to (hf_repo, subset, question_fields, answer_field).
If a dataset's schema/split differs on your HF version, tweak the preset here.
"""

import re

from datasets import load_dataset

from tasks.gsm8k import extract_answer, extract_last_number  # reuse the numeric parsers

# name -> (hf_repo, subset, [question fields to join], answer field)
PRESETS = {
    "svamp": ("ChilleD/SVAMP", None, ["Body", "Question"], "Answer"),
    "multiarith": ("ChilleD/MultiArith", None, ["question"], "final_ans"),
}


def _clean_number(x):
    """Normalize a gold answer to a bare numeric string (SVAMP answers are floats
    like 7.0 -> '7'; MultiArith are already ints)."""
    s = str(x).strip()
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return s
    n = m.group(0)
    if "." in n:  # drop trailing ".0" so it matches an integer prediction
        n = n.rstrip("0").rstrip(".")
    return n


class SimpleMath:
    def __init__(self, name="svamp", split="train"):
        assert name in PRESETS, f"unknown set '{name}'; add it to PRESETS"
        repo, subset, self.q_fields, self.a_field = PRESETS[name]
        ds = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
        self.ds = ds.shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        question = " ".join(str(row[f]).strip() for f in self.q_fields if row.get(f) is not None)
        gold = _clean_number(row[self.a_field])
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"#### {gold}"},
            ],
            "answer": gold,
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
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "svamp"
    t = SimpleMath(name=name, split="train")
    print(f"loaded {len(t)} '{name}' train examples")
    for i in range(3):
        ex = t[i]
        print("\nQ:", ex["messages"][0]["content"], "\nA:", ex["answer"])
    ex = t[0]
    print("\nreward('#### %s') =" % ex["answer"], t.reward(ex, f"#### {ex['answer']}"), "(expect 1.0)")
    print("reward('nonsense 99999') =", t.reward(ex, "nonsense 99999"), "(expect 0.0)")

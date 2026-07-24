"""Task registry — one place to build a verifiable-reward task by name.

All tasks share the same interface:
    __getitem__(i) -> {"messages": [...], "answer": <gold>}
    reward(example, text) / evaluate(example, text, lenient=...)

so rl_train.py and the pass@k gate are task-agnostic (`--task <name>`).

  gsm8k       — GSM8K grade-school math (hard for a small model; sparse signal)
  arithmetic  — synthetic add/sub/mul, tunable difficulty (guaranteed dense signal)
  svamp       — SVAMP word problems (easier real math)
  multiarith  — MultiArith word problems (easier real math)
"""

TASKS = ("gsm8k", "arithmetic", "svamp", "multiarith")


def make_task(name, split):
    """Return a task object for `name` on `split` ('train'|'test').
    Imports are lazy so we only pull in `datasets` for the task actually used."""
    if name == "gsm8k":
        from tasks.gsm8k import GSM8K
        return GSM8K(subset="main", split=split)
    if name == "arithmetic":
        from tasks.arithmetic import Arithmetic
        return Arithmetic(split=split)
    if name in ("svamp", "multiarith"):
        from tasks.simple_math import SimpleMath
        return SimpleMath(name=name, split=split)
    raise ValueError(f"unknown task '{name}'; choose from {TASKS}")

"""
Plot comparison graphs across the picollm runs vs GPT-2 references.

- loss curve:      val loss (FineWeb-Edu) vs training step, for our 3 runs.
- hellaswag curve: HellaSwag acc_norm vs step, for our runs + GPT-2 reference lines.

Note: external GPT-2 *loss* is on a different corpus/tokenizer (WebText), so it is
NOT comparable to our FineWeb-Edu val loss -> it is deliberately absent from the
loss plot. HellaSwag is a standard benchmark, so GPT-2 numbers appear as reference
lines there.

Run:  python plot_results.py   ->   writes plots/loss_curve.png, plots/hellaswag_curve.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOK_PER_STEP = 524288  # all runs use the same ~0.5M tokens/step, so step ∝ tokens

# our runs: label -> log file
RUNS = {
    "picollm GPT-2 124M (baseline)": "log/log_gpt2_baseline.txt",
    "picollm modern 124M": "log_modern/log.txt",
    "picollm modern 1B (d26)": "log_d26/log.txt",
}
COLORS = {
    "picollm GPT-2 124M (baseline)": "tab:gray",
    "picollm modern 124M": "tab:blue",
    "picollm modern 1B (d26)": "tab:red",
}

# published GPT-2 HellaSwag acc_norm (reference lines)
GPT2_HELLA_REF = {
    "GPT-2 124M (ref)": 0.2955,
    "GPT-2 Large 774M (ref)": 0.395,
    "GPT-2 XL 1.5B (ref)": 0.489,
}
# our GPT-2 124M baseline only has a final HellaSwag value (recovered), not a curve
BASELINE_FINAL_HELLA = (19072, 0.281)

# SFT runs (same SmolTalk + GSM8K×4 mix) — label -> (log file, color)
SFT_RUNS = {
    "SFT 124M (SmolTalk + GSM8K)": ("log_sft_mixed/log.txt", "tab:blue"),
    "SFT 1B / d26 (SmolTalk + GSM8K)": ("log_sft_d26/log.txt", "tab:red"),
}

# the successful arithmetic RLVR run (GRPO on the SFT-drilled 1B)
RL_LOG = "log_rl_d26_arith3/log.txt"


def parse_log(path, key):
    """Return (steps, values) for lines shaped like '<step> <key> <value>'."""
    steps, vals = [], []
    if not os.path.exists(path):
        print(f"  [warn] missing: {path}")
        return steps, vals
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 3 and p[1] == key:
                try:
                    steps.append(int(p[0]))
                    vals.append(float(p[2]))
                except ValueError:
                    pass
    return steps, vals


def plot_loss():
    plt.figure(figsize=(8, 5))
    for name, path in RUNS.items():
        s, v = parse_log(path, "val")
        pts = [(x, y) for x, y in zip(s, v) if x > 0]  # drop init (~10.8) for readable y-range
        if pts:
            xs, ys = zip(*pts)
            plt.plot(xs, ys, marker="o", ms=3, lw=1.6, color=COLORS[name], label=name)
    plt.xlabel("training step  (×0.5M tokens/step)")
    plt.ylabel("val loss — FineWeb-Edu (nats)")
    plt.title("Validation loss vs training step")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.figtext(0.5, -0.03,
                "external GPT-2 loss omitted: trained on a different corpus/tokenizer, not comparable",
                ha="center", fontsize=8, style="italic")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_curve.png", dpi=150, bbox_inches="tight")
    print("wrote plots/loss_curve.png")


def plot_hellaswag():
    plt.figure(figsize=(8, 5))
    for name, path in RUNS.items():
        s, v = parse_log(path, "hella")
        if v:
            plt.plot(s, v, marker="o", ms=3, lw=1.6, color=COLORS[name], label=name)
    # baseline: only a final recovered HellaSwag point
    bs, bh = BASELINE_FINAL_HELLA
    plt.scatter([bs], [bh], marker="*", s=160, color=COLORS["picollm GPT-2 124M (baseline)"],
                zorder=5, label="picollm GPT-2 124M (final)")
    # GPT-2 reference lines
    for name, y in GPT2_HELLA_REF.items():
        plt.axhline(y, ls="--", lw=1.2, alpha=0.7, label=name)
    plt.axhline(0.25, ls=":", lw=1.0, color="black", alpha=0.4, label="chance (0.25)")
    plt.xlabel("training step  (×0.5M tokens/step)")
    plt.ylabel("HellaSwag acc_norm")
    plt.title("HellaSwag vs training step")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/hellaswag_curve.png", dpi=150, bbox_inches="tight")
    print("wrote plots/hellaswag_curve.png")


def plot_sft():
    plt.figure(figsize=(8, 5))
    for name, (path, color) in SFT_RUNS.items():
        s, v = parse_log(path, "val")
        if v:
            plt.plot(s, v, marker="o", ms=3, lw=1.6, color=color, label=name)
            plt.annotate(f"{v[-1]:.2f}", (s[-1], v[-1]), textcoords="offset points",
                         xytext=(6, 0), fontsize=9, color=color, va="center")
    plt.xlabel("SFT step  (×0.5M tokens/step)")
    plt.ylabel("SFT val loss — SmolTalk + GSM8K (nats)")
    plt.title("SFT validation loss — 1B vs 124M (same data)")
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/sft_curve.png", dpi=150, bbox_inches="tight")
    print("wrote plots/sft_curve.png")


def plot_rl():
    """GRPO on arithmetic: pass@1 climbing from the SFT start toward pass@k."""
    s1, v1 = parse_log(RL_LOG, "eval_pass@1")
    s8, v8 = parse_log(RL_LOG, "eval_pass@8")
    if not v1:
        print(f"  [warn] no RL evals in {RL_LOG}; skipping rl_curve")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(s8, v8, marker="s", ms=3, lw=1.4, color="tab:orange", label="pass@8 (ceiling)")
    plt.plot(s1, v1, marker="o", ms=3, lw=1.8, color="tab:red", label="pass@1")
    plt.axhline(v1[0], ls=":", lw=1.0, color="tab:red", alpha=0.6,
                label=f"SFT start pass@1 = {v1[0]:.2f}")
    plt.annotate(f"{v1[-1]:.2f}", (s1[-1], v1[-1]), textcoords="offset points",
                 xytext=(6, 0), fontsize=9, color="tab:red", va="center")
    plt.xlabel("GRPO step")
    plt.ylabel("arithmetic accuracy (held-out 2-digit add/sub)")
    plt.title("RLVR that works: GRPO lifts pass@1 from SFT 0.48 → 0.76")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/rl_curve.png", dpi=150, bbox_inches="tight")
    print("wrote plots/rl_curve.png")


if __name__ == "__main__":
    plot_loss()
    plot_hellaswag()
    plot_sft()
    plot_rl()
    # quick summary of final numbers
    print("\nfinal numbers:")
    for name, path in RUNS.items():
        _, v = parse_log(path, "val")
        _, h = parse_log(path, "hella")
        vfin = v[-1] if v else None
        hfin = h[-1] if h else BASELINE_FINAL_HELLA[1]
        print(f"  {name:32s} val={vfin}  hella={hfin}")
    print("SFT final val:")
    for name, (path, _) in SFT_RUNS.items():
        _, v = parse_log(path, "val")
        print(f"  {name:34s} val={v[-1] if v else None}")

# picollm

Have fun training LLM!

A from-scratch reproduction of **GPT-2 (124M)**, pretrained on 10B tokens of FineWeb-Edu. Following Karpathy's [build-nanogpt](https://github.com/karpathy/nanogpt), and [build-nanochat](https://github.com/karpathy/nanochat)!

## Layout

| File | Purpose |
|------|---------|
| `model.py` | `GPT` model + `GPTConfig` (learned pos-emb, LayerNorm, GELU, flash attention, weight tying) |
| `dataloader.py` | `DataLoaderLite` — streams pre-tokenized `.npy` shards |
| `base_train.py` | `TrainConfig` + `BaseTrainer` (DDP, grad accum, cosine LR, val, sampling, HellaSwag, checkpoints) |
| `hellaswag.py` | HellaSwag rendering, scoring (`get_most_likely_row`), and a standalone GPT-2 baseline |
| `finewebedu_curator.py` | Downloads + tokenizes FineWeb-Edu into 100 shards under `edu_fineweb10B/` |

## Usage

```bash
# 1. Build the dataset (100 shards, ~19 GB, 10B tokens; val = shard 0)
python finewebedu_curator.py

# 2. Train
python base_train.py                                  # single GPU / CPU / MPS
torchrun --standalone --nproc_per_node=8 base_train.py  # multi-GPU

# 3. HellaSwag baseline for reference (needs internet the first time)
python hellaswag.py -m gpt2 -d cuda
```

Training config lives in `TrainConfig` (`base_train.py`): full run is
`max_steps=19073`, `total_batch_size=524288` (~0.5M tokens/step), i.e. ~10B
tokens = 1 epoch over the dataset.

## Results

Two architectures trained from scratch under identical config, data, and budget
(19073 steps, 10B tokens of FineWeb-Edu, ~0.5M tokens/step): the original **GPT-2
(2019)** recipe, and a **modern dense** stack (RoPE, RMSNorm, QK-norm, ReLU² MLP,
no biases, untied embeddings, logit softcap).

| Metric | GPT-2 baseline | modern-dense | reference (GPT-2 / build-nanogpt) |
|--------|----------------|--------------|------------------------------------|
| Final val loss (FineWeb-Edu) | 3.257 | **3.180** | ~3.28 |
| HellaSwag acc_norm | 0.281 | **0.287** | 0.2955 / ~0.305 |

The baseline is healthy: init loss 10.955 (≈ ln(50304)), final val on par with the
reference. The **modern architecture is ~0.08 nats lower across the whole curve**
(~7% lower perplexity) — a clean, consistent win. HellaSwag barely moves (+0.006):
it's an emergent benchmark that stays near chance at 124M scale regardless of loss.
Caveat: untying the embeddings adds ~39M params (~163M vs 124M total), so part of
the gain is extra capacity, not pure architectural efficiency.

## SFT — chat model

Supervised fine-tuning of the base model on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) turns the autocompleter into a chat model. Conversations are rendered with role special tokens (reusing the free padded vocab slots 50257–50260) and loss is masked to assistant tokens only (`ignore_index=-1`).

```bash
python chat_sft.py --overfit          # sanity check: memorize one batch (loss -> ~0)
torchrun --standalone --nproc_per_node=8 chat_sft.py   # full SFT (~2 epochs, ~1B tokens)
python chat_cli.py --ckpt log_sft/model_step_01999.pt  # chat with it
```

Result (2000 steps, LR 3e-5): train/val loss plateau at **~1.73**. The plateau is expected — SFT loss floors at the entropy of free-form assistant text; it can't reach 0 like the single-batch overfit. The model replies coherently in chat format and stops on `<|assistant_end|>` (it hallucinates freely — a 124M-scale limit, not an SFT bug).

## RL (GRPO) and why we move to a 1B model

We implemented GRPO-style RL on GSM8K (group-relative advantages, on-policy, no critic/KL) to push math reasoning. It ran correctly end-to-end — but **on the 124M base it didn't move the needle**, and that turned out to be a *scale* problem, not a pipeline problem:

- **SmolTalk-only SFT:** GSM8K pass@8 ≈ **6%**.
- **SFT with GSM8K blended in (×4 epochs):** pass@8 ≈ **5%** — the model learned the `#### <answer>` format but not how to *solve* the problems.
- **RL on top:** eval pass@1 stayed flat (~0.01) over 1000s of steps.

RL only *amplifies* ability the base already has, and at 124M there's almost none to amplify (pass@1 ≈ 0.5–1%). SFT and RL teach **format and consistency**, not raw reasoning — that capability is set in **pretraining**, and it's an emergent property of scale. The same reason HellaSwag stayed near chance is why GSM8K stays near zero: a 124M model is simply below the capability threshold for these tasks.

**So the bottleneck is model size, not the recipe** — which motivates scaling the (now-modernized) architecture to **~1B (`--depth 26`)**, trained Chinchilla-optimally on 20B tokens, where reasoning benchmarks begin to emerge.

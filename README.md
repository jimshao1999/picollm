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

## Results (2026-07-04)

Full run: 19073 steps, 10B tokens, ~0.5M tokens/step.

| Metric | picollm (this run) | GPT-2 (124M) | build-nanogpt repro |
|--------|--------------------|--------------|---------------------|
| Final val loss (FineWeb-Edu) | **3.257** | – | ~3.28 |
| HellaSwag acc_norm | **0.281** | 0.2955 | ~0.305 |

Training is healthy: init loss 10.955 (≈ ln(50304)), and the final validation loss is on par with (slightly better than) the reference. HellaSwag lands ~1–2 points below GPT-2 / the reference reproduction. The LM loss matches the reference while HellaSwag is slightly lower, the remaining gap is to be investigated.

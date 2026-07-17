import os

import torch
import torch.nn.functional as F
from chat_cli import device, load_model
from tasks.gsm8k import GSM8K
from tokenizer import ChatTokenizer


@torch.no_grad()
def sample_completion(
    model,
    tok,
    prompt_ids,
    device,
    device_type,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
):
    """Sample a completion from a prompt."""
    assistant_end = tok.encode_special("<|assistant_end|>")
    ids = list(prompt_ids)
    prefix_len = len(ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx = x[:, -model.config.block_size :]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(idx)
        logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, vocab_size)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tid = nxt.item()
        ids.append(tid)
        x = torch.cat((x, nxt), dim=-1)
        if tid == assistant_end:
            break
    return ids, prefix_len


@torch.no_grad()
def get_rollout(
    model,
    tok,
    task,
    example,
    device,
    device_type,
    k=16,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    strict_reward=False,
):
    """Get a rollout of k completions from a prompt."""
    assistant_start = tok.encode_special("<|assistant_start|>")
    assistant_end = tok.encode_special("<|assistant_end|>")

    question = example["messages"][0]["content"]
    prompt_ids, _ = tok.render_conversation(
        {"messages": [{"role": "user", "content": question}]}, max_tokens=10**9
    )
    prompt_ids = prompt_ids + [assistant_start]

    ## 1. sample k completions, score each
    seqs, rewards = [], []
    for _ in range(k):
        ids, prefix_len = sample_completion(
            model,
            tok,
            prompt_ids,
            device,
            device_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        gen_text = tok.tokenizer.decode([t for t in ids[prefix_len:] if t < 50257])
        r = task.evaluate(example, gen_text, lenient=not strict_reward)
        rewards.append(float(r))
        seqs.append((ids, prefix_len))

    rewards = torch.tensor(rewards, dtype=torch.float, device=device)

    ## 2. advantage = reward - group mean; if all rewards equal, advantage = 0 -> skip
    if rewards.std() == 0:
        return None
    advantages = rewards - rewards.mean()

    ## 3. pad to common length, builds (inputs, targets) with prompt+pad masked to -1
    max_len = max(len(fid) for fid, _ in seqs)
    ids_batch, mask_batch = [], []
    for ids, prefix_len in seqs:
        pad = max_len - len(ids)
        ids_batch.append(ids + [assistant_end] * pad)
        mask_batch.append([0] * prefix_len + [1] * (len(ids) - prefix_len) + [0] * pad)

    ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
    mask = torch.tensor(mask_batch, dtype=torch.long, device=device)

    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask[:, 1:] == 0] = -1

    return {
        "inputs": inputs,
        "targets": targets,
        "advantages": advantages,
        "rewards": rewards,
    }


@torch.no_grad()
def evaluate_passk(
    model,
    tok,
    task,
    device,
    device_type,
    n=50,
    k=8,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    lenient=True,
):
    """Honest progress metric: pass@1 / pass@k on the test split (uses the LIVE model)."""
    assistant_start = tok.encode_special("<|assistant_start|>")
    was_training = model.training
    model.eval()
    n = min(n, len(task))
    correct, total, solved = 0, 0, 0
    for i in range(n):
        ex = task[i]
        question = ex["messages"][0]["content"]
        prompt_ids, _ = tok.render_conversation(
            {"messages": [{"role": "user", "content": question}]}, max_tokens=10**9
        )
        prompt_ids = prompt_ids + [assistant_start]
        flags = []
        for _ in range(k):
            ids, prefix_len = sample_completion(
                model,
                tok,
                prompt_ids,
                device,
                device_type,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            gen = tok.tokenizer.decode([t for t in ids[prefix_len:] if t < 50257])
            flags.append(task.evaluate(ex, gen, lenient=lenient))
        correct += sum(flags)
        total += k
        solved += 1 if any(flags) else 0
    if was_training:
        model.train()
    return correct / max(total, 1), solved / max(n, 1)


def rl_train(
    ckpt_path,
    steps=3000,
    k=16,
    examples_per_step=8,
    lr=1e-6,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    strict_reward=False,
    grad_clip=1.0,
    eval_every=250,
    eval_n=50,
    eval_k=8,
    ckpt_every=1000,
    log_dir="log_rl",
):
    model = load_model(ckpt_path)
    tok = ChatTokenizer()
    task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    device_type = "cuda" if device.startswith("cuda") else device
    optimizer = model.configure_optimizers(
        weight_decay=0.0, learning_rate=lr, device=device
    )
    os.makedirs(log_dir, exist_ok=True)

    cursor = 0
    for step in range(steps):
        ## periodic eval on the test split (honest progress metric)
        if step % eval_every == 0:
            p1, pk = evaluate_passk(
                model,
                tok,
                val_task,
                device,
                device_type,
                n=eval_n,
                k=eval_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                lenient=not strict_reward,
            )
            print(f"  [eval step={step}] pass@1={p1:.4f} pass@{eval_k}={pk:.4f}")

        ## 1. gather a few NON-degenerate groups
        model.eval()
        rollouts = []
        degenerate = 0
        while len(rollouts) < examples_per_step:
            out = get_rollout(
                model,
                tok,
                task,
                task[cursor % len(task)],
                device,
                device_type,
                k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                strict_reward=strict_reward,
            )
            cursor += 1
            if out is not None:
                rollouts.append(out)
            else:
                degenerate += 1

        ## 2. policy gradient update
        model.train()
        optimizer.zero_grad()
        # token-level normalization across all rollouts in this step
        num_valid = max(sum(int((r["targets"] != -1).sum()) for r in rollouts), 1)
        reward_sum, sample_count = 0.0, 0
        for r in rollouts:
            inputs, targets, advantages = r["inputs"], r["targets"], r["advantages"]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # model returns (logits, loss); with reduction="none", loss is per-token
                _, loss = model(inputs, targets, loss_reduction="none")
                logp = -loss.view_as(inputs)  # (k, T) per-token log-probs
            pg_obj = (logp * advantages.unsqueeze(-1)).sum() / num_valid  # scalar
            (-pg_obj).backward()  # accumulate grads across rollouts
            reward_sum += r["rewards"].sum().item()
            sample_count += r["rewards"].numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # NOTE: reward is averaged over TRAINED (non-degenerate) groups only, so it
        # reads high vs true accuracy; watch the trend, and 'degenerate' for sparsity.
        mean_reward = reward_sum / sample_count
        print(
            f"step={step:4d} | reward={mean_reward:.4f} | groups={len(rollouts)} | "
            f"degenerate_skipped={degenerate} | valid_tokens={num_valid}"
        )

        ## periodic checkpoint (and always at the last step)
        if step > 0 and (step % ckpt_every == 0 or step == steps - 1):
            path = os.path.join(log_dir, f"model_step_{step}.pt")
            torch.save(
                {"model": model.state_dict(), "config": model.config, "step": step},
                path,
            )
            print(f"saved checkpoint to {path}")


def run_verify(
    ckpt_path,
    k=16,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    strict_reward=False,
    scan=30,
):
    """Inspect ONE non-degenerate rollout (no training) to sanity-check mechanics."""
    model = load_model(ckpt_path)
    tok = ChatTokenizer()
    task = GSM8K(subset="main", split="train")
    device_type = "cuda" if device.startswith("cuda") else device

    for i in range(scan):
        out = get_rollout(
            model,
            tok,
            task,
            task[i],
            device,
            device_type,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            strict_reward=strict_reward,
        )
        if out is None:
            print(f"[{i}] degenerate group (all rewards equal) — skipping")
            continue
        rewards, adv = out["rewards"], out["advantages"]
        print(f"\n[{i}] non-degenerate group found")
        print("  rewards:   ", [round(r, 1) for r in rewards.tolist()])
        print("  advantages:", [round(a, 3) for a in adv.tolist()])
        print("  adv.sum() ~0?", round(float(adv.sum()), 5))
        print("  inputs :", tuple(out["inputs"].shape))
        print("  targets:", tuple(out["targets"].shape))
        print("  trained tokens (targets != -1):", int((out["targets"] != -1).sum()))
        return
    print(
        "no non-degenerate group in first %d examples — rewards too sparse; "
        "bump k or the scan range" % scan
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["verify", "train"],
        default="verify",
        help="verify = inspect one rollout (no training); train = run the PG loop",
    )
    p.add_argument("--ckpt", default="log-sft-modern/model_step_1999.pt")
    p.add_argument(
        "--dummy",
        action="store_true",
        help="tiny fast settings to smoke-test the train loop end-to-end",
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--examples-per-step", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--strict", action="store_true", help="strict reward (require '#### N')"
    )
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-dir", default="log_rl")
    args = p.parse_args()

    if args.dummy:
        # fast end-to-end smoke test: exercises rollout + update + eval + checkpoint
        args.mode = "train"
        args.steps, args.k, args.examples_per_step, args.max_new_tokens = 2, 4, 1, 64
        args.eval_every, args.eval_n, args.ckpt_every, args.log_dir = (
            1,
            2,
            1,
            "log_rl_dummy",
        )
        print("[dummy] smoke run: tiny steps/k/eval/ckpt to exercise every code path")

    if args.mode == "verify":
        run_verify(
            args.ckpt,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            strict_reward=args.strict,
        )
    else:
        rl_train(
            args.ckpt,
            steps=args.steps,
            k=args.k,
            examples_per_step=args.examples_per_step,
            lr=args.lr,
            max_new_tokens=args.max_new_tokens,
            strict_reward=args.strict,
            eval_every=args.eval_every,
            eval_n=args.eval_n,
            ckpt_every=args.ckpt_every,
            log_dir=args.log_dir,
        )

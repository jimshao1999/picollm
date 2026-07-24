import contextlib
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from chat_cli import device, load_model
from tasks import make_task
from tokenizer import ChatTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP


@torch.no_grad()
def sample_completion_batched(
    model,
    tok,
    prompt_ids,
    device,
    device_type,
    k=16,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
):
    """Sample k completions for a prompt."""
    assistant_end = tok.encode_special("<|assistant_end|>")
    ids = list(prompt_ids)
    prefix_len = len(ids)
    x = torch.tensor([ids] * k, dtype=torch.long, device=device)
    finished = torch.zeros(k, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        idx = x[:, -model.config.block_size :]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(idx)
        logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, vocab_size)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, assistant_end), nxt)
        x = torch.cat((x, nxt.unsqueeze(1)), dim=1)
        finished = finished | (nxt == assistant_end)
        if finished.all():
            break
    seqs = []
    for row in x[:, prefix_len:].tolist():
        if assistant_end in row:
            row = row[: row.index(assistant_end) + 1]
        seqs.append(ids + row)
    return seqs, prefix_len


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
    seq_ids, prefix_len = sample_completion_batched(
        model,
        tok,
        prompt_ids,
        device,
        device_type,
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    for ids in seq_ids:
        gen_text = tok.tokenizer.decode([t for t in ids[prefix_len:] if t < 50257])
        r = task.evaluate(example, gen_text, lenient=not strict_reward)
        rewards.append(float(r))
        seqs.append((ids, prefix_len))

    rewards = torch.tensor(rewards, dtype=torch.float, device=device)

    ## 2. advantage = reward - group mean.
    ## degenerate groups (all rewards equal) naturally get advantages=0 -> zero gradient
    ## (harmless). we KEEP them so per-step work is fixed (no DDP straggler from resampling).
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
    ddp=False,
    world=1,
    rank=0,
):
    """Honest progress metric: pass@1 / pass@k on the test split (uses the LIVE model)."""
    assistant_start = tok.encode_special("<|assistant_start|>")
    was_training = model.training
    model.eval()
    n = min(n, len(task))
    correct, total, solved = 0, 0, 0
    for i in range(rank, n, world):
        ex = task[i]
        question = ex["messages"][0]["content"]
        prompt_ids, _ = tok.render_conversation(
            {"messages": [{"role": "user", "content": question}]}, max_tokens=10**9
        )
        prompt_ids = prompt_ids + [assistant_start]
        flags = []

        seq_ids, prefix_len = sample_completion_batched(
            model,
            tok,
            prompt_ids,
            device,
            device_type,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        for ids in seq_ids:
            gen = tok.tokenizer.decode([t for t in ids[prefix_len:] if t < 50257])
            flags.append(task.evaluate(ex, gen, lenient=lenient))
        correct += sum(flags)
        total += k
        solved += 1 if any(flags) else 0
    if ddp:
        t = torch.tensor([correct, total, solved], dtype=torch.float, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct, total, solved = t.tolist()
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
    kl_coef=0.0,  # KL-to-reference (SFT) penalty; 0 = off. anchors policy, prevents collapse
    warmup_steps=20,  # linear LR warmup — the first RL steps are the most destructive
    task_name="gsm8k",
    eval_every=250,
    eval_n=50,
    eval_k=8,
    ckpt_every=1000,
    log_dir="log_rl",
):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ.get("RANK"))
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world = int(os.environ.get("WORLD_SIZE"))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world = 1

    master = rank == 0
    raw_model = load_model(ckpt_path)
    model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model
    ref_model = None
    if kl_coef > 0:
        # frozen copy of the SFT model = the KL anchor (keeps the policy from drifting
        # into degenerate/collapsed outputs). no grad, no optimizer.
        ref_model = load_model(ckpt_path)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
    tok = ChatTokenizer()
    task = make_task(task_name, "train")
    val_task = make_task(task_name, "test")
    device_type = "cuda" if device.startswith("cuda") else device
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.0, learning_rate=lr, device=device
    )
    log_file = os.path.join(log_dir, "log.txt")
    if master:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "w") as f:  # fresh log for this run
            pass

    examples_per_rank = max(examples_per_step // world, 1)
    cursor = rank
    for step in range(steps):
        ## linear LR warmup: ramp 0 -> lr over the first `warmup_steps` steps so the
        ## fragile early updates can't destroy the SFT policy (the hard-dip failure mode)
        cur_lr = lr * min(1.0, (step + 1) / max(warmup_steps, 1))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        ## periodic eval on the test split (honest progress metric)
        if step % eval_every == 0:
            p1, pk = evaluate_passk(
                raw_model,
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
                ddp=ddp,
                world=world,
                rank=rank,
            )
            if master:
                print(f"[eval step={step}] pass@1={p1:.4f} pass@{eval_k}={pk:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} eval_pass@1 {p1:.4f}\n")
                    f.write(f"{step} eval_pass@{eval_k} {pk:.4f}\n")

        ## 1. gather a FIXED number of groups per rank (no skipping -> deterministic
        ## per-step cost, no DDP straggler). degenerate groups just contribute 0 gradient.
        model.eval()
        rollouts = []
        for _ in range(examples_per_rank):
            out = get_rollout(
                raw_model,
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
            cursor += world
            rollouts.append(out)
        degenerate = sum(1 for r in rollouts if float(r["rewards"].std()) == 0)

        ## 2. policy gradient update
        model.train()
        optimizer.zero_grad()
        # token-level normalization across all rollouts in this step
        num_valid = max(sum(int((r["targets"] != -1).sum()) for r in rollouts), 1)
        reward_sum, sample_count, kl_running = 0.0, 0, 0.0
        for j, r in enumerate(rollouts):
            is_last = j == len(rollouts) - 1
            sync_ctx = (
                contextlib.nullcontext() if (is_last or not ddp) else model.no_sync()
            )
            with sync_ctx:
                inputs, targets, advantages = r["inputs"], r["targets"], r["advantages"]
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    # model returns (logits, loss); with reduction="none", loss is per-token
                    _, loss = model(inputs, targets, loss_reduction="none")
                    logp = -loss.view_as(inputs)  # (k, T) per-token log-probs
                    kl_term = logp.new_zeros(())
                    if ref_model is not None:
                        with torch.no_grad():
                            _, loss_ref = ref_model(inputs, targets, loss_reduction="none")
                        logp_ref = -loss_ref.view_as(inputs)
                        # k3 KL estimator (>=0); auto-0 at masked tokens (both logp==0 there)
                        log_ratio = torch.clamp(logp_ref - logp, -10.0, 10.0)
                        kl = torch.exp(log_ratio) - log_ratio - 1.0
                        kl_term = kl.sum() / num_valid
                pg_obj = (logp * advantages.unsqueeze(-1)).sum() / num_valid  # scalar
                (-pg_obj + kl_coef * kl_term).backward()  # PG + KL anchor
            reward_sum += r["rewards"].sum().item()
            sample_count += r["rewards"].numel()
            kl_running += float(kl_term.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # NOTE: reward is over this step's groups (incl. degenerate=0-reward ones).
        # watch eval_pass@1 (the honest metric) and 'kl' (should stay small, not blow up).
        if master:
            mean_reward = reward_sum / sample_count
            mean_kl = kl_running / max(len(rollouts), 1)
            print(
                f"step={step:4d} | reward={mean_reward:.4f} | kl={mean_kl:.4f} | "
                f"groups={len(rollouts)} | degenerate={degenerate} | valid_tokens={num_valid}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} reward {mean_reward:.4f} kl {mean_kl:.4f} degenerate {degenerate}\n")

        ## periodic checkpoint (and always at the last step)
        if master and step > 0 and (step % ckpt_every == 0 or step == steps - 1):
            path = os.path.join(log_dir, f"model_step_{step}.pt")
            torch.save(
                {"model": raw_model.state_dict(), "config": raw_model.config, "step": step},
                path,
            )
            print(f"saved checkpoint to {path}")

    if ddp:
        dist.destroy_process_group()


def run_verify(
    ckpt_path,
    k=16,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    strict_reward=False,
    scan=30,
    task_name="gsm8k",
):
    """Inspect ONE non-degenerate rollout (no training) to sanity-check mechanics."""
    model = load_model(ckpt_path)
    tok = ChatTokenizer()
    task = make_task(task_name, "train")
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
        if float(out["rewards"].std()) == 0:
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
    p.add_argument("--lr", type=float, default=1e-5)  # 1e-6 was too low; RL needs a bigger step
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--strict", action="store_true", help="strict reward (require '#### N')"
    )
    p.add_argument("--kl-coef", type=float, default=0.1,
                   help="KL-to-SFT-reference penalty (anchors policy; 0 = off)")
    p.add_argument("--warmup-steps", type=int, default=20,
                   help="linear LR warmup steps (protects the fragile early RL updates)")
    p.add_argument("--task", default="gsm8k",
                   choices=["gsm8k", "arithmetic", "svamp", "multiarith"],
                   help="which verifiable-reward task to RL on (easier fallbacks if GSM8K is too hard)")
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
            task_name=args.task,
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
            kl_coef=args.kl_coef,
            warmup_steps=args.warmup_steps,
            task_name=args.task,
            eval_every=args.eval_every,
            eval_n=args.eval_n,
            ckpt_every=args.ckpt_every,
            log_dir=args.log_dir,
        )

#!/bin/bash
# Auto-resume d26 pretraining across NCCL flakes.
# Relaunches from the latest checkpoint every time the run dies, until it finishes.
set -u

LOG_DIR=log_d26
MAX_STEPS=38000

# fail fast on a stuck watchdog (default ~480s+hang) so we restart sooner, not ~2h later
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export TORCH_NCCL_DUMP_ON_TIMEOUT=0

while true; do
    # latest checkpoint = highest step number on disk
    LATEST=$(ls -1 "$LOG_DIR"/model_step_*.pt 2>/dev/null \
        | sed 's/.*model_step_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)

    RESUME=""
    if [ -n "${LATEST:-}" ]; then
        if [ "$LATEST" -ge "$((MAX_STEPS - 1))" ]; then
            echo "[wrapper] latest checkpoint $LATEST >= max steps — done."
            break
        fi
        RESUME="--resume $LOG_DIR/model_step_${LATEST}.pt"
        echo "[wrapper] resuming from step $LATEST"
    else
        echo "[wrapper] no checkpoint found — fresh start"
    fi

    torchrun --standalone --nproc_per_node=4 base_train.py \
        --depth 26 --device-batch-size 16 --data-dir edu_fineweb20B \
        --max-steps "$MAX_STEPS" --ckpt-every 3000 --log-dir "$LOG_DIR" $RESUME

    CODE=$?
    if [ "$CODE" -eq 0 ]; then
        echo "[wrapper] training finished cleanly."
        break
    fi
    echo "[wrapper] crashed (exit $CODE). Restarting from latest checkpoint in 30s..."
    sleep 30
done

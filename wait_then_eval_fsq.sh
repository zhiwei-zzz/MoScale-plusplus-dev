#!/usr/bin/env bash
# Wait for skelvq_fsq training to finish, then run the 20-repeat eval.
# Replaces the (now obsolete) watchdog_fsq_then_fsq_reg.sh — no chained reg run.
set -u
cd /media/z/SSD4T-N-Ubuntu/research_idea/skel_moscale/SALAD

TRAIN_PID=3672884
LOG=training_logs/wait_then_eval_fsq.log

echo "[$(date)] waiting for PID $TRAIN_PID (skelvq_fsq training) to exit" > "$LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] training PID $TRAIN_PID exited; starting 20-repeat eval" >> "$LOG"

# Give the filesystem a moment to flush the final checkpoint.
sleep 10

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate moscale 2>/dev/null || true

python test_skel_vq.py \
    --name skelvq_fsq \
    --quantizer_type fsq \
    --fsq_levels 8 \
    > training_logs/skelvq_fsq_finaleval.stdout \
    2> training_logs/skelvq_fsq_finaleval.stderr

echo "[$(date)] eval completed; see checkpoints/t2m/skelvq_fsq/eval/eval.log" >> "$LOG"

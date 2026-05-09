#!/usr/bin/env bash
# After skelvq_fsq training ends:
#   1) 20-repeat test_skel_vq.py for skelvq_fsq baseline
#   2) Launch skelvq_fsq_reg training (FSQ + entropy regularization)
set -uo pipefail
cd "$(dirname "$0")"
LOG=training_logs/watchdog_fsq_then_reg.log
echo "[$(date)] watchdog up. Waiting for skelvq_fsq training to exit..." | tee -a "$LOG"

while pgrep -f "train_skel_vq.py.*skelvq_fsq[^_]" > /dev/null; do
    sleep 60
done

echo "[$(date)] FSQ training exited. Confirming checkpoints..." | tee -a "$LOG"
sleep 10
if [ ! -f checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar ]; then
    echo "[$(date)] ERROR: net_best_fid.tar not found; aborting." | tee -a "$LOG"
    exit 1
fi

source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate moscale

echo "[$(date)] Running 20-rep eval on skelvq_fsq..." | tee -a "$LOG"
python test_skel_vq.py --name skelvq_fsq --quantizer_type fsq --fsq_levels 8 \
    > training_logs/skelvq_fsq_finaleval.stdout 2> training_logs/skelvq_fsq_finaleval.stderr
echo "[$(date)] FSQ eval done. eval.log:" | tee -a "$LOG"
cat checkpoints/t2m/skelvq_fsq/eval/eval.log | tee -a "$LOG"

echo "[$(date)] Launching skelvq_fsq_reg training (entropy regularizer)..." | tee -a "$LOG"
nohup python train_skel_vq.py \
    --name skelvq_fsq_reg \
    --quantizer_type fsq \
    --fsq_levels 8 \
    --fsq_entropy_weight 0.1 \
    --fsq_inv_temperature 20.0 \
    > training_logs/skelvq_fsq_reg.stdout 2> training_logs/skelvq_fsq_reg.stderr < /dev/null &
disown
PID=$!
echo "[$(date)] skelvq_fsq_reg launched, PID=$PID" | tee -a "$LOG"

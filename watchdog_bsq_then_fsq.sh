#!/usr/bin/env bash
# Wait for current skelvq_bsq training to finish, then:
#   1) Run 20-repeat test_skel_vq.py on skelvq_bsq for the publishable BSQ number.
#   2) Launch skelvq_fsq from-scratch training.
set -uo pipefail
cd "$(dirname "$0")"
LOG=training_logs/watchdog_bsq_then_fsq.log
echo "[$(date)] watchdog up. Waiting for skelvq_bsq training to exit..." | tee -a "$LOG"

# Block until no train_skel_vq process running.
while pgrep -f "train_skel_vq.py.*skelvq_bsq" > /dev/null; do
    sleep 60
done

echo "[$(date)] BSQ training exited. Confirming checkpoints exist..." | tee -a "$LOG"
sleep 10
if [ ! -f checkpoints/t2m/skelvq_bsq/model/net_best_fid.tar ]; then
    echo "[$(date)] ERROR: net_best_fid.tar not found; aborting." | tee -a "$LOG"
    exit 1
fi

source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate moscale

echo "[$(date)] Running 20-repeat eval on skelvq_bsq..." | tee -a "$LOG"
python test_skel_vq.py --name skelvq_bsq --quantizer_type bsq >> training_logs/skelvq_bsq_finaleval.stdout 2>> training_logs/skelvq_bsq_finaleval.stderr
echo "[$(date)] BSQ eval finished. eval.log:" | tee -a "$LOG"
cat checkpoints/t2m/skelvq_bsq/eval/eval.log | tee -a "$LOG"

echo "[$(date)] Launching skelvq_fsq from-scratch training..." | tee -a "$LOG"
nohup python train_skel_vq.py --name skelvq_fsq --quantizer_type fsq --fsq_levels 8 \
    > training_logs/skelvq_fsq.stdout 2> training_logs/skelvq_fsq.stderr < /dev/null &
disown
echo "[$(date)] skelvq_fsq launched, PID=$!" | tee -a "$LOG"

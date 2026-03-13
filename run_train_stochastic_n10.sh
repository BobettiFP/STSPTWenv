#!/bin/bash
set -e

# Stochastic N=10: 3 difficulties × 10 delay weights × 3 models = 90 training runs
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /home/ramy/Desktop/Temp/PIP-constraint/venv/bin/activate
cd /home/ramy/Desktop/Temp/PIP-constraint/POMO+PIP

# Quick smoke-test: ~10 episodes per epoch, 2 epochs (override for full runs)
COMMON_ARGS="--problem STSPTW --problem_size 10 --pomo_size 10 \
  --epochs 2 --train_episodes 10 --train_batch_size 4 \
  --val_episodes 10 --validation_batch_size 10 \
  --validation_interval 1 --model_save_interval 1"

DELAY_WEIGHTS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for hardness in easy medium hard; do
  for dw in $DELAY_WEIGHTS; do
    echo "=============================="
    echo "=== STSPTW N=10 $hardness delay_weight=$dw POMO $(date) ==="
    echo "=============================="
    python train.py $COMMON_ARGS --hardness $hardness --delay_scale $dw
  done
done

for hardness in easy medium hard; do
  for dw in $DELAY_WEIGHTS; do
    echo "=============================="
    echo "=== STSPTW N=10 $hardness delay_weight=$dw POMO* $(date) ==="
    echo "=============================="
    python train.py $COMMON_ARGS --hardness $hardness --delay_scale $dw --pomo_start True
  done
done

for hardness in easy medium hard; do
  for dw in $DELAY_WEIGHTS; do
    echo "=============================="
    echo "=== STSPTW N=10 $hardness delay_weight=$dw POMO*+PIP $(date) ==="
    echo "=============================="
    python train.py $COMMON_ARGS --hardness $hardness --delay_scale $dw --pomo_start True --generate_PI_mask
  done
done

echo "=============================="
echo "=== ALL STOCHASTIC N=10 DONE $(date) ==="
echo "=============================="

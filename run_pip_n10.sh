#!/bin/bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /home/ramy/Desktop/Temp/PIP-constraint/venv/bin/activate
cd /home/ramy/Desktop/Temp/PIP-constraint/POMO+PIP

COMMON_ARGS="--problem TSPTW --problem_size 10 --pomo_size 10 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 \
  --validation_interval 500 --model_save_interval 50"

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO*+PIP $hardness  $(date) ==="
  echo "=============================="
  python train.py $COMMON_ARGS --hardness $hardness --generate_PI_mask
done

echo "=============================="
echo "=== ALL PIP DONE  $(date) ==="
echo "=============================="

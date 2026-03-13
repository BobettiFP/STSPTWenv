#!/bin/bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /home/ramy/Desktop/Temp/PIP-constraint/venv/bin/activate
cd /home/ramy/Desktop/Temp/STSPTWenv/POMO+PIP

COMMON_ARGS="--problem TSPTW --problem_size 10 --pomo_size 10 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 \
  --validation_interval 500 --model_save_interval 1000"

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO (POMO) $hardness  $(date) ==="
  echo "=============================="
  python train.py $COMMON_ARGS --hardness $hardness --model_type POMO
done

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO* (POMO_STAR) $hardness  $(date) ==="
  echo "=============================="
  python train.py $COMMON_ARGS --hardness $hardness --model_type POMO_STAR
done

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO*+PIP (POMO_STAR_PIP) $hardness  $(date) ==="
  echo "=============================="
  python train.py $COMMON_ARGS --hardness $hardness --model_type POMO_STAR_PIP
done

echo "=============================="
echo "=== ALL DONE  $(date) ==="
echo "=============================="

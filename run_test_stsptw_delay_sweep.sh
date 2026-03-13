#!/bin/bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /home/ramy/Desktop/Temp/PIP-constraint/venv/bin/activate

# Sweep STSPTW delay_scale from 0.00 to 1.00 in 0.01 steps,
# using TSPTW-trained checkpoints (easy/medium/hard) and keeping other
# test settings at defaults.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/POMO+PIP"

PROBLEM="STSPTW"

# You can override checkpoints by running, for example:
#   CHECKPOINT_EASY=/path/to/easy.pt CHECKPOINT_MEDIUM=/path/to/medium.pt CHECKPOINT_HARD=/path/to/hard.pt ./run_test_stsptw_delay_sweep.sh
# If not provided, we fall back to the pretrained layout:
#   pretrained/TSPTW/tsptw50_{hardness}/POMO_star_PIP-D/epoch-10000.pt
CHECKPOINT_EASY="${CHECKPOINT_EASY:-pretrained/TSPTW/tsptw50_easy/POMO_star_PIP-D/epoch-10000.pt}"
CHECKPOINT_MEDIUM="${CHECKPOINT_MEDIUM:-pretrained/TSPTW/tsptw50_medium/POMO_star_PIP-D/epoch-10000.pt}"
CHECKPOINT_HARD="${CHECKPOINT_HARD:-pretrained/TSPTW/tsptw50_hard/POMO_star_PIP-D/epoch-10000.pt}"

LOG_DIR="${ROOT_DIR}/logs/stsptw_delay_sweep"
mkdir -p "${LOG_DIR}"

INDEX_FILE="${LOG_DIR}/index.csv"
if [ ! -f "${INDEX_FILE}" ]; then
  echo "hardness,delay_scale,log_file" > "${INDEX_FILE}"
fi

for hardness in easy medium hard; do
  case "${hardness}" in
    easy)   CHECKPOINT="${CHECKPOINT_EASY}" ;;
    medium) CHECKPOINT="${CHECKPOINT_MEDIUM}" ;;
    hard)   CHECKPOINT="${CHECKPOINT_HARD}" ;;
  esac

  for delay in $(seq 0 0.01 1.0); do
    delay_str=$(printf "%.2f" "${delay}")
    echo "=============================="
    echo "=== STSPTW hardness=${hardness}, delay_scale=${delay_str} ==="
    echo "=============================="

    log_file="${LOG_DIR}/${hardness}_delay_${delay_str//./_}.log"

    python test.py \
      --problem "${PROBLEM}" \
      --hardness "${hardness}" \
      --delay_scale "${delay_str}" \
      --checkpoint "${CHECKPOINT}" \
      "$@" | tee "${log_file}"

    echo "${hardness},${delay_str},${log_file}" >> "${INDEX_FILE}"
  done
done

echo "Delay sweep finished. Logs and index: ${LOG_DIR}"


#!/usr/bin/env bash
set -euo pipefail

# Run this script on the lab server inside ARM-demo repository.
# Example:
#   bash scripts/server_pipeline.sh data/raw 200 100 80 world_model

DATA_DIR="${1:-data/raw}"
NUM_EPISODES="${2:-200}"
MAX_STEPS="${3:-100}"
EPOCHS="${4:-80}"
SAVE_NAME="${5:-world_model}"
IMAGE_SIZE="${6:-64}"
ROLLOUT_HORIZON="${7:-20}"

echo "[1/4] Install dependencies"
python -m pip install -r requirements.txt

echo "[2/4] Collect ManiSkill data -> ${DATA_DIR}"
python scripts/collect_data.py \
  --output_dir "${DATA_DIR}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --image_size "${IMAGE_SIZE}"

echo "[3/4] Train world model"
python scripts/train.py \
  --config configs/default.yaml \
  --data_dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --save_name "${SAVE_NAME}"

echo "[4/4] Generate rollout video"
python scripts/visualize_rollout.py \
  --checkpoint "checkpoints/${SAVE_NAME}_best.pt" \
  --data_dir "${DATA_DIR}" \
  --horizon "${ROLLOUT_HORIZON}"

echo "Done. Check outputs/rollout_videos/"

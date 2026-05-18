#!/usr/bin/env bash
set -eu
set -o pipefail

cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

if ! python -c "import mani_skill, torch, requests; print('deps_ok')"; then
  echo "Installing missing dependencies in remote base env..."
  python -m pip install -r requirements.txt
  python -c "import mani_skill, torch, requests; print('deps_ok_after_install')"
fi

# Train world model + risk estimator for 3-second horizon safety prediction.
python scripts/train_world_model_safety.py \
  --episodes 60 \
  --max_steps 90 \
  --horizon_steps 30 \
  --epochs 8 \
  --batch_size 64 \
  --alpha_risk 1.0 \
  --forbidden_x 0.0 \
  --forbidden_y 0.0 \
  --forbidden_radius 0.08 \
  --forbidden_z_max 0.18 \
  --output_dir outputs/safety_wm_v1 \
  --checkpoint checkpoints/safety_wm_v1.pt

# Demo: Predict future -> estimate risk -> stop-or-allow (no replanning).
python scripts/run_world_model_safety_demo.py \
  --checkpoint checkpoints/safety_wm_v1.pt \
  --output_dir outputs/safety_demo_v1 \
  --episodes 4 \
  --risk_threshold 0.52 \
  --horizon_steps 30 \
  --n_samples 5

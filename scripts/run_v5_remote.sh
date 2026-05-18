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

# Re-train grasp visual actor and export detailed component loss curves.
python scripts/train_grasp_visual_skill.py \
  --total_steps 18000 \
  --eval_episodes 8 \
  --output_dir outputs/grasp_visual_v4_rotcal \
  --checkpoint checkpoints/grasp_visual_actor_v4_rotcal.pt

# Run full pipeline with upgraded dual-view and new grasp checkpoint.
python scripts/run_visual_calibrated_two_skill.py \
  --episodes 8 \
  --model gpt-4o-mini \
  --grasp_checkpoint checkpoints/grasp_visual_actor_v4_rotcal.pt \
  --place_checkpoint checkpoints/move_to_visual_actor_v3.pt \
  --success_radius 0.08 \
  --output_dir outputs/llm_full_v9_rotcal \
  --output_json outputs/llm_full_v9_rotcal/result.json \
  --api_base_url https://sg.uiuiapi.com \
  --api_key sk-0ZBuMldpARlqrmOGiKcFHG0WIcS4cAjfN2fKTGfUxClu9vDF

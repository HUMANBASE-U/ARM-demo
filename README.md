# ARM Demo: Action-Conditioned World Model

This project builds an end-to-end world model demo for ManiSkill `PickCube-v1`.

Core task:

- Input: current frame `frame_t` and current action `action_t`
- Predict: next frame `frame_{t+1}`

Model modules:

1. Encoder (CNN): image -> latent `z_t`
2. Dynamics (MLP residual): `(z_t, a_t)` -> `z_hat_{t+1}`
3. Decoder (deconv CNN): latent -> image

## 1) Environment setup (run on lab server)

Use Python 3.10+ with CUDA-capable PyTorch.

```bash
pip install -r requirements.txt
```

### Recommended remote workflow

```bash
# local Windows: push your latest code
git add .
git commit -m "update world model pipeline"
git push

# on lab server
ssh ra-ugrad@trinity
cd ~/Ajax/ARM-demo
git pull
```

## 2) Collect ManiSkill data

```bash
python scripts/collect_data.py --output_dir data/raw --num_episodes 200 --max_steps 100 --image_size 64
```

This saves per-episode `.npz` files with:

- `frames`: `(T+1, H, W, 3)` uint8
- `actions`: `(T, A)` float32
- `rewards`: `(T,)` float32
- `dones`: `(T,)` float32

## 3) Train world model

```bash
python scripts/train.py --config configs/default.yaml
```

Outputs:

- checkpoints in `checkpoints/`
- visual samples in `outputs/recon_samples/`

## 4) Generate rollout prediction video

```bash
python scripts/visualize_rollout.py --checkpoint checkpoints/world_model_best.pt --data_dir data/raw --horizon 20
```

Output video:

- `outputs/rollout_videos/rollout_*.mp4`

## 5) Optional quick smoke test (no ManiSkill needed)

If you want to test full pipeline quickly before collecting ManiSkill data:

```bash
python scripts/collect_data.py --use_dummy_data --output_dir data/raw_dummy --num_episodes 50 --max_steps 40 --image_size 64
python scripts/train.py --config configs/default.yaml --data_dir data/raw_dummy --epochs 5 --save_name dummy_quick
python scripts/visualize_rollout.py --checkpoint checkpoints/dummy_quick_best.pt --data_dir data/raw_dummy --horizon 15
```

## 6) Automated server run from local machine

If direct `ssh` is interactive in your terminal, you can run the full server pipeline via Paramiko:

```bash
set LAB_SERVER_PASSWORD=your_password
python scripts/remote_run.py --host trinity --username ra-ugrad --remote_root ~/Ajax/ARM-demo --num_episodes 30 --max_steps 60 --epochs 5 --save_name remote_quick
```

This uploads project code, runs collection/training/rollout on server, and downloads the newest rollout video to:

- `outputs/rollout_videos_server/`

Notes:

- For long server runs, timeout defaults are intentionally relaxed for lab GPUs.
- If you forget server credentials, check `C:\Users\yaoda\Desktop\HOW-TO-USE-LAB-MACHINE.txt`.

## Notes

- Actions are normalized with training-set statistics.
- Training loss includes reconstruction, next-frame prediction, latent consistency, reward, and done.
- Start with single-step training. Then expand to multi-step rollout losses.
- One-command server pipeline:

```bash
bash scripts/server_pipeline.sh data/raw 200 100 80 world_model 64 20
```

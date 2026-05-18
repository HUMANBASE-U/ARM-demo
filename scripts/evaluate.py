import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import TransitionDataset, load_indices
from src.losses.losses import compute_losses
from src.models.world_model import WorldModel
from src.utils.io import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    default_index_dir = os.path.join(
        "data",
        "processed",
        os.path.basename(os.path.normpath(cfg["data"]["data_dir"])),
    )
    index_dir = args.index_dir or default_index_dir
    _, val_idx, action_stats = load_indices(index_dir)
    val_ds = TransitionDataset(val_idx, action_stats)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    action_dim = int(ckpt["action_dim"])
    model = WorldModel(
        latent_dim=cfg["model"]["latent_dim"],
        action_dim=action_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meter = {"total": 0.0, "recon": 0.0, "pred": 0.0, "latent": 0.0, "reward": 0.0, "done": 0.0}
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["x_t"], batch["a_t"], batch["x_tp1"])
            losses = compute_losses(
                out,
                x_t=batch["x_t"],
                x_tp1=batch["x_tp1"],
                r_t=batch["r_t"],
                d_t=batch["d_t"],
                lambda_recon=cfg["train"]["lambda_recon"],
                lambda_pred=cfg["train"]["lambda_pred"],
                lambda_latent=cfg["train"]["lambda_latent"],
                lambda_reward=cfg["train"]["lambda_reward"],
                lambda_done=cfg["train"]["lambda_done"],
            )
            for k in meter:
                meter[k] += float(losses[k].item())
            steps += 1
    for k in meter:
        meter[k] /= max(steps, 1)
    print("Validation losses:", meter)


if __name__ == "__main__":
    main()

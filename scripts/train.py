import argparse
import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import TransitionDataset, build_indices, load_indices, save_indices
from src.losses.losses import compute_losses
from src.models.world_model import WorldModel
from src.utils.io import ensure_dir, load_yaml, save_checkpoint
from src.utils.seed import set_seed


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def save_vis_grid(path: str, x_t, x_t_recon, x_tp1, x_tp1_pred, n: int = 4):
    ensure_dir(os.path.dirname(path))
    x_t = x_t[:n].detach().cpu()
    x_t_recon = x_t_recon[:n].detach().cpu()
    x_tp1 = x_tp1[:n].detach().cpu()
    x_tp1_pred = x_tp1_pred[:n].detach().cpu()
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    for i in range(n):
        imgs = [x_t[i], x_t_recon[i], x_tp1[i], x_tp1_pred[i]]
        titles = ["x_t", "x_t_recon", "x_tp1_gt", "x_tp1_pred"]
        for j, (img, title) in enumerate(zip(imgs, titles)):
            axes[i, j].imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run_epoch(model, loader, optimizer, cfg_train, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()
    meter = {"total": 0.0, "recon": 0.0, "pred": 0.0, "latent": 0.0, "reward": 0.0, "done": 0.0}
    steps = 0
    iterator = tqdm(loader, desc="train" if train_mode else "val", leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        x_t = batch["x_t"]
        a_t = batch["a_t"]
        x_tp1 = batch["x_tp1"]
        r_t = batch["r_t"]
        d_t = batch["d_t"]

        with torch.set_grad_enabled(train_mode):
            out = model(x_t, a_t, x_tp1)
            losses = compute_losses(
                out,
                x_t=x_t,
                x_tp1=x_tp1,
                r_t=r_t,
                d_t=d_t,
                lambda_recon=cfg_train["lambda_recon"],
                lambda_pred=cfg_train["lambda_pred"],
                lambda_latent=cfg_train["lambda_latent"],
                lambda_reward=cfg_train["lambda_reward"],
                lambda_done=cfg_train["lambda_done"],
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train["grad_clip"])
                optimizer.step()

        for k in meter:
            meter[k] += float(losses[k].item())
        steps += 1
    for k in meter:
        meter[k] /= max(steps, 1)
    return meter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--index_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.save_name is not None:
        cfg["train"]["save_name"] = args.save_name

    set_seed(cfg["seed"])
    device_str = "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)

    default_index_dir = os.path.join(
        "data",
        "processed",
        os.path.basename(os.path.normpath(cfg["data"]["data_dir"])),
    )
    index_dir = args.index_dir or default_index_dir
    ensure_dir(index_dir)
    train_index_path = os.path.join(index_dir, "train_index.pkl")
    if not os.path.exists(train_index_path):
        train_idx, val_idx, action_stats = build_indices(
            data_dir=cfg["data"]["data_dir"],
            val_ratio=cfg["data"]["val_ratio"],
            seed=cfg["seed"],
        )
        save_indices(train_idx, val_idx, action_stats, index_dir)
    else:
        train_idx, val_idx, action_stats = load_indices(index_dir)

    train_ds = TransitionDataset(train_idx, action_stats)
    val_ds = TransitionDataset(val_idx, action_stats)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    sample = train_ds[0]
    action_dim = int(sample["a_t"].shape[0])
    model = WorldModel(
        latent_dim=cfg["model"]["latent_dim"],
        action_dim=action_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    ensure_dir("checkpoints")
    ensure_dir("outputs/recon_samples")

    best_val = 1e9
    save_name = cfg["train"]["save_name"]
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_meter = run_epoch(model, train_loader, optimizer, cfg["train"], device, train_mode=True)
        val_meter = run_epoch(model, val_loader, optimizer, cfg["train"], device, train_mode=False)
        print(f"[Epoch {epoch}] train={train_meter} val={val_meter}")

        with torch.no_grad():
            batch = next(iter(val_loader))
            batch = move_batch(batch, device)
            out = model(batch["x_t"], batch["a_t"], batch["x_tp1"])
            save_vis_grid(
                path=os.path.join("outputs/recon_samples", f"epoch_{epoch:03d}.png"),
                x_t=batch["x_t"],
                x_t_recon=out["x_t_recon"],
                x_tp1=batch["x_tp1"],
                x_tp1_pred=out["x_tp1_pred"],
            )

        latest_path = os.path.join("checkpoints", f"{save_name}_latest.pt")
        save_checkpoint(
            latest_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "action_mean": action_stats.mean,
                "action_std": action_stats.std,
                "action_dim": action_dim,
            },
        )
        if val_meter["pred"] < best_val:
            best_val = val_meter["pred"]
            best_path = os.path.join("checkpoints", f"{save_name}_best.pt")
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "action_mean": action_stats.mean,
                    "action_std": action_stats.std,
                    "action_dim": action_dim,
                },
            )
            print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

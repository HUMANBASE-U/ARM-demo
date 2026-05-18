from typing import Dict

import torch
import torch.nn.functional as F


def compute_losses(
    model_out: Dict[str, torch.Tensor],
    x_t: torch.Tensor,
    x_tp1: torch.Tensor,
    r_t: torch.Tensor,
    d_t: torch.Tensor,
    lambda_recon: float,
    lambda_pred: float,
    lambda_latent: float,
    lambda_reward: float,
    lambda_done: float,
) -> Dict[str, torch.Tensor]:
    l_recon = F.l1_loss(model_out["x_t_recon"], x_t)
    l_pred = F.l1_loss(model_out["x_tp1_pred"], x_tp1)
    l_latent = F.mse_loss(model_out["z_tp1_pred"], model_out["z_tp1_true"].detach())
    l_reward = F.mse_loss(model_out["r_hat"], r_t)
    l_done = F.binary_cross_entropy_with_logits(model_out["d_logit"], d_t)

    total = (
        lambda_recon * l_recon
        + lambda_pred * l_pred
        + lambda_latent * l_latent
        + lambda_reward * l_reward
        + lambda_done * l_done
    )
    return {
        "total": total,
        "recon": l_recon,
        "pred": l_pred,
        "latent": l_latent,
        "reward": l_reward,
        "done": l_done,
    }

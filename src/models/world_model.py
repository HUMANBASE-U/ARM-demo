import torch
import torch.nn as nn

from .decoder import Decoder
from .dynamics import DynamicsModel
from .encoder import Encoder


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.dynamics = DynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor, x_tp1: torch.Tensor):
        z_t = self.encoder(x_t)
        z_tp1_true = self.encoder(x_tp1)
        z_tp1_pred, r_hat, d_logit = self.dynamics(z_t, a_t)
        x_t_recon = self.decoder(z_t)
        x_tp1_pred = self.decoder(z_tp1_pred)
        return {
            "z_t": z_t,
            "z_tp1_true": z_tp1_true,
            "z_tp1_pred": z_tp1_pred,
            "x_t_recon": x_t_recon,
            "x_tp1_pred": x_tp1_pred,
            "r_hat": r_hat,
            "d_logit": d_logit,
        }

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, actions: torch.Tensor):
        z = self.encoder(x0)
        preds = []
        for t in range(actions.shape[1]):
            a_t = actions[:, t, :]
            z, _, _ = self.dynamics(z, a_t)
            x_pred = self.decoder(z)
            preds.append(x_pred)
        return torch.stack(preds, dim=1)

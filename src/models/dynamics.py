import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor):
        inp = torch.cat([z_t, a_t], dim=-1)
        delta = self.transition(inp)
        z_hat_tp1 = z_t + delta
        r_hat = self.reward_head(inp)
        d_logit = self.done_head(inp)
        return z_hat_tp1, r_hat, d_logit

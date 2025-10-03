import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, temperature: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.temperature = temperature
        self.register_buffer("usage_ma", torch.zeros(3))  # moving average of usage
        self.ma_momentum = 0.9

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        logits = self.net(state_features)
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs

    def sample_action(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(state_features)
        probs = F.softmax(logits / self.temperature, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, probs

    def update_usage_ma(self, probs: torch.Tensor):
        batch_usage = probs.mean(dim=0).detach()
        self.usage_ma = self.ma_momentum * self.usage_ma + (1 - self.ma_momentum) * batch_usage

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return - (probs * (probs + eps).log()).sum(dim=1).mean()

    def load_balance_loss(self) -> torch.Tensor:
        # KL divergence against uniform using moving average usage
        eps = 1e-8
        q = (self.usage_ma + eps) / (self.usage_ma.sum() + 3 * eps)
        uniform = torch.full_like(q, 1.0 / len(q))
        kl = (q * (q / (uniform + eps)).log()).sum()
        return kl



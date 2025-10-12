import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from mlp import MLP


class ExpertBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        arch = [input_dim, hidden_dim, 2]
        self.backbone = MLP(arch, dropout=0.3).float()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, logits = self.backbone(x)
        probs = F.softmax(logits, dim=1)
        return hidden, probs

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.predict_proba(x)


class Expert1(ExpertBase):
    """Utility-focused expert using standard cross-entropy."""

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden, probs = self.forward(x)
        loss = F.cross_entropy(probs, y.long())
        return {"loss": loss, "ce": loss, "aux": torch.tensor(0.0, device=x.device)}


class Expert2(ExpertBase):
    """
    Result-driven fairness expert.
    - Representation alignment: minimize distance between group mean embeddings
    - Differentiable DP/EO surrogates on probabilities
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8, lambda_rep: float = 1.0, lambda_fair: float = 1.0):
        super().__init__(input_dim, hidden_dim)
        self.lambda_rep = lambda_rep
        self.lambda_fair = lambda_fair

    @staticmethod
    def _group_means(h: torch.Tensor, sens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask0 = (sens == 0).float().unsqueeze(1)
        mask1 = (sens == 1).float().unsqueeze(1)
        n0 = mask0.sum(dim=0).clamp_min(1.0)
        n1 = mask1.sum(dim=0).clamp_min(1.0)
        mu0 = (h * mask0).sum(dim=0, keepdim=True) / n0
        mu1 = (h * mask1).sum(dim=0, keepdim=True) / n1
        return mu0, mu1

    @staticmethod
    def _dp_gap(probs_pos: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
        # demographic parity gap over probabilities (differentiable)
        p0 = probs_pos[sens == 0].mean() if (sens == 0).any() else probs_pos.mean()
        p1 = probs_pos[sens == 1].mean() if (sens == 1).any() else probs_pos.mean()
        return (p0 - p1).abs()

    @staticmethod
    def _eo_gap(probs_pos: torch.Tensor, sens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # equal opportunity on positives (TPR surrogate using probs)
        mask_y1 = (labels == 1)
        p0 = probs_pos[torch.bitwise_and(sens == 0, mask_y1)].mean() if torch.bitwise_and(sens == 0, mask_y1).any() else probs_pos[mask_y1].mean()
        p1 = probs_pos[torch.bitwise_and(sens == 1, mask_y1)].mean() if torch.bitwise_and(sens == 1, mask_y1).any() else probs_pos[mask_y1].mean()
        return (p0 - p1).abs()

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, sens: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden, probs = self.forward(x)
        ce = F.cross_entropy(probs, y.long())

        # Representation alignment - with stability improvements
        mu0, mu1 = self._group_means(hidden, sens)
        l_rep = F.mse_loss(mu0, mu1)
        
        # Add small regularization to prevent collapse
        l_rep = l_rep + 1e-6 * (mu0.norm() + mu1.norm())

        # Differentiable DP/EO on probabilities of positive class
        probs_pos = probs[:, 1]
        l_dp = self._dp_gap(probs_pos, sens)
        l_eo = self._eo_gap(probs_pos, sens, y)
        l_fair = (l_dp + l_eo) / 2.0

        # Add stability to prevent NaN/Inf
        l_fair = torch.clamp(l_fair, 0.0, 10.0)
        l_rep = torch.clamp(l_rep, 0.0, 10.0)

        loss = ce + self.lambda_rep * l_rep + self.lambda_fair * l_fair
        return {"loss": loss, "ce": ce, "rep": l_rep, "fair": l_fair}


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


class Expert3(ExpertBase):
    """
    Procedural fairness expert.
    - Attention alignment: reuse attention fairness loss from attention_fairness_utils
    - Adversarial debiasing: predict sensitive attribute from hidden and from logits
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8, lambda_attention: float = 1.0, lambda_adv: float = 1.0):
        super().__init__(input_dim, hidden_dim)
        self.lambda_attention = lambda_attention
        self.lambda_adv = lambda_adv
        self.grl = GRL(1.0)
        self.adv_hidden = nn.Sequential(
            nn.Linear(hidden_dim, max(4, hidden_dim // 2), bias=True),
            nn.ReLU(),
            nn.Linear(max(4, hidden_dim // 2), 2, bias=True),
        )
        self.adv_logit = nn.Sequential(
            nn.Linear(2, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 2, bias=True),
        )

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sens: torch.Tensor,
        attention_loss_fn,  # callable(model, X_batch, sens_batch) -> scalar
    ) -> Dict[str, torch.Tensor]:
        hidden, probs = self.forward(x)
        ce = F.cross_entropy(probs, y.long())

        # Attention fairness loss (robustly handle failures)
        try:
            l_att = attention_loss_fn(self, x, sens)
        except Exception:
            l_att = torch.tensor(0.0, device=x.device)

        # Adversarial debiasing: hidden and logits
        adv_hidden_in = self.grl(hidden)
        adv_logits_h = self.adv_hidden(adv_hidden_in)
        l_adv_h = F.cross_entropy(adv_logits_h, sens.long())

        adv_logit_in = self.grl(probs.detach() if probs.requires_grad is False else probs)
        adv_logits_l = self.adv_logit(adv_logit_in)
        l_adv_l = F.cross_entropy(adv_logits_l, sens.long())

        l_adv = (l_adv_h + l_adv_l) / 2.0

        loss = ce + self.lambda_attention * l_att + self.lambda_adv * l_adv
        return {"loss": loss, "ce": ce, "att": l_att, "adv": l_adv}



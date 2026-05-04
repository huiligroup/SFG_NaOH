"""Loss functions for evidential interatomic potentials."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def quantile_residual_loss(residual: Tensor, q: float) -> Tensor:
    q_tensor = torch.as_tensor(q, dtype=residual.dtype, device=residual.device)
    return torch.maximum(q_tensor * residual, (q_tensor - 1.0) * residual)


def evidential_quantile_nll(
    target_force: Tensor,
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    q: float = 0.5,
    eps: float = 1.0e-8,
) -> Tensor:
    """Student-t NLL from evidential Bayesian quantile regression.

    Shapes are ``[num_atoms, 3]`` for all tensors.
    """

    tau = (1.0 - 2.0 * q) / (q * (1.0 - q))
    omega = 2.0 / (q * (1.0 - q))
    z = alpha / beta.clamp_min(eps)
    omega_term = 4.0 * beta * (1.0 + omega * z * nu)
    loc = gamma + tau * z
    residual = target_force - loc

    return (
        0.5 * (math.log(math.pi) - torch.log(nu.clamp_min(eps)))
        - alpha * torch.log(omega_term.clamp_min(eps))
        + (alpha + 0.5) * torch.log((residual.square() * nu + omega_term).clamp_min(eps))
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def evidential_regularizer(
    target_force: Tensor,
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    q: float = 0.5,
    eps: float = 1.0e-8,
) -> Tensor:
    residual = target_force - gamma
    confidence = 2.0 * nu + alpha + beta.clamp_min(eps).reciprocal()
    return quantile_residual_loss(residual, q) * confidence


def eip_loss(
    pred_energy: Tensor,
    true_energy: Tensor,
    pred_force: Tensor,
    true_force: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    force_weight: float = 1.0,
    reg_weight: float = 1.0e-2,
    q: float = 0.5,
) -> dict[str, Tensor]:
    energy_loss = F.l1_loss(pred_energy.view_as(true_energy), true_energy)
    nll = evidential_quantile_nll(true_force, pred_force, nu, alpha, beta, q=q).mean()
    reg = evidential_regularizer(true_force, pred_force, nu, alpha, beta, q=q).mean()
    total = energy_loss + force_weight * (nll + reg_weight * reg)
    return {
        "loss": total,
        "energy_l1": energy_loss,
        "force_nll": nll,
        "evidence_reg": reg,
    }


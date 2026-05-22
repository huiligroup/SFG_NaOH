"""Uncertainty aggregation and UDD bias utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def aggregate_atom_uncertainty(epistemic_per_axis: Tensor, eps: float = 1.0e-8) -> Tensor:
    return torch.sqrt(epistemic_per_axis.square().mean(dim=-1) + eps)


def aggregate_frame_uncertainty(atom_uncertainty: Tensor, power: float = 8.0, eps: float = 1.0e-8) -> Tensor:
    return (atom_uncertainty.clamp_min(eps).pow(power).mean() + eps).pow(1.0 / power)


def calibrate_reference(
    values: list[float] | np.ndarray,
    *,
    quantile: float = 0.90,
    eps: float = 1.0e-8,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot calibrate UDD reference with no uncertainty samples")
    u_ref = float(np.quantile(array, quantile))
    q75, q25 = np.quantile(array, [0.75, 0.25])
    u_scale = float(max(q75 - q25, eps))
    return u_ref, u_scale


def threshold_value(u_ref: float, u_scale: float, threshold_sigma: float) -> float:
    return float(u_ref + threshold_sigma * u_scale)


def bias_energy_ev(
    frame_uncertainty_ev_per_angstrom: Tensor,
    *,
    lambda_bias_ev: float,
    u_ref_ev_per_angstrom: float,
    u_scale_ev_per_angstrom: float,
) -> Tensor:
    scaled = (frame_uncertainty_ev_per_angstrom - float(u_ref_ev_per_angstrom)) / float(u_scale_ev_per_angstrom)
    return -float(lambda_bias_ev) * F.softplus(scaled)

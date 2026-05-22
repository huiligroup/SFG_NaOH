"""ASE calculator wrapper for ViSNet-eIP with optional UDD bias."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

from visnet.checkpoint import build_model_from_checkpoint

from .udd import aggregate_atom_uncertainty, aggregate_frame_uncertainty, bias_energy_ev


_Z_TABLE = {"H": 1, "O": 8, "Na": 11}


class VisNetEIPASECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        mode: str = "physical",
        lambda_bias_ev: float = 0.0,
        u_ref_ev_per_angstrom: float | None = None,
        u_scale_ev_per_angstrom: float | None = None,
        aggregator_power: float = 8.0,
        eps: float = 1.0e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in {"physical", "udd"}:
            raise ValueError("mode must be 'physical' or 'udd'")
        self.checkpoint = checkpoint
        self.device_name = device
        self.mode = mode
        self.lambda_bias_ev = float(lambda_bias_ev)
        self.u_ref_ev_per_angstrom = None if u_ref_ev_per_angstrom is None else float(u_ref_ev_per_angstrom)
        self.u_scale_ev_per_angstrom = None if u_scale_ev_per_angstrom is None else float(u_scale_ev_per_angstrom)
        self.aggregator_power = float(aggregator_power)
        self.eps = float(eps)
        self._model = None

    def set_mode(self, mode: str) -> None:
        if mode not in {"physical", "udd"}:
            raise ValueError("mode must be 'physical' or 'udd'")
        self.mode = mode
        self.reset()

    def set_udd_calibration(self, u_ref_ev_per_angstrom: float, u_scale_ev_per_angstrom: float) -> None:
        self.u_ref_ev_per_angstrom = float(u_ref_ev_per_angstrom)
        self.u_scale_ev_per_angstrom = float(max(u_scale_ev_per_angstrom, self.eps))
        self.reset()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes) -> None:
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        if atoms is None:
            raise ValueError("ASE calculator requires atoms")
        if not np.all(atoms.get_pbc()):
            raise ValueError("ViSNet active learning expects fully periodic atoms with PBC=True in XYZ")
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if cell.shape != (3, 3) or np.linalg.norm(cell) == 0.0:
            raise ValueError("Atoms object must carry a full 3x3 periodic cell")

        torch_mod, model = self._load_model()
        device = torch.device(self.device_name)
        species = atoms.get_chemical_symbols()
        try:
            z = torch_mod.as_tensor([_Z_TABLE[item] for item in species], dtype=torch.long, device=device)
        except KeyError as exc:
            raise ValueError(f"Unsupported element for active learning calculator: {exc.args[0]!r}") from exc

        pos = torch_mod.as_tensor(atoms.get_positions(wrap=False), dtype=torch.float32, device=device).requires_grad_(True)
        batch = torch_mod.zeros(len(species), dtype=torch.long, device=device)
        cell_tensor = torch_mod.as_tensor(cell[None, :, :], dtype=torch.float32, device=device)

        model.pbc_distance.set_graph(cell=cell_tensor, edge_index=None, cell_shift=None)
        x, v = model._representation(z, pos, batch)
        atom_energy = model.backbone.output_model.pre_reduce(x, v)
        atom_energy = atom_energy * model.backbone.std
        if model.backbone.prior_model is not None:
            atom_energy = model.backbone.prior_model(atom_energy, z)
        physical_energy_h = model.scatter(atom_energy, batch, dim=0, reduce=model.backbone.reduce_op).view(-1)
        physical_energy_h = physical_energy_h + model.backbone.mean
        physical_energy_scalar = physical_energy_h.sum()

        nu, alpha, beta = model.evidential_head(v)
        epistemic_h_per_ang = beta / (nu * (alpha - 1.0)).clamp_min(1.0e-8)
        aleatoric_h_per_ang = beta / (alpha - 1.0).clamp_min(1.0e-8)

        atom_unc_h_per_ang = aggregate_atom_uncertainty(epistemic_h_per_ang, eps=self.eps)
        atom_unc_ev_per_ang = atom_unc_h_per_ang * Hartree
        frame_unc_ev_per_ang = aggregate_frame_uncertainty(
            atom_unc_ev_per_ang,
            power=self.aggregator_power,
            eps=self.eps,
        )

        physical_forces_h_per_ang = -torch_mod.autograd.grad(
            physical_energy_scalar,
            pos,
            create_graph=False,
            retain_graph=self.mode == "udd",
        )[0]

        if self.mode == "udd":
            if self.u_ref_ev_per_angstrom is None or self.u_scale_ev_per_angstrom is None:
                raise ValueError("UDD mode requires calibrated u_ref_ev_per_angstrom and u_scale_ev_per_angstrom")
            bias_ev = bias_energy_ev(
                frame_unc_ev_per_ang,
                lambda_bias_ev=self.lambda_bias_ev,
                u_ref_ev_per_angstrom=self.u_ref_ev_per_angstrom,
                u_scale_ev_per_angstrom=self.u_scale_ev_per_angstrom,
            )
            total_energy_h = physical_energy_scalar + bias_ev / Hartree
            total_forces_h_per_ang = -torch_mod.autograd.grad(total_energy_h, pos, create_graph=False, retain_graph=False)[0]
        else:
            bias_ev = torch_mod.zeros((), dtype=physical_energy_scalar.dtype, device=device)
            total_forces_h_per_ang = physical_forces_h_per_ang

        frame_feature = torch_mod.cat(
            [
                x.mean(dim=0),
                x.std(dim=0, unbiased=False),
            ],
            dim=0,
        )
        total_energy_ev = float((physical_energy_scalar + bias_ev / Hartree).detach().cpu()) * Hartree
        physical_energy_ev = float(physical_energy_scalar.detach().cpu()) * Hartree
        self.results = {
            "energy": total_energy_ev,
            "free_energy": total_energy_ev,
            "forces": total_forces_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False) * Hartree,
            "physical_energy_hartree": float(physical_energy_scalar.detach().cpu()),
            "physical_energy_ev": physical_energy_ev,
            "physical_forces_hartree_per_angstrom": physical_forces_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False),
            "physical_forces_ev_per_angstrom": physical_forces_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False) * Hartree,
            "epistemic_per_atom_per_axis_hartree_per_angstrom": epistemic_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False),
            "epistemic_per_atom_per_axis_ev_per_angstrom": epistemic_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False) * Hartree,
            "aleatoric_per_atom_per_axis_hartree_per_angstrom": aleatoric_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False),
            "aleatoric_per_atom_per_axis_ev_per_angstrom": aleatoric_h_per_ang.detach().cpu().numpy().astype(np.float64, copy=False) * Hartree,
            "uncertainty_per_atom_ev_per_angstrom": atom_unc_ev_per_ang.detach().cpu().numpy().astype(np.float64, copy=False),
            "uncertainty_frame_ev_per_angstrom": float(frame_unc_ev_per_ang.detach().cpu()),
            "udd_bias_energy_ev": float(bias_ev.detach().cpu()),
            "mode": self.mode,
            "u_ref_ev_per_angstrom": self.u_ref_ev_per_angstrom,
            "u_scale_ev_per_angstrom": self.u_scale_ev_per_angstrom,
            "frame_feature": frame_feature.detach().cpu().numpy().astype(np.float32, copy=False),
        }

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None:
            model, _, _ = build_model_from_checkpoint(self.checkpoint, device=self.device_name, eval_mode=True)
            self._model = model
        return torch, self._model

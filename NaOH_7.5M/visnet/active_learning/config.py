"""TOML-backed configuration for ViSNet active learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str | Path
    device: str = "cpu"
    batch_size: int = 1

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("model.batch_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint"] = str(self.checkpoint)
        return data


@dataclass(frozen=True)
class MDConfig:
    initial_xyz: str | Path
    initial_frame: int = 0
    cell: list[float] | list[list[float]] | None = None
    integrator: str = "langevin"
    temperature_k: float = 300.0
    timestep_fs: float = 0.5
    friction: float = 0.01
    warmup_steps: int = 2000
    steps_per_round: int = 5000
    max_rounds: int = 1
    eval_interval: int = 10
    trajectory_interval: int = 10
    log_interval: int = 10
    seed: int | None = None

    def validate(self) -> None:
        if self.integrator.lower() != "langevin":
            raise ValueError("Only Langevin integrator is implemented in v1")
        if self.initial_frame < 0:
            raise ValueError("md.initial_frame must be >= 0")
        if self.temperature_k <= 0.0:
            raise ValueError("md.temperature_k must be > 0")
        if self.timestep_fs <= 0.0:
            raise ValueError("md.timestep_fs must be > 0")
        if self.warmup_steps < 1:
            raise ValueError("md.warmup_steps must be >= 1")
        if self.steps_per_round < self.warmup_steps:
            raise ValueError("md.steps_per_round must be >= md.warmup_steps")
        if self.max_rounds < 1:
            raise ValueError("md.max_rounds must be >= 1")
        if self.eval_interval < 1:
            raise ValueError("md.eval_interval must be >= 1")
        if self.trajectory_interval < 1:
            raise ValueError("md.trajectory_interval must be >= 1")
        if self.log_interval < 1:
            raise ValueError("md.log_interval must be >= 1")
        if self.cell is None:
            raise ValueError("md.cell is required for periodic NaOH12 active learning")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["initial_xyz"] = str(self.initial_xyz)
        return data


@dataclass(frozen=True)
class UDDConfig:
    lambda_bias_ev: float = 0.05
    reference_quantile: float = 0.90
    threshold_sigma: float = 2.0
    threshold_hits: int = 3
    aggregator_power: float = 8.0
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        if self.lambda_bias_ev < 0.0:
            raise ValueError("udd.lambda_bias_ev must be >= 0")
        if not 0.0 < self.reference_quantile < 1.0:
            raise ValueError("udd.reference_quantile must be in (0, 1)")
        if self.threshold_sigma < 0.0:
            raise ValueError("udd.threshold_sigma must be >= 0")
        if self.threshold_hits < 1:
            raise ValueError("udd.threshold_hits must be >= 1")
        if self.aggregator_power < 1.0:
            raise ValueError("udd.aggregator_power must be >= 1")
        if self.epsilon <= 0.0:
            raise ValueError("udd.epsilon must be > 0")


@dataclass(frozen=True)
class SelectionConfig:
    batch_query_size: int = 20
    cooldown_steps: int = 100
    min_feature_distance: float = 1.0e-6

    def validate(self) -> None:
        if self.batch_query_size < 1:
            raise ValueError("selection.batch_query_size must be >= 1")
        if self.cooldown_steps < 0:
            raise ValueError("selection.cooldown_steps must be >= 0")
        if self.min_feature_distance < 0.0:
            raise ValueError("selection.min_feature_distance must be >= 0")


@dataclass(frozen=True)
class CP2KConfig:
    template_source: str | Path
    execution: str = "mock"
    command: str = "cp2k.popt"
    nproc: int = 1
    keep_workdirs: bool = True
    submit_command: str = "sbatch"
    poll_interval_seconds: float = 30.0
    max_wait_seconds: float = 86400.0
    sbatch_args: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.execution not in {"mock", "direct", "slurm"}:
            raise ValueError("cp2k.execution must be one of mock/direct/slurm")
        if self.nproc < 1:
            raise ValueError("cp2k.nproc must be >= 1")
        if self.poll_interval_seconds <= 0.0:
            raise ValueError("cp2k.poll_interval_seconds must be > 0")
        if self.max_wait_seconds <= 0.0:
            raise ValueError("cp2k.max_wait_seconds must be > 0")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["template_source"] = str(self.template_source)
        return data


@dataclass(frozen=True)
class RetrainConfig:
    base_data: str | Path
    epochs: int = 5
    batch_size: int = 4
    grad_accumulation_steps: int = 1
    lr: float = 1.0e-4
    device: str = "cpu"
    frame_stride: int = 4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    batches_per_step: int = 100
    val_interval_steps: int = 10
    limit_batches: int | None = None
    limit_val_batches: int | None = None
    grad_clip: float = 10.0
    gradient_checkpointing: bool = False
    build_edges_if_missing: bool = True
    wandb: bool = False
    wandb_project: str = "naoh-visnet-active-learning"
    wandb_name: str | None = None

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("retrain.epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("retrain.batch_size must be >= 1")
        if self.grad_accumulation_steps < 1:
            raise ValueError("retrain.grad_accumulation_steps must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("retrain.lr must be > 0")
        if self.frame_stride < 1:
            raise ValueError("retrain.frame_stride must be >= 1")
        if self.early_stopping_patience < 1:
            raise ValueError("retrain.early_stopping_patience must be >= 1")
        if self.batches_per_step < 1:
            raise ValueError("retrain.batches_per_step must be >= 1")
        if self.val_interval_steps < 1:
            raise ValueError("retrain.val_interval_steps must be >= 1")
        if self.grad_clip <= 0.0:
            raise ValueError("retrain.grad_clip must be > 0")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_data"] = str(self.base_data)
        return data


@dataclass(frozen=True)
class OutputConfig:
    run_name: str
    root_dir: str | Path = ROOT / "data" / "active_learning"
    resume: bool = True
    overwrite: bool = False
    keep_intermediates: bool = True

    def validate(self) -> None:
        if not self.run_name:
            raise ValueError("output.run_name cannot be empty")
        if self.resume and self.overwrite:
            raise ValueError("output.resume and output.overwrite cannot both be true")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root_dir"] = str(self.root_dir)
        return data

    def resolved_run_dir(self) -> Path:
        return Path(self.root_dir) / self.run_name


@dataclass(frozen=True)
class ActiveLearningConfig:
    model: ModelConfig
    md: MDConfig
    udd: UDDConfig
    selection: SelectionConfig
    cp2k: CP2KConfig
    retrain: RetrainConfig
    output: OutputConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.model.validate()
        self.md.validate()
        self.udd.validate()
        self.selection.validate()
        self.cp2k.validate()
        self.retrain.validate()
        self.output.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "md": self.md.to_dict(),
            "udd": asdict(self.udd),
            "selection": asdict(self.selection),
            "cp2k": self.cp2k.to_dict(),
            "retrain": self.retrain.to_dict(),
            "output": self.output.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveLearningConfig":
        config = cls(
            model=ModelConfig(**data.get("model", {})),
            md=MDConfig(**data.get("md", {})),
            udd=UDDConfig(**data.get("udd", {})),
            selection=SelectionConfig(**data.get("selection", {})),
            cp2k=CP2KConfig(**data.get("cp2k", {})),
            retrain=RetrainConfig(**data.get("retrain", {})),
            output=OutputConfig(**data.get("output", {})),
            metadata=dict(data.get("metadata", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_toml(cls, path: str | Path) -> "ActiveLearningConfig":
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

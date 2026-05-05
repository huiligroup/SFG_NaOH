"""Runner for LQVE result analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lqve.dvr.result import DVRResult

from .config import AnalysisConfig
from .io import load_shift_summary, select_transition_indices, write_table
from .report import write_report
from .result import AnalysisResult
from .stats import column_statistics, compact_statistics, frame_quality_rows, value_counts


def analyze_lqve_results(config: AnalysisConfig) -> AnalysisResult:
    config.validate()
    output_dir = config.resolved_output_dir()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = load_shift_summary(config.shifts)
    transition_indices = select_transition_indices(data["transitions_cm1"].shape[1], config.transitions)
    labels = [f"T{idx + 1}" for idx in transition_indices]
    transitions = data["transitions_cm1"][:, transition_indices]
    shifts = data["shifts_cm1"][:, transition_indices]
    success = data["success"]
    x = _frame_axis_values(data["frame_ids"], config.frame_axis)

    transition_stats = column_statistics(transitions, labels)
    shift_stats = column_statistics(shifts, labels)
    quality_rows = frame_quality_rows(
        frame_ids=data["frame_ids"],
        success=success,
        transitions=transitions,
        shifts=shifts,
        source_qc_dir=data["source_qc_dir"],
        reference_id=data["reference_id"],
    )
    compact = compact_statistics(success, transitions, shifts)
    compact["selected_transitions"] = ",".join(labels)
    compact["summary_rows"] = len(data["frame_ids"])
    compact["reference_mode"] = _reference_mode(data["shift_dir"])
    compact["reference_counts"] = value_counts(data["reference_id"])
    result = AnalysisResult(output_dir=output_dir, statistics=compact)

    transition_stats_path = tables_dir / "transition_stats.csv"
    shift_stats_path = tables_dir / "shift_stats.csv"
    frame_quality_path = tables_dir / "frame_quality.csv"
    write_table(transition_stats_path, transition_stats, _stats_fieldnames())
    write_table(shift_stats_path, shift_stats, _stats_fieldnames())
    write_table(
        frame_quality_path,
        quality_rows,
        [
            "frame_index",
            "frame_id",
            "success",
            "finite_transition_count",
            "finite_shift_count",
            "transition_nan_count",
            "shift_nan_count",
            "source_qc_dir",
            "reference_id",
        ],
    )
    result.tables.update(
        {
            "transition_stats": transition_stats_path,
            "shift_stats": shift_stats_path,
            "frame_quality": frame_quality_path,
        }
    )

    fmt = config.plot_format.lower()
    figures = {
        "transitions_timeseries": figures_dir / f"transitions_timeseries.{fmt}",
        "shifts_timeseries": figures_dir / f"shifts_timeseries.{fmt}",
        "transition_histograms": figures_dir / f"transition_histograms.{fmt}",
        "shift_histograms": figures_dir / f"shift_histograms.{fmt}",
        "transition_correlation": figures_dir / f"transition_correlation.{fmt}",
        "frame_quality": figures_dir / f"frame_quality.{fmt}",
    }
    from .plots import (
        plot_correlation,
        plot_dvr_wavefunction_diagnostic,
        plot_frame_quality,
        plot_histograms,
        plot_timeseries,
    )

    plot_timeseries(x, transitions, labels, "Transition / cm^-1", "LQVE Transitions", figures["transitions_timeseries"])
    plot_timeseries(x, shifts, labels, "Shift / cm^-1", "LQVE Frequency Shifts", figures["shifts_timeseries"])
    plot_histograms(transitions, labels, "Transition / cm^-1", "Transition Distributions", figures["transition_histograms"])
    plot_histograms(shifts, labels, "Shift / cm^-1", "Shift Distributions", figures["shift_histograms"])
    plot_correlation(transitions, labels, "Transition Correlation", figures["transition_correlation"])
    nan_counts = np.isnan(transitions).sum(axis=1) + np.isnan(shifts).sum(axis=1)
    plot_frame_quality(success, nan_counts, figures["frame_quality"])
    result.figures.update(figures)

    if config.include_dvr_diagnostics and config.dvr_data is not None:
        dvr = DVRResult.load(config.dvr_data)
        dvr_figure = figures_dir / f"dvr_wavefunction_diagnostic.{fmt}"
        plot_dvr_wavefunction_diagnostic(dvr.wavefunctions, dvr.basis_shape, dvr_figure)
        result.figures["dvr_wavefunction_diagnostic"] = dvr_figure

    failed_frames = [
        row
        for row in quality_rows
        if (not row["success"]) or row["transition_nan_count"] > 0 or row["shift_nan_count"] > 0
    ]
    report_path = output_dir / "report.md"
    write_report(
        path=report_path,
        title=config.report_title,
        input_shift_dir=str(data["shift_dir"]),
        statistics=compact,
        tables=result.tables,
        figures=result.figures,
        failed_frames=failed_frames,
    )
    result.report = report_path

    with (logs_dir / "analysis_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)
    result.save_metadata()
    return result


def _frame_axis_values(frame_ids: list[str], frame_axis: str) -> np.ndarray:
    if frame_axis == "index":
        return np.arange(len(frame_ids), dtype=np.float64)
    values = []
    for idx, frame_id in enumerate(frame_ids):
        try:
            values.append(float(frame_id))
        except ValueError:
            values.append(float(idx))
    return np.asarray(values, dtype=np.float64)


def _stats_fieldnames() -> list[str]:
    return ["label", "count", "finite_count", "nan_count", "nan_fraction", "mean", "std", "min", "q05", "median", "q95", "max"]


def _reference_mode(shift_dir: Path) -> str:
    metadata_path = shift_dir / "metadata.json"
    if not metadata_path.exists():
        return "unknown"
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return "unknown"
    return str(metadata.get("config", {}).get("reference_mode", "unknown"))

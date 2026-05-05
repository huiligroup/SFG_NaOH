"""Plotting utilities for LQVE analysis."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lqve_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "lqve_cache"))
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_timeseries(
    x: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    ylabel: str,
    title: str,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for idx, label in enumerate(labels):
        ax.plot(x, values[:, idx], marker=".", linewidth=1.2, markersize=4, label=label)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_histograms(
    values: np.ndarray,
    labels: list[str],
    xlabel: str,
    title: str,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ncols = min(3, len(labels))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.0 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)
    for idx, label in enumerate(labels):
        ax = axes_array[idx]
        finite = values[:, idx][np.isfinite(values[:, idx])]
        if finite.size:
            ax.hist(finite, bins=min(40, max(8, int(np.sqrt(finite.size)))), color="#4477aa", alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
    for ax in axes_array[len(labels) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_correlation(values: np.ndarray, labels: list[str], title: str, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if values.shape[1] == 1 or values.shape[0] < 2:
        corr = np.ones((1, 1), dtype=np.float64)
        labels = labels[:1]
    elif np.all(~np.isfinite(values)):
        corr = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    else:
        means = np.nanmean(values, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        filled = np.nan_to_num(values, nan=means)
        corr = np.corrcoef(filled, rowvar=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title(title)
    if np.all(~np.isfinite(corr)):
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_frame_quality(success: np.ndarray, nan_counts: np.ndarray, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(success.size)
    fig, ax1 = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax1.bar(x, nan_counts, color="#cc6677", alpha=0.75, label="NaN count")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("NaN count")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, success.astype(int), color="#228833", marker=".", linewidth=1.0, label="Success")
    ax2.set_ylabel("Success")
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_title("Frame Quality")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_dvr_wavefunction_diagnostic(
    wavefunctions: np.ndarray,
    basis_shape: tuple[int, ...],
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wf0 = np.asarray(wavefunctions)[:, 0]
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    if len(basis_shape) == 1:
        ax.plot(np.arange(wf0.size), wf0)
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Wavefunction")
    elif len(basis_shape) == 2:
        image = ax.imshow(wf0.reshape(basis_shape), origin="lower", aspect="auto", cmap="RdBu_r")
        fig.colorbar(image, ax=ax, label="Wavefunction")
    else:
        array = wf0.reshape(basis_shape)
        mid = array.shape[-1] // 2
        image = ax.imshow(array[..., mid].reshape(array.shape[0], -1), origin="lower", aspect="auto", cmap="RdBu_r")
        fig.colorbar(image, ax=ax, label="Wavefunction")
    ax.set_title("DVR Ground-State Wavefunction Diagnostic")
    fig.savefig(output, dpi=180)
    plt.close(fig)

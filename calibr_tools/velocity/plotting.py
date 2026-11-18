from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import ensure_dir
from .math_utils import optimal_scalar


def plot_cross_correlation(path: Path, lags, corr, offset, title):
    ensure_dir(Path(path).parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, corr, lw=1.2, color="0.20")
    ax.axvline(offset, color="C3", ls="--", lw=1.2, label=f"peak {offset:.4f}s")
    ax.set_xlabel("lag / residual offset (s)")
    ax.set_ylabel("normalized correlation")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_offset_cost_curve(path: Path, t_ref, ref_mag, t_sens, sens_mag, offset, window, fit_scale=True):
    ensure_dir(Path(path).parent)
    offsets = np.linspace(offset - window, offset + window, 161)
    costs = []
    scales = []
    for off in offsets:
        aligned = np.interp(t_ref, t_sens - off, sens_mag, left=np.nan, right=np.nan)
        valid = np.isfinite(aligned)
        if np.count_nonzero(valid) < 20:
            costs.append(np.nan)
            scales.append(np.nan)
            continue
        x = ref_mag[valid]
        y = aligned[valid]
        s = optimal_scalar(x, y) if fit_scale else 1.0
        costs.append(float(np.mean((s * x - y) ** 2)))
        scales.append(s)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(offsets, costs, lw=1.3, color="0.20", label="speed magnitude MSE")
    ax1.axvline(offset, color="C3", ls="--", lw=1.2, label=f"chosen {offset:.4f}s")
    ax1.set_xlabel("sensor - reference time offset (s)")
    ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.3)
    if fit_scale:
        ax2 = ax1.twinx()
        ax2.plot(offsets, scales, lw=1.0, color="C0", alpha=0.65, label="best global scale")
        ax2.set_ylabel("best global scale")
    ax1.set_title("Radar time-offset cost curve")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_alignment(path: Path, t_ref, sig_ref, t_sens, sig_sens, offset, title, ylabel):
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_ref - t_ref[0], sig_ref, lw=1.2, label="reference")
    axes[0].plot(t_sens - t_ref[0], sig_sens, lw=1.0, label="sensor")
    axes[0].set_title("Before")
    axes[1].plot(t_ref - t_ref[0], sig_ref, lw=1.2, label="reference")
    axes[1].plot(t_sens - offset - t_ref[0], sig_sens, lw=1.0, label="sensor")
    axes[1].set_title(f"After, offset={offset:.4f}s")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_components(path: Path, t, body_meas, body_pred, residual, title):
    ensure_dir(Path(path).parent)
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    t_rel = t - t[0]
    for i in range(3):
        axes[i].plot(t_rel, body_meas[:, i], "0.45", lw=0.9, label="radar velocity rotated to body")
        axes[i].plot(t_rel, body_pred[:, i], "C3", lw=1.1, label="reference rigid prediction")
        axes[i].set_ylabel(f"{labels[i]} m/s")
        axes[i].grid(True, alpha=0.35)
    axes[0].legend(ncol=2, fontsize=8)
    axes[3].plot(t_rel, np.linalg.norm(residual.reshape(-1, 3), axis=1), "C3", lw=0.9)
    axes[3].set_ylabel("|res| m/s")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.35)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_residual_hist(path: Path, residual, title, xlabel):
    ensure_dir(Path(path).parent)
    r = np.asarray(residual).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=80, color="0.25", alpha=0.85)
    rms = float(np.sqrt(np.mean(r * r)))
    ax.axvline(rms, color="C3", lw=1.2, label="RMS")
    ax.axvline(-rms, color="C3", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_vector_components(path: Path, t, series, labels, title, ylabel):
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(series[0][1].shape[1], 1, figsize=(10, 7), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    t0 = t[0]
    comp = labels if labels is not None else ["x", "y", "z"]
    for i, ax in enumerate(axes):
        for name, values, style in series:
            ax.plot(t - t0, values[:, i], style, lw=0.9, label=name)
        ax.set_ylabel(f"{comp[i]} {ylabel}")
        ax.grid(True, alpha=0.35)
    axes[0].legend(ncol=max(1, min(3, len(series))), fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_scalar_series(path: Path, series, title, ylabel):
    ensure_dir(Path(path).parent)
    fig, ax = plt.subplots(figsize=(10, 4))
    t0 = min(float(t[0]) for _, t, _, _ in series)
    for name, t, values, style in series:
        ax.plot(t - t0, values, style, lw=0.9, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

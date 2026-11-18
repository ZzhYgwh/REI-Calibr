from __future__ import annotations

import csv
import inspect
import json
import time
from dataclasses import dataclass

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

from .io import ensure_dir, load_reference_twist, load_vector_series, make_se3, save_json
from .math_utils import edge_trim_mask, estimate_offset_cc, interp_vectors, kabsch_body_from_sensor, optimal_scalar
from .plotting import (
    plot_alignment,
    plot_components,
    plot_cross_correlation,
    plot_offset_cost_curve,
    plot_residual_hist,
    plot_scalar_series,
    plot_vector_components,
)


# ======================================================================
# Debug helpers
# ======================================================================

class StageLogger:
    """Per-stage structured logger: writes stage_log.txt/json + prints to stdout."""

    def __init__(self, out_dir: Path):
        self.out = Path(out_dir) / "debug"
        ensure_dir(self.out)
        self.events = []
        self.t0 = time.time()
        self.log_path = self.out / "stage_log.txt"
        self.log_path.write_text("", encoding="utf-8")

    @staticmethod
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    def _scalarize(self, v):
        if isinstance(v, np.ndarray):
            if v.size < 64:
                return v.tolist()
            return f"ndarray(shape={v.shape}, min={float(np.nanmin(v)):.4g}, max={float(np.nanmax(v)):.4g})"
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    def log(self, step: str, **kw):
        rec = {"t": round(time.time() - self.t0, 4), "step": step,
               **{k: self._scalarize(v) for k, v in kw.items()}}
        self.events.append(rec)
        line = json.dumps(rec, ensure_ascii=False, default=self._default)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)
        return rec

    def flush(self):
        (self.out / "stage_log.json").write_text(
            json.dumps(self.events, ensure_ascii=False, indent=2, default=self._default),
            encoding="utf-8",
        )


def _plot_step_preprocess(out: Path, t_r, vr_raw, vr_clean, vr, quality_mask):
    ensure_dir(out / "debug")
    t_rel = t_r - t_r[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(t_rel, np.linalg.norm(vr_raw, axis=1),   color="0.70", lw=0.8, label="raw")
    axes[0].plot(t_rel, np.linalg.norm(vr_clean, axis=1), color="C0",   lw=0.9, label="cleaned")
    axes[0].plot(t_rel, np.linalg.norm(vr, axis=1),       color="C3",   lw=1.1, label="filtered")
    axes[0].set_ylabel("|v| (m/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Speed: raw vs cleaned vs filtered")

    axes[1].plot(t_rel, quality_mask.astype(float), color="C2", lw=0.9, label="quality kept")
    rejected = np.flatnonzero(~quality_mask)
    if len(rejected):
        axes[1].scatter(t_rel[rejected], np.zeros_like(rejected),
                        color="C1", s=4, label=f"rejected ({len(rejected)})")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("t - t0 (s)")
    axes[1].set_ylabel("mask")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Step 02: preprocessing")
    fig.tight_layout()
    fig.savefig(out / "debug" / "02_preprocess.png", dpi=200)
    plt.close(fig)


def _plot_step_kabsch(out: Path, tr, kabsch_meas, body_comp, kabsch_res, rms_init_kabsch):
    ensure_dir(out / "debug")
    title = f"Step 05: Kabsch initial (RMS={rms_init_kabsch:.4f} m/s)"
    plot_components(out / "debug" / "05_init_kabsch.png", tr, kabsch_meas, body_comp,
                    kabsch_res.reshape(-1), title)


def _plot_step_optimizer(out: Path, cost_hist):
    ensure_dir(out / "debug")
    if not cost_hist:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    c = np.maximum(np.asarray(cost_hist, dtype=float), 1e-12)
    ax.semilogy(c, "C0.-", lw=1.2)
    ax.set_xlabel("callback index")
    ax.set_ylabel("sum of squared residuals")
    ax.set_title("Step 06: optimizer cost curve")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "debug" / "06_optimizer_cost.png", dpi=200)
    plt.close(fig)


def _plot_step_residual(out: Path, t_r, residual_mat):
    ensure_dir(out / "debug")
    t_rel = t_r - t_r[0]
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex="col")
    for i, lab in enumerate(["x", "y", "z"]):
        axes[i, 0].plot(t_rel, residual_mat[:, i], "C0", lw=0.6)
        axes[i, 0].axhline(0.0, color="k", lw=0.5)
        axes[i, 0].set_ylabel(f"res {lab} (m/s)")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].hist(residual_mat[:, i], bins=60, color="C0", alpha=0.75)
        axes[i, 1].set_ylabel("count")
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("t - t0 (s)")
    axes[-1, 1].set_xlabel("res (m/s)")
    fig.suptitle("Step 07: per-axis residual")
    fig.tight_layout()
    fig.savefig(out / "debug" / "07_residual_per_axis.png", dpi=200)
    plt.close(fig)


def write_overview(out: Path, payload: Dict, debug_summary: Dict):
    ensure_dir(out)
    ti = payload.get("time_initialization", {}) or {}
    prep = payload.get("preprocessing", {}) or {}
    lp = prep.get("velocity_lowpass", {}) or {}
    obs = payload.get("translation_observability", {}) or {}

    rms_init = debug_summary.get("rms_init")
    rms_final = debug_summary.get("rms_final")
    impr = debug_summary.get("improvement_x")
    rms_xyz = debug_summary.get("rms_xyz")
    bias_xyz = debug_summary.get("bias_xyz")

    def _fmt(v, nd=4):
        return "n/a" if v is None else f"{float(v):.{nd}f}"

    lines = [
        "# Radar 链路总览",
        "",
        "## 数据规模",
        f"- reference 样本: {debug_summary.get('n_ref', 'n/a')}",
        f"- radar 样本: {debug_summary.get('n_radar', 'n/a')}",
        f"- 预处理后保留比例: {_fmt((debug_summary.get('quality_kept_frac') or 0) * 100, 2)} %",
        "",
        "## 时间对齐",
        f"- coarse_offset_adjust_sec: {ti.get('coarse_offset_adjust_sec')}",
        f"- cross_correlation_offset_sec: {ti.get('cross_correlation_offset_sec')}",
        f"- peak_corr: {debug_summary.get('peak_corr', 'n/a')}",
        f"- refined tau0 (s): {ti.get('refined_offset_sec')}",
        f"- final tau (s): {payload.get('time_offset_radar_minus_body_sec')}",
        "",
        "## 速度低通",
        f"- enabled: {lp.get('enabled', False)}",
        f"- method: {lp.get('method', 'none')}",
        f"- window_samples: {lp.get('window_samples', 'n/a')}",
        f"- edge_mode: {lp.get('edge_mode', 'n/a')}",
        f"- delta_rms (m/s): {lp.get('delta_rms', 'n/a')}",
        "",
        "## 误差对比（m/s）",
        f"- RMS 初值: {_fmt(rms_init)}",
        f"- RMS 终值: {_fmt(rms_final)}",
        f"- 提升倍数: {_fmt(impr, 2)} x",
        f"- RMS 逐轴 (x, y, z): {rms_xyz}",
        f"- bias 逐轴 (x, y, z): {bias_xyz}",
        f"- |res| p95: {_fmt(debug_summary.get('p95'))}",
        f"- |res| p99: {_fmt(debug_summary.get('p99'))}",
        "",
        "## 平移可观性",
        f"- analytic singular values: {obs.get('analytic_singular_values')}",
        f"- analytic condition: {_fmt(obs.get('analytic_condition'), 2)}",
        f"- numeric  condition: {_fmt(obs.get('numeric_condition'), 2)}",
        f"- omega norm median / p95 (rad/s): "
        f"{_fmt(obs.get('omega_norm_median'))} / {_fmt(obs.get('omega_norm_p95'))}",
        f"- lever speed median / p95 (m/s): "
        f"{_fmt(obs.get('lever_speed_median'))} / {_fmt(obs.get('lever_speed_p95'))}",
        "",
        "## 参数",
        f"- Euler xyz body<-radar (deg): {payload.get('euler_xyz_deg_body_radar')}",
        f"- p_body_radar_m: {payload.get('p_body_radar_m')}",
        f"- velocity_scale: {payload.get('velocity_scale')}",
        f"- scale_mode: {payload.get('scale_mode')}",
        "",
        "## 图片索引（按阶段）",
        "- debug/02_preprocess.png          raw / cleaned / filtered + quality mask",
        "- radar_cross_correlation.png      speed 互相关",
        "- radar_speed_activity.png         ref / radar epoch-aligned / final-aligned",
        "- radar_time_offset_cost.png      时间偏移 cost 曲线",
        "- radar_time_alignment.png        最终时间对齐",
        "- debug/05_init_kabsch.png        Kabsch 初值 vs 参考",
        "- debug/06_optimizer_cost.png     优化器 cost 收敛",
        "- radar_initial_body_velocity_fit.png  优化前拟合",
        "- radar_body_velocity_fit.png          优化后拟合",
        "- debug/07_residual_per_axis.png  逐轴残差时序 + 直方图",
        "- radar_translation_observability.png  平移可观性",
        "- radar_extrinsic_alignment_effect.png 外参效果",
        "- radar_residual_hist.png         残差直方图",
        "",
        "## 调试日志",
        "- debug/stage_log.txt             每步单行 JSON（按时间顺序）",
        "- debug/stage_log.json            同上，结构化",
        "- debug/velocity_trajectories.csv raw / clean / filtered 全轨迹",
        "- debug/optimizer_cost_curve.csv  优化器 cost",
    ]
    (out / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================
# Existing core helpers (unchanged)
# ======================================================================

@dataclass
class RadarStageResult:
    name: str
    output_dir: Path
    result_json: Path
    data_npz: Path
    payload: Dict


def _refine_offset(t_ref, sig_ref, t_sens, sig_sens, init_offset, search_range, fit_scale=True):
    def cost(dt):
        aligned = np.interp(t_ref, t_sens - dt, sig_sens, left=np.nan, right=np.nan)
        valid = np.isfinite(aligned)
        if np.count_nonzero(valid) < 20:
            return 1e30
        x = sig_ref[valid]
        y = aligned[valid]
        scale = optimal_scalar(x, y) if fit_scale else 1.0
        r = scale * x - y
        return float(np.mean(r * r))

    res = minimize_scalar(
        cost,
        bounds=(init_offset - search_range, init_offset + search_range),
        method="bounded",
    )
    return float(res.x) if res.success else float(init_offset)


def _initial_translation_from_magnitude(t_r, vr, t_ref, v_ref, w_ref, tau, scale=1.0):
    tq = t_r - tau
    vb = interp_vectors(tq, t_ref, v_ref)
    wb = interp_vectors(tq, t_ref, w_ref)
    valid = np.isfinite(vb).all(axis=1) & np.isfinite(wb).all(axis=1)
    if np.count_nonzero(valid) < 6:
        return np.zeros(3), valid
    vb_v = vb[valid]
    wb_v = wb[valid]
    vr_v = vr[valid]
    delta = np.linalg.norm(vr_v, axis=1) ** 2 - (scale ** 2) * np.linalg.norm(vb_v, axis=1) ** 2
    a = 2.0 * (scale ** 2) * np.cross(vb_v, wb_v)
    try:
        p, *_ = np.linalg.lstsq(a, delta, rcond=1e-6)
    except np.linalg.LinAlgError:
        p = np.zeros(3)
    if not np.isfinite(p).all() or np.linalg.norm(p) > 5.0:
        p = np.zeros(3)
    return p, valid


def _select_overlap(t_r, t_ref, tau0, search_range, margin=0.01):
    tq_min = t_ref[0] + search_range + margin
    tq_max = t_ref[-1] - search_range - margin
    tq = t_r - tau0
    return (tq >= tq_min) & (tq <= tq_max)


def _active_window(t, signal, threshold):
    idx = np.flatnonzero(np.asarray(signal) > threshold)
    if len(idx) == 0:
        return t, signal, None
    return t[idx[0]:idx[-1] + 1], signal[idx[0]:idx[-1] + 1], (float(t[idx[0]]), float(t[idx[-1]]))


def _odd_window_from_seconds(t, window_sec, min_window=5):
    if len(t) < min_window:
        return 0
    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 0
    n = int(round(float(window_sec) / float(np.median(dt)))) + 1
    n = max(n, min_window)
    if n % 2 == 0:
        n += 1
    if n >= len(t):
        n = len(t) - 1 if len(t) % 2 == 0 else len(t)
    if n < min_window:
        return 0
    if n % 2 == 0:
        n -= 1
    return n


def _lowpass_radar_velocity(t, velocity, filter_cfg):
    if not bool(filter_cfg.get("enabled", False)):
        return velocity.copy(), {
            "enabled": False,
            "method": "none",
        }
    method = str(filter_cfg.get("method", "savgol")).lower()
    window_sec = float(filter_cfg.get("window_sec", 0.45))
    polyorder = int(filter_cfg.get("polyorder", 2))
    window = _odd_window_from_seconds(t, window_sec)
    if method != "savgol":
        raise ValueError(f"Unsupported radar velocity filter method: {method}")
    if window <= polyorder + 2:
        return velocity.copy(), {
            "enabled": False,
            "method": method,
            "reason": "sequence too short for requested Savitzky-Golay filter",
            "requested_window_sec": window_sec,
            "window_samples": window,
            "polyorder": polyorder,
        }
    filtered = savgol_filter(
        velocity,
        window_length=window,
        polyorder=min(polyorder, window - 2),
        axis=0,
        mode=str(filter_cfg.get("edge_mode", "interp")),
    )
    return filtered, {
        "enabled": True,
        "method": method,
        "requested_window_sec": window_sec,
        "window_samples": int(window),
        "polyorder": int(min(polyorder, window - 2)),
        "edge_mode": str(filter_cfg.get("edge_mode", "interp")),
        "raw_speed_median": float(np.median(np.linalg.norm(velocity, axis=1))),
        "filtered_speed_median": float(np.median(np.linalg.norm(filtered, axis=1))),
        "delta_rms": float(np.sqrt(np.mean((filtered - velocity) ** 2))),
    }


def _read_radar_debug(path: Path) -> Dict[str, np.ndarray]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    out: Dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        vals = []
        numeric = True
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (TypeError, ValueError):
                numeric = False
                break
        if numeric:
            out[key] = np.asarray(vals, dtype=float)
    return out


def _radar_quality_mask(t_r, debug_path: Optional[Path], quality_cfg: Dict):
    if debug_path is None or not debug_path.exists() or not bool(quality_cfg.get("enabled", True)):
        return np.ones(len(t_r), dtype=bool), {
            "enabled": False,
            "reason": "no radar_debug CSV or disabled",
        }

    dbg = _read_radar_debug(debug_path)
    if "timestamp" not in dbg:
        return np.ones(len(t_r), dtype=bool), {
            "enabled": False,
            "debug_csv": str(debug_path),
            "reason": "timestamp column not found",
        }

    td = dbg["timestamp"]
    order = np.argsort(td)
    td = td[order]
    tol = float(quality_cfg.get("time_tolerance_sec", 0.03))
    idx = np.searchsorted(td, t_r)
    nearest = np.full(len(t_r), -1, dtype=int)
    best_dt = np.full(len(t_r), np.inf, dtype=float)
    for cand in (idx - 1, idx):
        ok = (cand >= 0) & (cand < len(td))
        dist = np.full(len(t_r), np.inf, dtype=float)
        dist[ok] = np.abs(td[cand[ok]] - t_r[ok])
        use = dist < best_dt
        nearest[use] = cand[use]
        best_dt[use] = dist[use]

    mask = best_dt <= tol
    reject_reasons = {"unmatched": int(np.count_nonzero(~mask))}

    def apply_threshold(column, predicate, reason):
        nonlocal mask
        if column not in dbg:
            return
        values = dbg[column][order][np.maximum(nearest, 0)]
        before = mask.copy()
        mask &= predicate(values)
        reject_reasons[reason] = int(np.count_nonzero(before & ~mask))

    if bool(quality_cfg.get("success_only", True)):
        apply_threshold("success", lambda v: v > 0.5, "success")
    if bool(quality_cfg.get("exclude_zero_velocity", True)):
        apply_threshold("zero_velocity", lambda v: v < 0.5, "zero_velocity")
    apply_threshold("valid_points", lambda v: v >= float(quality_cfg.get("min_valid_points", 10)), "valid_points")
    apply_threshold("unique_points", lambda v: v >= float(quality_cfg.get("min_unique_points", 8)), "unique_points")
    apply_threshold("inliers", lambda v: v >= float(quality_cfg.get("min_inliers", 6)), "inliers")
    apply_threshold("inlier_ratio", lambda v: v >= float(quality_cfg.get("min_inlier_ratio", 0.45)), "inlier_ratio")
    apply_threshold("cond", lambda v: v <= float(quality_cfg.get("max_cond", 30.0)), "cond")
    apply_threshold("rmse_inlier", lambda v: v <= float(quality_cfg.get("max_rmse_inlier", 0.15)), "rmse_inlier")

    return mask, {
        "enabled": True,
        "debug_csv": str(debug_path),
        "time_tolerance_sec": tol,
        "num_input": int(len(t_r)),
        "num_kept": int(np.count_nonzero(mask)),
        "kept_fraction": float(np.mean(mask)) if len(mask) else 0.0,
        "reject_reasons": reject_reasons,
        "thresholds": {
            "success_only": bool(quality_cfg.get("success_only", True)),
            "exclude_zero_velocity": bool(quality_cfg.get("exclude_zero_velocity", True)),
            "min_valid_points": float(quality_cfg.get("min_valid_points", 10)),
            "min_unique_points": float(quality_cfg.get("min_unique_points", 8)),
            "min_inliers": float(quality_cfg.get("min_inliers", 6)),
            "min_inlier_ratio": float(quality_cfg.get("min_inlier_ratio", 0.45)),
            "max_cond": float(quality_cfg.get("max_cond", 30.0)),
            "max_rmse_inlier": float(quality_cfg.get("max_rmse_inlier", 0.15)),
        },
    }


def _radar_speed_outlier_mask(velocity, cfg: Dict):
    speed = np.linalg.norm(np.asarray(velocity, dtype=float), axis=1)
    if not bool(cfg.get("enabled", True)):
        return np.ones(len(speed), dtype=bool), {
            "enabled": False,
            "reason": "disabled",
        }
    mask = np.isfinite(speed)
    absolute_max = cfg.get("max_speed_m_s", 2.0)
    if absolute_max is not None:
        mask &= speed <= float(absolute_max)
    median = float(np.nanmedian(speed))
    mad = float(np.nanmedian(np.abs(speed - median)))
    if mad > 1e-12:
        robust_max = median + float(cfg.get("mad_k", 8.0)) * 1.4826 * mad
        mask &= speed <= robust_max
    else:
        robust_max = float("inf")
    return mask, {
        "enabled": True,
        "num_input": int(len(speed)),
        "num_kept": int(np.count_nonzero(mask)),
        "kept_fraction": float(np.mean(mask)) if len(mask) else 0.0,
        "max_speed_m_s": float(absolute_max) if absolute_max is not None else None,
        "median_speed_m_s": median,
        "mad_speed_m_s": mad,
        "robust_max_speed_m_s": robust_max,
    }


def _interpolate_bad_velocity(t, velocity, good_mask):
    velocity = np.asarray(velocity, dtype=float)
    good_mask = np.asarray(good_mask, dtype=bool)
    if np.count_nonzero(good_mask) < 2:
        return velocity.copy(), {
            "enabled": False,
            "reason": "fewer than two good samples",
            "num_replaced": 0,
        }
    cleaned = velocity.copy()
    for i in range(velocity.shape[1]):
        cleaned[:, i] = np.interp(t, t[good_mask], velocity[good_mask, i])
    return cleaned, {
        "enabled": True,
        "num_replaced": int(np.count_nonzero(~good_mask)),
    }


def _save_radar_filter_diagnostics(out: Path, t, raw, cleaned, filtered, info):
    np.savetxt(
        out / "radar_velocity_filter_debug.csv",
        np.column_stack([t, raw, cleaned, filtered, cleaned - raw, filtered - cleaned]),
        delimiter=",",
        header="t,raw_vx,raw_vy,raw_vz,cleaned_vx,cleaned_vy,cleaned_vz,filtered_vx,filtered_vy,filtered_vz,clean_delta_vx,clean_delta_vy,clean_delta_vz,filter_delta_vx,filter_delta_vy,filter_delta_vz",
        comments="",
    )
    if not info.get("enabled", False):
        return
    t_rel = t - t[0]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)
    labels = ["vx", "vy", "vz"]
    for i, label in enumerate(labels):
        axes[i].plot(t_rel, raw[:, i], color="0.70", lw=0.8, label=f"raw {label}")
        axes[i].plot(t_rel, filtered[:, i], color=f"C{i}", lw=1.2, label=f"filtered {label}")
        axes[i].set_ylabel(f"{label} (m/s)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8, loc="upper right")
    axes[3].plot(t_rel, np.linalg.norm(raw, axis=1), color="0.70", lw=0.8, label="raw speed")
    axes[3].plot(t_rel, np.linalg.norm(filtered, axis=1), color="C3", lw=1.2, label="filtered speed")
    axes[3].set_ylabel("speed (m/s)")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Radar velocity low-pass filter ({info.get('method')}, window={info.get('window_samples')} samples, edge={info.get('edge_mode')})"
    )
    fig.tight_layout()
    fig.savefig(out / "radar_velocity_lowpass.png", dpi=240)
    plt.close(fig)


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def radar_body_prediction(params, t_r, vr, t_ref, v_ref, w_ref, scale_mode="global"):
    rvec = params[:3]
    p_br = params[3:6]
    tau = params[6]
    scale = float(np.exp(params[7])) if scale_mode == "global" else 1.0
    r_br = Rot.from_rotvec(rvec).as_matrix()
    tq = t_r - tau
    vb = interp_vectors(tq, t_ref, v_ref)
    wb = interp_vectors(tq, t_ref, w_ref)
    measured_body = (r_br @ vr.T).T
    predicted_body = scale * (vb + np.cross(wb, p_br))
    valid = np.isfinite(predicted_body).all(axis=1)
    return measured_body, predicted_body, valid


def radar_body_residual(params, t_r, vr, t_ref, v_ref, w_ref, scale_mode="global"):
    measured_body, predicted_body, valid = radar_body_prediction(
        params, t_r, vr, t_ref, v_ref, w_ref, scale_mode=scale_mode
    )
    residual = measured_body - predicted_body
    if not np.all(valid):
        residual = residual.copy()
        residual[~valid] = 1e3
    return residual.reshape(-1)


def _radar_translation_observability(params, result_jac, t_r, t_ref, v_ref, w_ref, residual_mat, scale_mode):
    p_br = params[3:6]
    tau = params[6]
    scale = float(np.exp(params[7])) if scale_mode == "global" else 1.0
    wb = interp_vectors(t_r - tau, t_ref, w_ref)
    j_blocks = np.asarray([-scale * _skew(w) for w in wb])
    j_analytic = j_blocks.reshape(-1, 3)
    h_analytic = j_analytic.T @ j_analytic
    sv_analytic = np.linalg.svd(j_analytic, compute_uv=False)
    j_numeric = result_jac[:, 3:6]
    h_numeric = j_numeric.T @ j_numeric
    sv_numeric = np.linalg.svd(j_numeric, compute_uv=False)
    lever = np.cross(wb, p_br)
    frame_rows = np.column_stack([
        t_r,
        wb,
        np.linalg.norm(wb, axis=1),
        np.linalg.norm(j_blocks.reshape(len(t_r), -1), axis=1),
        lever,
        np.linalg.norm(lever, axis=1),
        np.linalg.norm(residual_mat, axis=1),
    ])
    return {
        "analytic_jacobian": j_analytic,
        "analytic_hessian": h_analytic,
        "analytic_singular_values": sv_analytic,
        "numeric_hessian": h_numeric,
        "numeric_singular_values": sv_numeric,
        "condition_analytic": float(sv_analytic[0] / max(sv_analytic[-1], 1e-12)),
        "condition_numeric": float(sv_numeric[0] / max(sv_numeric[-1], 1e-12)),
        "frame_rows": frame_rows,
        "lever": lever,
        "omega_body": wb,
    }


def _save_radar_translation_diagnostics(out: Path, obs, payload: Dict):
    np.savetxt(
        out / "radar_translation_hessian_analytic.csv",
        obs["analytic_hessian"],
        delimiter=",",
        header="tx,ty,tz",
        comments="",
    )
    np.savetxt(
        out / "radar_translation_hessian_numeric.csv",
        obs["numeric_hessian"],
        delimiter=",",
        header="tx,ty,tz",
        comments="",
    )
    np.savetxt(
        out / "radar_translation_singular_values.csv",
        np.column_stack([obs["analytic_singular_values"], obs["numeric_singular_values"]]),
        delimiter=",",
        header="analytic,numeric_optimizer_jacobian",
        comments="",
    )
    np.savetxt(
        out / "radar_translation_frame_diagnostics.csv",
        obs["frame_rows"],
        delimiter=",",
        header="t,omega_x,omega_y,omega_z,omega_norm,jp_fro_norm,lever_x,lever_y,lever_z,lever_norm,residual_norm",
        comments="",
    )
    payload["translation_observability"] = {
        "analytic_singular_values": obs["analytic_singular_values"].tolist(),
        "numeric_singular_values": obs["numeric_singular_values"].tolist(),
        "analytic_condition": obs["condition_analytic"],
        "numeric_condition": obs["condition_numeric"],
        "lever_speed_median": float(np.median(np.linalg.norm(obs["lever"], axis=1))),
        "lever_speed_p95": float(np.percentile(np.linalg.norm(obs["lever"], axis=1), 95)),
        "omega_norm_median": float(np.median(np.linalg.norm(obs["omega_body"], axis=1))),
        "omega_norm_p95": float(np.percentile(np.linalg.norm(obs["omega_body"], axis=1), 95)),
    }


def _plot_radar_translation_observability(path: Path, t, obs):
    ensure_dir(Path(path).parent)
    rows = obs["frame_rows"]
    t_rel = t - t[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(t_rel, rows[:, 4], "0.35", lw=0.9, label="|omega|")
    axes[0, 0].set_title("Angular excitation")
    axes[0, 0].set_ylabel("rad/s")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_rel, rows[:, 5], "C0", lw=0.9, label="||J_p||")
    axes[0, 1].set_title("Translation Jacobian strength")
    axes[0, 1].set_ylabel("Frobenius norm")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_rel, rows[:, 9], "C3", lw=0.9, label="|omega x p|")
    axes[1, 0].set_title("Estimated lever-arm velocity")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    x = np.arange(3)
    axes[1, 1].bar(x - 0.16, obs["analytic_singular_values"], width=0.32, label="analytic")
    axes[1, 1].bar(x + 0.16, obs["numeric_singular_values"], width=0.32, label="optimizer")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["s1", "s2", "s3"])
    axes[1, 1].set_title("Translation singular values")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Radar Translation Observability Diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _summarize_result(result, residual, scale_mode):
    rms = float(np.sqrt(np.mean(residual * residual)))
    dof = len(result.x)
    sigma2 = float(np.sum(residual * residual) / max(len(residual) - dof, 1))
    cov = None
    std = None
    try:
        cov = np.linalg.pinv(result.jac.T @ result.jac) * sigma2
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        pass
    rot = Rot.from_rotvec(result.x[:3])
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "num_residuals": int(len(residual)),
        "cost": float(result.cost),
        "rms": rms,
        "sigma2": sigma2,
        "R_body_radar": rot.as_matrix().tolist(),
        "rotvec_body_radar": result.x[:3].tolist(),
        "euler_xyz_deg_body_radar": rot.as_euler("xyz", degrees=True).tolist(),
        "p_body_radar_m": result.x[3:6].tolist(),
        "time_offset_radar_minus_body_sec": float(result.x[6]),
        "velocity_scale": float(np.exp(result.x[7])) if scale_mode == "global" else 1.0,
        "scale_mode": scale_mode,
        "parameter_std": std.tolist() if std is not None else None,
        "covariance": cov.tolist() if cov is not None else None,
    }


def _write_radar_summary(path: Path, payload: Dict) -> None:
    lowpass = (payload.get("preprocessing") or {}).get("velocity_lowpass") or {}
    obs = payload.get("translation_observability") or {}
    rms = payload.get("rms")
    lever_p95 = obs.get("lever_speed_p95")
    rms_to_lever = None
    if rms is not None and lever_p95 is not None:
        rms_to_lever = float(rms) / max(float(lever_p95), 1e-9)
    lines = [
        "# Radar Calibration Stage",
        "",
        "## Preprocessing",
        f"- velocity low-pass enabled: {lowpass.get('enabled', False)}",
        f"- filter method: {lowpass.get('method', 'none')}",
        f"- filter window samples: {lowpass.get('window_samples', 'n/a')}",
        f"- filter edge mode: {lowpass.get('edge_mode', 'n/a')}",
        f"- filter delta RMS (m/s): {lowpass.get('delta_rms', 'n/a')}",
        "",
        "## Time initialization",
        f"- coarse offset adjust (s): {payload['time_initialization']['coarse_offset_adjust_sec']}",
        f"- cross-correlation offset (s): {payload['time_initialization']['cross_correlation_offset_sec']}",
        f"- initial time offset (s): {payload['time_initialization']['refined_offset_sec']}",
        f"- final time offset (s): {payload['time_offset_radar_minus_body_sec']}",
        f"- max correlation: {payload['time_initialization']['max_cross_correlation']}",
        "",
        "## Optimization",
        f"- initial RMS: {payload.get('initial_rms')}",
        f"- final RMS: {payload['rms']}",
        f"- velocity scale: {payload['velocity_scale']}",
        f"- Euler xyz body<-radar (deg): {payload['euler_xyz_deg_body_radar']}",
        f"- p_body_radar (m): {payload['p_body_radar_m']}",
        "",
        "## Translation Observability and Motion Quality",
        f"- omega norm median / p95 (rad/s): {obs.get('omega_norm_median', 'n/a')} / {obs.get('omega_norm_p95', 'n/a')}",
        f"- lever velocity median / p95 (m/s): {obs.get('lever_speed_median', 'n/a')} / {obs.get('lever_speed_p95', 'n/a')}",
        f"- translation singular values analytic: {obs.get('analytic_singular_values', 'n/a')}",
        f"- translation condition analytic / numeric: {obs.get('analytic_condition', 'n/a')} / {obs.get('numeric_condition', 'n/a')}",
        f"- RMS divided by lever p95: {rms_to_lever if rms_to_lever is not None else 'n/a'}",
        "- interpretation: translation is reliable when angular directions are well ranked and lever velocity is clearly above the residual floor.",
        "",
        "## Key figures",
        "- radar_velocity_lowpass.png",
        "- radar_time_alignment.png",
        "- radar_initial_body_velocity_fit.png",
        "- radar_body_velocity_fit.png",
        "- radar_residual_hist.png",
        "- overview.md  (new: end-of-stage consolidated comparison)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================
# Main stage (with per-step logging + plots)
# ======================================================================

def run_radar_reference_stage(stage_cfg: Dict, reference_cfg: Dict, output_root: Path, config_dir: Path) -> RadarStageResult:
    name = stage_cfg.get("name", "radar")
    out = output_root / name
    ensure_dir(out)
    lg = StageLogger(out)

    ref_path = Path(reference_cfg["twist"]).expanduser()
    inputs_cfg = stage_cfg.get("inputs", {})
    radar_path = Path(inputs_cfg["radar_velocity"]).expanduser()
    debug_path = Path(inputs_cfg["radar_debug"]).expanduser() if inputs_cfg.get("radar_debug") else None
    if not ref_path.is_absolute():
        ref_path = (config_dir / ref_path).resolve()
    if not radar_path.is_absolute():
        radar_path = (config_dir / radar_path).resolve()
    if debug_path is not None and not debug_path.is_absolute():
        debug_path = (config_dir / debug_path).resolve()

    io_cfg = stage_cfg.get("io") or {}
    t_ref, v_ref, w_ref = load_reference_twist(ref_path, delimiter=io_cfg.get("reference_delimiter"))
    t_r, vr = load_vector_series(
        radar_path,
        cols=stage_cfg.get("inputs", {}).get("radar_cols", "auto"),
        delimiter=io_cfg.get("radar_delimiter"),
    )
    lg.log(
        "01_io",
        n_ref=int(len(t_ref)), n_radar=int(len(t_r)),
        t_ref_span=[float(t_ref[0]), float(t_ref[-1])],
        t_r_span=[float(t_r[0]), float(t_r[-1])],
        ref_speed_median=float(np.median(np.linalg.norm(v_ref, axis=1))),
        radar_speed_median=float(np.median(np.linalg.norm(vr, axis=1))),
        ref_omega_median=float(np.median(np.linalg.norm(w_ref, axis=1))),
    )

    # ---------------- preprocessing ----------------
    preprocessing_cfg = stage_cfg.get("preprocessing") or {}
    trim_edges_sec = float(preprocessing_cfg.get("trim_edges_sec", 0.0))
    edge_mask = edge_trim_mask(t_r, trim_edges_sec)
    t_r, vr = t_r[edge_mask], vr[edge_mask]
    vr_raw = vr.copy()
    quality_cfg = preprocessing_cfg.get("quality_filter") or {}
    quality_mask_debug, quality_info = _radar_quality_mask(t_r, debug_path, quality_cfg)
    speed_mask, speed_info = _radar_speed_outlier_mask(vr_raw, preprocessing_cfg.get("speed_outlier_filter") or {})
    quality_mask = quality_mask_debug & speed_mask
    vr_clean, clean_info = _interpolate_bad_velocity(t_r, vr_raw, quality_mask)
    filter_cfg = preprocessing_cfg.get("velocity_lowpass") or {}
    vr, filter_info = _lowpass_radar_velocity(t_r, vr_clean, filter_cfg)

    lg.log(
        "02_preprocess",
        n_after_edge_trim=int(len(t_r)),
        quality_kept=int(quality_mask.sum()),
        quality_kept_frac=float(quality_mask.mean()) if len(quality_mask) else 0.0,
        speed_outlier_kept=int(speed_mask.sum()),
        interp_replaced=int(clean_info.get("num_replaced", 0)),
        lowpass_enabled=bool(filter_info.get("enabled", False)),
        lowpass_window_samples=filter_info.get("window_samples"),
        lowpass_delta_rms=filter_info.get("delta_rms"),
    )

    ensure_dir(out / "debug")
    np.savetxt(
        out / "debug" / "velocity_trajectories.csv",
        np.column_stack([t_r, vr_raw, vr_clean, vr]),
        delimiter=",",
        header="t,raw_x,raw_y,raw_z,clean_x,clean_y,clean_z,filt_x,filt_y,filt_z",
        comments="",
    )
    _plot_step_preprocess(out, t_r, vr_raw, vr_clean, vr, quality_mask)

    # ---------------- time init ----------------
    time_cfg = stage_cfg.get("time") or {}
    ablation = stage_cfg.get("ablation") or {}
    max_lag = float(time_cfg.get("max_lag_sec", 0.5))
    search_range = float(time_cfg.get("search_range_sec", 0.03))
    coarse_start = bool(time_cfg.get("coarse_start", False))
    coarse_adjust = float(time_cfg.get("coarse_offset_adjust_sec", 0.0))
    activity_threshold = float(time_cfg.get("activity_threshold", 0.05))
    trim_activity = bool(time_cfg.get("trim_activity_for_correlation", True))
    fit_scale_for_time = bool(time_cfg.get("fit_scale_for_time", True))

    ref_sig = np.linalg.norm(v_ref, axis=1)
    radar_sig = np.linalg.norm(vr, axis=1)
    raw_offset = (float(t_r[0] - t_ref[0]) if coarse_start else 0.0) + coarse_adjust
    t_r_coarse = t_r - raw_offset
    cc_t_ref, cc_ref_sig, ref_active_window = _active_window(t_ref, ref_sig, activity_threshold) if trim_activity else (t_ref, ref_sig, None)
    cc_t_r, cc_radar_sig, radar_active_window = _active_window(t_r_coarse, radar_sig, activity_threshold) if trim_activity else (t_r_coarse, radar_sig, None)
    if ablation.get("skip_time_init", False):
        off_cc, corr, lags = 0.0, np.zeros(1), np.zeros(1)
    else:
        off_cc, corr, lags = estimate_offset_cc(cc_t_ref, cc_ref_sig, cc_t_r, cc_radar_sig, max_lag_sec=max_lag)
    initial_offset = time_cfg.get("initial_offset_sec")
    if initial_offset is not None and not ablation.get("skip_time_init", False):
        raw_offset = 0.0
        t_r_coarse = t_r
        off_cc = float(initial_offset)
    off_refined = 0.0 if ablation.get("skip_time_init", False) else _refine_offset(
        t_ref, ref_sig, t_r_coarse, radar_sig, off_cc,
        search_range=search_range, fit_scale=fit_scale_for_time,
    )
    tau0 = raw_offset + off_refined
    peak_corr = float(np.max(corr))

    lg.log(
        "03a_coarse_offset",
        raw_offset=float(raw_offset),
        coarse_adjust=float(coarse_adjust),
        coarse_start=bool(coarse_start),
        t_r_coarse_span=[float(t_r_coarse[0]), float(t_r_coarse[-1])],
        ref_active_window=ref_active_window,
        radar_active_window=radar_active_window,
    )
    lg.log(
        "03b_cross_correlation",
        off_cc=float(off_cc),
        peak_corr=peak_corr,
        lag_at_peak=float(lags[int(np.argmax(corr))]),
        max_lag_sec=max_lag,
        activity_threshold=activity_threshold,
        trim_activity=bool(trim_activity),
    )
    lg.log(
        "03c_refine",
        off_refined=float(off_refined),
        tau0=float(tau0),
        search_range=search_range,
        fit_scale_for_time=bool(fit_scale_for_time),
    )

    # ---------------- init translation ----------------
    init_cfg = stage_cfg.get("initialization") or {}
    t_init = t_r[quality_mask] if np.count_nonzero(quality_mask) >= 20 else t_r
    vr_init = vr[quality_mask] if np.count_nonzero(quality_mask) >= 20 else vr
    if ablation.get("skip_translation_init", False):
        p0, mag_valid = np.zeros(3), np.zeros(len(t_init), dtype=bool)
    else:
        p0, mag_valid = _initial_translation_from_magnitude(
            t_init, vr_init, t_ref, v_ref, w_ref, tau0, scale=1.0,
        )
    if "translation_m" in init_cfg and not ablation.get("skip_translation_init", False):
        p0 = np.asarray(init_cfg["translation_m"], dtype=float)

    if np.count_nonzero(mag_valid) > 6:
        tt = t_init[mag_valid] - tau0
        vb_m = interp_vectors(tt, t_ref, v_ref)
        wb_m = interp_vectors(tt, t_ref, w_ref)
        vr_m = vr_init[mag_valid]
        pred_m = vb_m + np.cross(wb_m, p0)
        mag_res = np.linalg.norm(vr_m, axis=1) ** 2 - np.linalg.norm(pred_m, axis=1) ** 2
        mag_res_rms = float(np.sqrt(np.mean(mag_res ** 2)))
    else:
        mag_res_rms = None

    lg.log(
        "04_init_translation",
        p0_m=p0.tolist(),
        magnitude_valid_samples=int(np.count_nonzero(mag_valid)),
        magnitude_residual_rms=mag_res_rms,
        override_from_cfg=("translation_m" in init_cfg),
    )

    # ---------------- init rotation (Kabsch) ----------------
    opt_cfg = stage_cfg.get("optimization") or {}
    opt_trim_edges_sec = float(opt_cfg.get("trim_edges_sec", 0.0))
    optimization_edge_mask = edge_trim_mask(t_r, opt_trim_edges_sec)
    common_radius = float(ablation.get("common_time_radius_sec", 0.0))
    overlap_center = raw_offset if common_radius > 0.0 else tau0
    overlap_radius = common_radius if common_radius > 0.0 else search_range
    valid = _select_overlap(t_r, t_ref, overlap_center, overlap_radius) & quality_mask & optimization_edge_mask
    if np.count_nonzero(valid) < 20:
        lg.log("error_too_few_samples", n_valid=int(np.count_nonzero(valid)), tau0=float(tau0))
        lg.flush()
        raise ValueError(
            f"Too few radar samples after overlap and quality filtering: {np.count_nonzero(valid)}"
        )
    tq0 = t_r[valid] - tau0
    vb0 = interp_vectors(tq0, t_ref, v_ref)
    wb0 = interp_vectors(tq0, t_ref, w_ref)
    body_comp = vb0 + np.cross(wb0, p0)
    r0 = np.eye(3) if ablation.get("skip_rotation_init", False) else kabsch_body_from_sensor(vr[valid], body_comp)
    if ablation.get("skip_rotation_init", False):
        rvec0 = np.zeros(3)
    elif "rotvec_body_radar" in init_cfg:
        rvec0 = np.asarray(init_cfg["rotvec_body_radar"], dtype=float)
    else:
        rvec0 = Rot.from_matrix(r0).as_rotvec()

    kabsch_meas = (np.asarray(r0) @ vr[valid].T).T
    kabsch_res = kabsch_meas - body_comp
    kabsch_rms = float(np.sqrt(np.mean(kabsch_res ** 2)))
    lg.log(
        "05_init_rotation",
        rvec0=rvec0.tolist(),
        euler_deg=Rot.from_rotvec(rvec0).as_euler("xyz", degrees=True).tolist(),
        kabsch_rms=kabsch_rms,
        override_from_cfg=("rotvec_body_radar" in init_cfg),
        n_samples_valid=int(np.count_nonzero(valid)),
    )
    _plot_step_kabsch(out, t_r[valid], kabsch_meas, body_comp, kabsch_res, kabsch_rms)

    # ---------------- optimizer ----------------
    scale_mode = opt_cfg.get("scale_mode", "global")
    if scale_mode not in {"fixed", "global"}:
        raise ValueError(f"Unsupported radar scale_mode: {scale_mode}")
    log_scale0 = float(np.log(max(optimal_scalar(np.linalg.norm(body_comp, axis=1),
                                                 np.linalg.norm(vr[valid], axis=1)), 1e-6)))
    if scale_mode == "fixed":
        log_scale0 = 0.0
    x0 = np.r_[rvec0, p0, tau0, log_scale0]

    tau_window = float(opt_cfg.get("tau_refine_window_sec", search_range))
    rot_bound = float(opt_cfg.get("rotvec_bound_rad", np.pi))
    trans_bound = float(opt_cfg.get("translation_bound_m", 5.0))
    scale_bound = float(opt_cfg.get("log_scale_bound", 1.0))
    lower = np.array([-rot_bound, -rot_bound, -rot_bound, -trans_bound, -trans_bound, -trans_bound, tau0 - tau_window, -scale_bound])
    upper = np.array([rot_bound, rot_bound, rot_bound, trans_bound, trans_bound, trans_bound, tau0 + tau_window, scale_bound])
    if common_radius > 0.0:
        lower[6], upper[6] = raw_offset - common_radius, raw_offset + common_radius
    if scale_mode == "fixed":
        lower[7] = -1e-12
        upper[7] = 1e-12
        x0[7] = 0.0

    tr = t_r[valid]
    vrr = vr[valid]
    loss = str(opt_cfg.get("loss", "huber"))
    f_scale = float(opt_cfg.get("f_scale", 0.2))
    max_nfev = int(opt_cfg.get("max_nfev", 300))

    cost_hist = []

    def _cb(xk):
        c = np.sum(radar_body_residual(xk, tr, vrr, t_ref, v_ref, w_ref, scale_mode) ** 2)
        cost_hist.append(float(c))
        return None

    _has_callback = "callback" in inspect.signature(least_squares).parameters

    lg.log(
        "06_optimizer_start",
        x0=x0.tolist(),
        lower=lower.tolist(),
        upper=upper.tolist(),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        scale_mode=scale_mode,
        n_valid_samples=int(np.count_nonzero(valid)),
        scipy_has_callback=bool(_has_callback),
    )

    _ls_kwargs = dict(
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        bounds=(lower, upper),
    )

    if _has_callback:
        # scipy >= 1.1: use the official callback, one record per accepted step
        result = least_squares(
            radar_body_residual,
            x0,
            args=(tr, vrr, t_ref, v_ref, w_ref, scale_mode),
            callback=_cb,
            **_ls_kwargs,
        )
    else:
        # scipy < 1.1 fallback: record cost inside a residual wrapper
        # NOTE: this records every residual evaluation (line-search included),
        # so cost_hist is denser but still shows the convergence trend.
        def _residual_logged(params, *args):
            r = radar_body_residual(params, *args)
            cost_hist.append(float(np.sum(r ** 2)))
            return r

        result = least_squares(
            _residual_logged,
            x0,
            args=(tr, vrr, t_ref, v_ref, w_ref, scale_mode),
            **_ls_kwargs,
        )

    residual = radar_body_residual(result.x, tr, vrr, t_ref, v_ref, w_ref, scale_mode)
    lg.log(
        "06_optimizer_end",
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        njev=int(getattr(result, "njev", 0)) if getattr(result, "njev", None) is not None else None,
        cost=float(result.cost),
        rms_final=float(np.sqrt(np.mean(residual ** 2))),
        n_cost_samples=len(cost_hist),
        cost_first=cost_hist[0] if cost_hist else None,
        cost_last=cost_hist[-1] if cost_hist else None,
    )
    if cost_hist:
        np.savetxt(out / "debug" / "optimizer_cost_curve.csv",
                   np.asarray(cost_hist), header="cost", comments="", fmt="%.6f")
    _plot_step_optimizer(out, cost_hist)

    body_meas, body_pred, pred_valid = radar_body_prediction(
        result.x, tr, vrr, t_ref, v_ref, w_ref, scale_mode
    )
    residual_mat = residual.reshape(-1, 3)
    per_axis_rms = np.sqrt(np.mean(residual_mat ** 2, axis=0))
    per_axis_bias = residual_mat.mean(axis=0)
    abs_res = np.abs(residual)
    rms_final = float(np.sqrt(np.mean(residual ** 2)))
    p95 = float(np.percentile(abs_res, 95))
    p99 = float(np.percentile(abs_res, 99))
    lg.log(
        "07_residual",
        rms_overall=rms_final,
        rms_xyz=per_axis_rms.tolist(),
        bias_xyz=per_axis_bias.tolist(),
        p95=p95,
        p99=p99,
        max=float(np.max(abs_res)),
    )
    _plot_step_residual(out, tr, residual_mat)

    # ---------------- compare init vs final ----------------
    initial_residual = radar_body_residual(x0, tr, vrr, t_ref, v_ref, w_ref, scale_mode)
    rms_init = float(np.sqrt(np.mean(initial_residual ** 2)))
    improvement = float(rms_init / max(rms_final, 1e-9))
    lg.log(
        "08_compare",
        rms_init=rms_init,
        rms_final=rms_final,
        improvement_x=improvement,
        scale_init=float(np.exp(x0[7])) if scale_mode == "global" else 1.0,
        scale_final=float(np.exp(result.x[7])) if scale_mode == "global" else 1.0,
    )

    # ---------------- payload (unchanged fields) ----------------
    payload = {
        "stage": name,
        "type": "radar_reference",
        "frame_convention": {
            "rotation": "R_body_radar maps radar-frame velocity to body frame",
            "translation": "p_body_radar is radar origin expressed in body frame",
            "time": "t_body = t_radar - tau",
            "residual": "R_body_radar * v_radar - s * (v_body + omega_body x p_body_radar)",
        },
        "inputs": {
            "reference_twist": str(ref_path),
            "radar_velocity": str(radar_path),
            "radar_debug": str(debug_path) if debug_path is not None else None,
            "radar_cols": inputs_cfg.get("radar_cols", "auto"),
        },
        "time_initialization": {
            "raw_start_difference_sec": raw_offset,
            "coarse_offset_adjust_sec": coarse_adjust,
            "initial_offset_sec": float(initial_offset) if initial_offset is not None else None,
            "activity_threshold": activity_threshold,
            "ref_active_window_after_epoch_align": ref_active_window,
            "radar_active_window_after_epoch_align": radar_active_window,
            "cross_correlation_offset_sec": off_cc,
            "refined_offset_sec": tau0,
            "max_cross_correlation": peak_corr,
        },
        "initial_guess": {
            "rotvec_body_radar": rvec0.tolist(),
            "p_body_radar_m": p0.tolist(),
            "time_offset_sec": tau0,
            "log_scale": log_scale0,
            "magnitude_translation_samples": int(np.count_nonzero(mag_valid)),
        },
        "preprocessing": {
            "trim_edges_sec": trim_edges_sec,
            "optimization_trim_edges_sec": opt_trim_edges_sec,
            "num_samples_after_edge_trim": int(len(t_r)),
            "velocity_lowpass": filter_info,
            "quality_filter": quality_info,
            "speed_outlier_filter": speed_info,
            "pre_lowpass_interpolation": clean_info,
        },
    }
    payload.update(_summarize_result(result, residual, scale_mode))
    payload["T_body_radar"] = make_se3(payload["R_body_radar"], payload["p_body_radar_m"])
    payload["initial_rms"] = rms_init

    translation_obs = _radar_translation_observability(
        result.x, result.jac, tr, t_ref, v_ref, w_ref, residual_mat, scale_mode
    )
    _save_radar_translation_diagnostics(out, translation_obs, payload)

    # ---------------- save artifacts ----------------
    result_json = out / "radar_result.json"
    data_npz = out / "radar_stage_data.npz"
    save_json(result_json, payload)
    _write_radar_summary(out / "stage_summary.md", payload)
    _save_radar_filter_diagnostics(out, t_r, vr_raw, vr_clean, vr, filter_info)
    np.savetxt(
        out / "radar_quality_mask.csv",
        np.column_stack([t_r, np.linalg.norm(vr_raw, axis=1), np.linalg.norm(vr, axis=1), quality_mask.astype(int)]),
        delimiter=",",
        header="t_radar,raw_speed,filtered_speed,quality_kept",
        comments="",
    )
    np.savez_compressed(
        data_npz,
        t_radar=tr,
        radar_velocity=vrr,
        radar_velocity_raw=vr_raw[valid],
        body_velocity_measured=body_meas,
        body_velocity_predicted=body_pred,
        residual=residual_mat,
        params=result.x,
        initial_params=x0,
        valid=pred_valid,
    )
    np.savetxt(
        out / "radar_aligned_debug.csv",
        np.column_stack([tr, vrr, body_meas, body_pred, residual_mat]),
        delimiter=",",
        header="t_radar,radar_vx,radar_vy,radar_vz,body_meas_x,body_meas_y,body_meas_z,body_pred_x,body_pred_y,body_pred_z,res_x,res_y,res_z",
        comments="",
    )
    np.savetxt(
        out / "radar_step_metrics.csv",
        np.array([
            ["coarse_offset_adjust_sec", coarse_adjust],
            ["cross_correlation_offset_sec", off_cc],
            ["initial_time_offset_sec", tau0],
            ["final_time_offset_sec", float(result.x[6])],
            ["initial_rms", rms_init],
            ["final_rms", rms_final],
            ["improvement_x", improvement],
            ["initial_scale", float(np.exp(x0[7])) if scale_mode == "global" else 1.0],
            ["final_scale", payload["velocity_scale"]],
        ], dtype=object),
        delimiter=",",
        fmt="%s",
    )

    # ---------------- final plots (existing) ----------------
    diag_cfg = stage_cfg.get("diagnostics") or {}
    if bool(diag_cfg.get("save_plots", True)):
        plot_cross_correlation(out / "radar_cross_correlation.png", lags, corr, off_cc, "Radar/reference speed cross-correlation")
        plot_scalar_series(
            out / "radar_speed_activity.png",
            [
                ("reference speed", t_ref, ref_sig, "k-"),
                ("radar speed epoch-aligned", t_r_coarse, radar_sig, "0.55"),
                ("radar speed final-aligned", t_r - float(result.x[6]), radar_sig, "C3"),
            ],
            "Radar speed activity and time alignment",
            "speed (m/s)",
        )
        plot_offset_cost_curve(
            out / "radar_time_offset_cost.png",
            t_ref,
            ref_sig,
            t_r,
            radar_sig,
            tau0,
            max(search_range * 3.0, 0.05),
            fit_scale=fit_scale_for_time,
        )
        plot_alignment(
            out / "radar_time_alignment.png",
            t_ref,
            ref_sig,
            t_r,
            radar_sig,
            float(result.x[6]),
            "Radar-reference speed alignment",
            "speed (m/s)",
        )
        initial_body_meas, initial_body_pred, _ = radar_body_prediction(
            x0, tr, vrr, t_ref, v_ref, w_ref, scale_mode
        )
        plot_components(out / "radar_initial_body_velocity_fit.png", tr, initial_body_meas, initial_body_pred,
                        initial_residual, "Radar initial velocity fit in body frame")
        plot_components(out / "radar_body_velocity_fit.png", tr, body_meas, body_pred, residual,
                        "Radar velocity calibration in body frame")
        plot_vector_components(
            out / "radar_extrinsic_alignment_effect.png",
            tr,
            [
                ("radar raw as body", vrr, "0.70"),
                ("radar rotated to body", body_meas, "C0"),
                ("reference rigid prediction", body_pred, "C3"),
            ],
            ["x", "y", "z"],
            "Radar spatial calibration effect",
            "m/s",
        )
        _plot_radar_translation_observability(out / "radar_translation_observability.png", tr, translation_obs)
        plot_residual_hist(out / "radar_residual_hist.png", residual, "Radar velocity residuals", "residual (m/s)")

    # ---------------- final overview ----------------
    debug_summary = {
        "n_ref": int(len(t_ref)),
        "n_radar": int(len(t_r)),
        "quality_kept_frac": float(quality_mask.mean()) if len(quality_mask) else 0.0,
        "peak_corr": peak_corr,
        "rms_init": rms_init,
        "rms_final": rms_final,
        "improvement_x": improvement,
        "rms_xyz": per_axis_rms.tolist(),
        "bias_xyz": per_axis_bias.tolist(),
        "p95": p95,
        "p99": p99,
    }
    write_overview(out, payload, debug_summary)
    lg.log("09_done", overview=str(out / "overview.md"), stage_log=str(out / "debug" / "stage_log.json"))
    lg.flush()

    return RadarStageResult(name=name, output_dir=out, result_json=result_json, data_npz=data_npz, payload=payload)

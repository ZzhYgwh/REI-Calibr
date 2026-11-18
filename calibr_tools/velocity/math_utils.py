from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import correlate, correlation_lags
from scipy.spatial.transform import Rotation as Rot


EPS = 1e-12


def interp_vectors(t_query, t_ref, y_ref):
    out = np.empty((len(t_query), y_ref.shape[1]))
    for i in range(y_ref.shape[1]):
        out[:, i] = np.interp(t_query, t_ref, y_ref[:, i], left=np.nan, right=np.nan)
    return out


def estimate_offset_cc(t_ref, sig_ref, t_sens, sig_sens, max_lag_sec=0.5, dt_frac=5):
    dt = min(np.median(np.diff(t_ref)), np.median(np.diff(t_sens))) / float(dt_frac)
    t0 = max(t_ref[0], t_sens[0])
    t1 = min(t_ref[-1], t_sens[-1])
    if not np.isfinite(dt) or dt <= 0 or t1 <= t0:
        raise ValueError("No valid time overlap for cross-correlation.")
    common_t = np.arange(t0, t1, dt)
    a = np.interp(common_t, t_ref, sig_ref)
    b = np.interp(common_t, t_sens, sig_sens)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = max(float(np.std(a) * np.std(b) * len(a)), EPS)
    max_lag = int(max_lag_sec / dt)
    corr_raw = correlate(b, a, mode="full")
    lags = correlation_lags(len(b), len(a), mode="full")
    valid = np.abs(lags) <= max_lag
    corr = corr_raw[valid] / denom
    lag_sec = lags[valid] * dt
    idx = int(np.argmax(corr))
    return float(lag_sec[idx]), corr, lag_sec


def optimal_scalar(x, y):
    denom = float(np.dot(x, x))
    if denom < EPS:
        return 1.0
    return float(np.dot(x, y) / denom)


def edge_trim_mask(t, trim_sec):
    t = np.asarray(t, dtype=float)
    trim_sec = float(trim_sec or 0.0)
    if len(t) == 0 or trim_sec <= 0.0:
        return np.ones(len(t), dtype=bool)
    return (t >= t[0] + trim_sec) & (t <= t[-1] - trim_sec)


def odd_window_from_seconds(t, window_sec, min_window=5):
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


def smooth_vector_savgol(t, values, enabled=False, window_sec=0.45, polyorder=2, edge_mode="interp"):
    values = np.asarray(values, dtype=float)
    if not bool(enabled):
        return values.copy(), {"enabled": False, "method": "none"}
    window = odd_window_from_seconds(t, window_sec)
    if window <= int(polyorder) + 2:
        return values.copy(), {
            "enabled": False,
            "method": "savgol",
            "reason": "sequence too short",
            "requested_window_sec": float(window_sec),
            "window_samples": int(window),
            "polyorder": int(polyorder),
        }
    filtered = savgol_filter(
        values,
        window_length=window,
        polyorder=min(int(polyorder), window - 2),
        axis=0,
        mode=str(edge_mode),
    )
    return filtered, {
        "enabled": True,
        "method": "savgol",
        "requested_window_sec": float(window_sec),
        "window_samples": int(window),
        "polyorder": int(min(int(polyorder), window - 2)),
        "edge_mode": str(edge_mode),
        "delta_rms": float(np.sqrt(np.mean((filtered - values) ** 2))),
    }


def kabsch_body_from_sensor(sensor_vec, body_vec):
    sensor_vec = np.asarray(sensor_vec)
    body_vec = np.asarray(body_vec)
    valid = np.isfinite(sensor_vec).all(axis=1) & np.isfinite(body_vec).all(axis=1)
    a = sensor_vec[valid]
    b = body_vec[valid]
    if len(a) < 3:
        return np.eye(3)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def rotation_error_deg(ra, rb):
    return float(np.linalg.norm(Rot.from_matrix(ra @ rb.T).as_rotvec()) * 180.0 / math.pi)

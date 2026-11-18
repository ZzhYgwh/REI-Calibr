from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as Rot, Slerp

from .io import ensure_dir, load_imu, load_reference_twist, make_se3, save_json
from .math_utils import (
    edge_trim_mask,
    estimate_offset_cc,
    interp_vectors,
    kabsch_body_from_sensor,
    optimal_scalar,
    smooth_vector_savgol,
)
from .plotting import plot_alignment, plot_cross_correlation, plot_residual_hist, plot_vector_components


@dataclass
class ImuStageResult:
    name: str
    output_dir: Path
    result_json: Path
    data_npz: Path
    payload: Dict


def _refine_offset(t_ref, sig_ref, t_sens, sig_sens, init_offset, search_range):
    def cost(dt):
        aligned = np.interp(t_ref, t_sens - dt, sig_sens, left=np.nan, right=np.nan)
        valid = np.isfinite(aligned)
        if np.count_nonzero(valid) < 20:
            return 1e30
        r = aligned[valid] - sig_ref[valid]
        return float(np.mean(r * r))

    res = minimize_scalar(
        cost,
        bounds=(init_offset - search_range, init_offset + search_range),
        method="bounded",
    )
    return float(res.x) if res.success else float(init_offset)


def _initial_body_bias(t_imu, gyro, t_ref, w_ref, tau):
    w = interp_vectors(t_imu - tau, t_ref, w_ref)
    valid = np.isfinite(w).all(axis=1)
    if np.count_nonzero(valid) < 6:
        return np.zeros(3), valid
    delta = np.linalg.norm(gyro[valid], axis=1) ** 2 - np.linalg.norm(w[valid], axis=1) ** 2
    a = 2.0 * w[valid]
    try:
        c, *_ = np.linalg.lstsq(a, delta, rcond=1e-6)
    except np.linalg.LinAlgError:
        c = np.zeros(3)
    if not np.isfinite(c).all() or np.linalg.norm(c) > 5.0:
        c = np.zeros(3)
    return c, valid


def imu_gyro_prediction(params, t_imu, t_ref, w_ref):
    rvec = params[:3]
    tau = params[3]
    bg = params[4:7]
    r_bi = Rot.from_rotvec(rvec).as_matrix()
    r_ib = r_bi.T
    wb = interp_vectors(t_imu - tau, t_ref, w_ref)
    pred = (r_ib @ wb.T).T + bg
    valid = np.isfinite(pred).all(axis=1)
    return pred, valid


def imu_gyro_residual(params, t_imu, gyro, t_ref, w_ref):
    pred, valid = imu_gyro_prediction(params, t_imu, t_ref, w_ref)
    residual = gyro - pred
    if not np.all(valid):
        residual = residual.copy()
        residual[~valid] = 1e3
    return residual.reshape(-1)


def imu_accel_window_residual(params, t_imu, accel, t_ref, velocity, omega,
                              window_sec=0.1, step_sec=0.05, bias_sigma=0.1,
                              weighted=True, regularize=True):
    """Re-evaluate the acceleration-stage local-gravity-cancelling windows."""
    if len(t_imu) != len(accel):
        raise ValueError("Acceleration and IMU timestamps must have equal length")
    if len(t_imu) < 12:
        raise ValueError("At least 12 IMU acceleration samples are required")
    dt = np.median(np.diff(t_imu))
    win = max(int(window_sec / dt), 10)
    step = max(int(step_sec / dt), 1)
    starts = np.arange(0, len(t_imu) - win, step)
    if len(starts) == 0:
        raise ValueError("No acceleration windows fit the IMU time span")
    ends = starts + win

    gyro_params = params[:7]
    p, ba, scale = params[7:10], params[10:13], params[13]
    rotation = Rot.from_rotvec(gyro_params[:3]).as_matrix()
    query = t_imu - gyro_params[3]
    ref_accel = _numerical_derivative(t_ref, velocity) + np.cross(omega, velocity)
    ref_alpha = _numerical_derivative(t_ref, omega)
    w = interp_vectors(query, t_ref, omega)
    alpha = interp_vectors(query, t_ref, ref_alpha)
    a_ref = interp_vectors(query, t_ref, ref_accel)
    if not (np.isfinite(w).all() and np.isfinite(alpha).all() and np.isfinite(a_ref).all()):
        count = len(starts) * 3 + (4 if regularize else 0)
        return np.full(count, 1e3)

    def skew(v):
        m = np.zeros((len(v), 3, 3))
        m[:, 0, 1], m[:, 0, 2] = -v[:, 2], v[:, 1]
        m[:, 1, 0], m[:, 1, 2] = v[:, 2], -v[:, 0]
        m[:, 2, 0], m[:, 2, 1] = -v[:, 1], v[:, 0]
        return m

    cross = skew(w)
    lever = skew(alpha) + cross @ cross
    a_imu_body = accel @ rotation.T
    dt_arr = np.diff(query)

    def integral(values):
        increments = 0.5 * (values[:-1] + values[1:]) * dt_arr.reshape((-1,) + (1,) * (values.ndim - 1))
        return np.concatenate((np.zeros_like(values[:1]), np.cumsum(increments, axis=0)))

    cross_int = integral(cross)
    lever_int = integral(cross @ lever)
    imu_int = integral(np.einsum("nij,nj->ni", cross, a_imu_body))
    ref_int = integral(np.einsum("nij,nj->ni", cross, a_ref))
    a_win = lever[ends] - lever[starts] + lever_int[ends] - lever_int[starts]
    b_win = cross_int[ends] - cross_int[starts]
    imu_win = a_imu_body[ends] - a_imu_body[starts] + imu_int[ends] - imu_int[starts]
    ref_win = a_ref[ends] - a_ref[starts] + ref_int[ends] - ref_int[starts]
    residual = np.einsum("nij,j->ni", a_win, p)
    residual += np.einsum("nij,j->ni", b_win, ba) + scale * ref_win - imu_win
    if weighted:
        omega_norm = np.linalg.norm(w, axis=1)
        omega_sum = np.r_[0.0, np.cumsum(omega_norm)]
        omega_mean = (omega_sum[ends] - omega_sum[starts]) / win
        p95 = max(np.percentile(omega_mean, 95), 1e-6)
        residual *= np.clip(omega_mean / p95, 0.1, 1.0)[:, None]
    flat = residual.reshape(-1)
    if regularize:
        flat = np.r_[flat, ba / bias_sigma, scale - 1.0]
    return flat


def _numerical_derivative(t, y):
    out = np.empty_like(y)
    for i in range(y.shape[1]):
        out[:, i] = np.gradient(y[:, i], t)
    return out


def _finite_flat_scale(reference, measured):
    valid = np.isfinite(reference).all(axis=1) & np.isfinite(measured).all(axis=1)
    if np.count_nonzero(valid) < 6:
        return 1.0
    return optimal_scalar(reference[valid].reshape(-1), measured[valid].reshape(-1))


def _finite_rms_scale(reference, measured):
    valid = np.isfinite(reference).all(axis=1) & np.isfinite(measured).all(axis=1)
    if np.count_nonzero(valid) < 6:
        return 1.0
    ref_rms = float(np.sqrt(np.mean(reference[valid] ** 2)))
    meas_rms = float(np.sqrt(np.mean(measured[valid] ** 2)))
    if ref_rms < 1e-12:
        return 1.0
    return meas_rms / ref_rms


def _integrate_reference_attitude(t_ref, omega):
    """Integrate body rates into body-to-initial-body rotations."""
    if len(t_ref) < 2 or not np.all(np.diff(t_ref) > 0):
        raise ValueError("Reference timestamps must be strictly increasing")
    increments = 0.5 * (omega[:-1] + omega[1:]) * np.diff(t_ref)[:, None]
    rotations = np.empty((len(t_ref), 4))
    attitude = Rot.identity()
    rotations[0] = attitude.as_quat()
    for i, delta in enumerate(increments, 1):
        attitude = attitude * Rot.from_rotvec(delta)
        rotations[i] = attitude.as_quat()
    return Slerp(t_ref, Rot.from_quat(rotations))


def _summarize(result, residual):
    rms = float(np.sqrt(np.mean(residual * residual)))
    sigma2 = float(np.sum(residual * residual) / max(len(residual) - len(result.x), 1))
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
        "R_body_imu": rot.as_matrix().tolist(),
        "rotvec_body_imu": result.x[:3].tolist(),
        "euler_xyz_deg_body_imu": rot.as_euler("xyz", degrees=True).tolist(),
        "p_body_imu_m": [0.0, 0.0, 0.0],
        "time_offset_imu_minus_body_sec": float(result.x[3]),
        "gyro_bias_imu_rad_s": result.x[4:7].tolist(),
        "accel_bias_imu_m_s2": [0.0, 0.0, 0.0],
        "parameter_std": std.tolist() if std is not None else None,
        "covariance": cov.tolist() if cov is not None else None,
    }


# ============================================================
# === 加速度计标定：相对运动预积分 + 消去重力 + 幅值联合估计 ===
# ============================================================
def estimate_imu_translation(t_imu, accel, rotation, tau, t_ref, velocity, omega, options, debug_store=None):
    """
    滑动窗口相对积分（预积分）法 + 幅值自适应（Scale） + 窗口去均值 + Bias正则化。
    """
    print("\n" + "-" * 70)
    print(">>> [加速度计标定 - 预积分消元 + 幅值联合估计] 开始")
    print("-" * 70)

    # ========== Step A: 真值数据准备 ==========
    query = t_imu - tau
    ref_accel = _numerical_derivative(t_ref, velocity) + np.cross(omega, velocity)
    angular_accel = _numerical_derivative(t_ref, omega)

    angular = interp_vectors(query, t_ref, omega)
    angular_rate = interp_vectors(query, t_ref, angular_accel)
    acceleration = interp_vectors(query, t_ref, ref_accel)

    valid = np.isfinite(acceleration).all(axis=1) & np.isfinite(angular_rate).all(axis=1)
    valid &= np.isfinite(angular).all(axis=1) & np.isfinite(accel).all(axis=1)
    valid &= (query >= t_ref[0]) & (query <= t_ref[-1])
    if np.count_nonzero(valid) < 20:
        print(f"  [FAIL] 有效样本 {np.count_nonzero(valid)} < 20，无法标定")
        return {"status": "unavailable", "reason": "Insufficient overlapping acceleration and angular-rate samples"}

    angular, angular_rate = angular[valid], angular_rate[valid]
    t_valid = query[valid]
    a_m_valid = accel[valid]
    a_ref_valid = acceleration[valid]

    print(f"  [A] 有效样本数: {len(angular)} / {len(t_imu)}")
    print(f"      时间跨度: {t_valid[-1] - t_valid[0]:.2f} s")

    # ========== Step B: 构建物理矩阵 L ==========
    cross_matrix = np.zeros((len(angular), 3, 3))
    rate_matrix = np.zeros_like(cross_matrix)
    for values, matrix in [(angular, cross_matrix), (angular_rate, rate_matrix)]:
        matrix[:, 0, 1], matrix[:, 0, 2] = -values[:, 2], values[:, 1]
        matrix[:, 1, 0], matrix[:, 1, 2] = values[:, 2], -values[:, 0]
        matrix[:, 2, 0], matrix[:, 2, 1] = -values[:, 1], values[:, 0]
    
    lever = rate_matrix + cross_matrix @ cross_matrix  # [α]× + [ω]×²
    
    # 拆分 u 项，用于估计缩放因子 s
    u_am = a_m_valid @ rotation.T
    u_aref = a_ref_valid

    # ========== Step C: 滑动窗口预积分堆叠 (梯形积分) ==========
    dt_median = np.median(np.diff(t_valid))
    win_size = int(options.get("window_sec", 0.1) / dt_median)
    step_size = int(options.get("step_sec", 0.05) / dt_median)
    win_size = max(win_size, 10)
    step_size = max(step_size, 1)

    cross_L = cross_matrix @ lever  # [N, 3, 3]
    cross_u_am = (cross_matrix @ u_am[:, :, None]).squeeze(-1)
    cross_u_aref = (cross_matrix @ u_aref[:, :, None]).squeeze(-1)

    A_list, B_list, Y_aref_list, Y_am_list, W_list = [], [], [], [], []
    win_omega_norms, win_alpha_norms, win_lever_norms = [], [], []
    
    for i in range(0, len(t_valid) - win_size, step_size):
        j = i + win_size
        dt_arr = np.diff(t_valid[i:j+1])
        
        int_cross_omega = np.sum(0.5 * (cross_matrix[i:j] + cross_matrix[i+1:j+1]) * dt_arr[:, None, None], axis=0)
        int_cross_omega_L = np.sum(0.5 * (cross_L[i:j] + cross_L[i+1:j+1]) * dt_arr[:, None, None], axis=0)
        
        int_cross_omega_u_am = np.sum(0.5 * (cross_u_am[i:j] + cross_u_am[i+1:j+1]) * dt_arr[:, None], axis=0)
        int_cross_omega_u_aref = np.sum(0.5 * (cross_u_aref[i:j] + cross_u_aref[i+1:j+1]) * dt_arr[:, None], axis=0)
        
        A_win = (lever[j] - lever[i]) + int_cross_omega_L
        B_win = int_cross_omega
        Y_aref_win = (u_aref[j] - u_aref[i]) + int_cross_omega_u_aref
        Y_am_win = (u_am[j] - u_am[i]) + int_cross_omega_u_am
        
        w_omega = np.mean(np.linalg.norm(angular[i:j], axis=1))
        w_alpha = np.mean(np.linalg.norm(angular_rate[i:j], axis=1))
        w_lever = np.mean(np.linalg.norm(lever[i:j].reshape(j-i, -1), axis=1))
        
        W_list.append(w_omega)
        win_omega_norms.append(w_omega)
        win_alpha_norms.append(w_alpha)
        win_lever_norms.append(w_lever)
        
        A_list.append(A_win)
        B_list.append(B_win)
        Y_aref_list.append(Y_aref_win)
        Y_am_list.append(Y_am_win)

    A_global = np.vstack(A_list)
    B_global = np.vstack(B_list)
    Y_aref_global = np.concatenate(Y_aref_list)
    Y_am_global = np.concatenate(Y_am_list)
    W_global = np.array(W_list)

    w_omega_p95 = max(np.percentile(win_omega_norms, 95), 1e-6)
    w_weights = np.clip(W_global / w_omega_p95, 0.1, 1.0) ** 2
    w_rep = np.repeat(w_weights, 3)
    w_sqrt = np.sqrt(w_rep)

    # 新的设计矩阵：[A, B, Y_aref]，未知数 [p, ba, s]
    A_full = np.hstack([A_global, B_global, Y_aref_global.reshape(-1, 1)])
    A_w = A_full * w_sqrt[:, None]
    Y_w = Y_am_global * w_sqrt

    print(f"  [C] 预积分堆叠完成: 窗口数 {len(A_list)} (长度 {win_size} 帧, 步长 {step_size} 帧)")

    # ========== Step D: 线性初值 (直接解 7 个未知数) ==========
    if options.get("skip_linear_init", False):
        singular = np.linalg.svd(A_w, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-8))
        initial_lin = np.r_[np.zeros(6), 1.0]
    else:
        initial_lin, _, rank, singular = np.linalg.lstsq(A_w, Y_w, rcond=1e-8)
    if rank < 7:
        print(f"  [D] [FAIL] rank={rank} < 7，数据激励不足")
        return {"status": "unobservable", "design_rank": int(rank), "reason": "rank < 7"}

    p_lin, ba_lin, s_lin = initial_lin[:3], initial_lin[3:6], initial_lin[6]
    # 给 s 强加一个合理的初始范围，防止线性解偶然偏到了极端值
    s_lin = np.clip(s_lin, 1.0, 5.0)
    initial_lin[6] = s_lin

    print(f"  [D] 线性初值: p={p_lin}, b_a={ba_lin}, scale={s_lin:.3f}")

    # ========== Step E: 非线性优化 ==========
    cad_prior_cfg = options.get("cad_translation_prior") or {}
    use_cad_prior = bool(cad_prior_cfg.get("enabled", False))
    if use_cad_prior:
        prior_p = np.asarray(cad_prior_cfg.get("p_m", [0.0, 0.0, 0.0]), dtype=float)
        prior_sigma = np.asarray(cad_prior_cfg.get("sigma_m", [0.05, 0.05, 0.05]), dtype=float)
        print(f"  [E] CAD 先验启用: p={prior_p}, sigma={prior_sigma}")
    else:
        prior_p = np.zeros(3)
        prior_sigma = np.ones(3)

    bias_sigma = float(options.get("bias_sigma", 0.1))
    bias_bound = float(options.get("bias_bound", 1.0))
    scale_bound = float(options.get("scale_bound", 5.0))

    accel_best_cost = float("inf")
    accel_best_params = []
    accel_best_costs = []

    def residual(params):
        nonlocal accel_best_cost
        p_est = params[:3]
        ba_est = params[3:6]
        s_est = params[6]
        
        r_main = A_w @ params - Y_w
        r_ba = ba_est / bias_sigma
        # 针对 s 的正则项 (鼓励 s 接近 1.0，防止发散)
        r_s = (s_est - 1.0) / 1.0
        
        if use_cad_prior:
            r_prior = (p_est - prior_p) / prior_sigma
            full = np.r_[r_main, r_ba, r_s, r_prior]
        else:
            full = np.r_[r_main, r_ba, r_s]
        z = (full / float(options.get("f_scale", 0.1))) ** 2
        f_scale = float(options.get("f_scale", 0.1))
        cost = 0.5 * f_scale ** 2 * float(np.sum(2 * (np.sqrt(1 + z) - 1)))
        if cost < accel_best_cost - 1e-10:
            accel_best_cost = cost
            accel_best_params.append(params.copy())
            accel_best_costs.append(cost)
        return full

    lower_bounds = np.r_[-np.inf * np.ones(3), -bias_bound * np.ones(3), 0.5]
    upper_bounds = np.r_[ np.inf * np.ones(3),  bias_bound * np.ones(3), scale_bound]

    # 确保初值严格在边界内部 (scipy least_squares 要求)
    lower_safe = np.where(np.isfinite(lower_bounds), lower_bounds + 1e-4, -np.inf)
    upper_safe = np.where(np.isfinite(upper_bounds), upper_bounds - 1e-4, np.inf)
    initial_lin = np.fmax(initial_lin, lower_safe)
    initial_lin = np.fmin(initial_lin, upper_safe)

    result = least_squares(
        residual,
        initial_lin,
        bounds=(lower_bounds, upper_bounds),
        loss=str(options.get("loss", "soft_l1")),
        f_scale=float(options.get("f_scale", 0.1)),
        max_nfev=int(options.get("max_nfev", 300)),
        diff_step=1e-4,
    )

    if debug_store is not None:
        debug_store["accel_best_params"] = np.asarray(accel_best_params)
        debug_store["accel_best_costs"] = np.asarray(accel_best_costs)
        debug_store["accel_linear_initial"] = initial_lin.copy()
        debug_store["accel_design_weighted"] = A_w.copy()
        debug_store["accel_target_weighted"] = Y_w.copy()

    # ========== Step F: 物理残差与统计 ==========
    errors = A_full @ result.x - Y_am_global
    errors_3d = errors.reshape(-1, 3)
    variance = float(errors @ errors / max(len(errors) - 7, 1))
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * variance

    p_est = result.x[0:3]
    ba_est = result.x[3:6]
    s_est = result.x[6]

    rms_xyz = np.sqrt(np.mean(errors_3d ** 2, axis=0))
    bias_xyz = errors_3d.mean(axis=0)

    print(f"  [F] 非线性优化完成:")
    print(f"      success={result.success}, nfev={result.nfev}")
    print(f"      p_final = {p_est}")
    print(f"      b_a_final = {ba_est}")
    print(f"      s_final = {s_est:.4f}  (参考加速度被放大的倍数)")
    print(f"      RMS_total = {np.sqrt(np.mean(errors ** 2)):.4f} m/s²")
    print(f"      RMS_xyz   = x:{rms_xyz[0]:.4f}  y:{rms_xyz[1]:.4f}  z:{rms_xyz[2]:.4f}")

    # ========== Step G: 反推重力项 ==========
    u_corrected = u_am - s_est * u_aref
    g_series = u_corrected - np.einsum('nij,j->ni', lever, p_est) - ba_est
    g_raw = np.median(g_series, axis=0)
    
    gravity_norm = float(options.get("gravity_norm_m_s2", 9.80665))
    g_dir = g_raw / max(np.linalg.norm(g_raw), 1e-9)
    g_est = g_dir * gravity_norm
    
    print(f"  [G] 物理一致性验证 (反推重力):")
    print(f"      全序列反推重力中位数 = {g_raw}  |g|={np.linalg.norm(g_raw):.4f}")
    print(f"      投影至 9.81 后 = {g_est}")

    std_p = np.sqrt(np.maximum(np.diag(covariance)[:3], 0))
    std_ba = np.sqrt(np.maximum(np.diag(covariance)[3:6], 0))
    snr_p = np.linalg.norm(p_est) / max(np.linalg.norm(std_p), 1e-9)
    snr_ba = np.linalg.norm(ba_est) / max(np.linalg.norm(std_ba), 1e-9)

    fake_solution_9d = np.r_[p_est, ba_est, g_est]
    fake_linear_9d = np.r_[p_lin, ba_lin, g_est]

    # ========== Step H: 局部重力补偿 (用于可视化) ==========
    # 逐点计算瞬时重力 (基于优化后的 p, ba, s)
    g_pointwise = u_am - np.einsum('nij,j->ni', lever, p_est) - ba_est - (u_aref * s_est)
    
    # 对每个滑动窗口内的瞬时重力求平均，得到该窗口的局部常数重力
    local_g = np.zeros_like(g_pointwise)
    counts = np.zeros(len(u_am))
    
    for i in range(0, len(t_valid) - win_size, step_size):
        j = i + win_size
        g_win_mean = np.mean(g_pointwise[i:j], axis=0)
        local_g[i:j] += g_win_mean[None, :]
        counts[i:j] += 1
        
    valid_counts = counts > 0
    local_g[valid_counts] /= counts[valid_counts][:, None]
    
    # 处理边缘没有被窗口覆盖的点（用最近的有效值填充）
    if not np.all(valid_counts):
        idx = np.where(valid_counts)[0]
        if len(idx) > 0:
            first, last = idx[0], idx[-1]
            local_g[:first] = local_g[first]
            local_g[last:] = local_g[last]
            for k in range(first, last):
                if not valid_counts[k]:
                    local_g[k] = local_g[k-1]
                    
    # 使用局部重力进行补偿
    accel_compensated = u_am - np.einsum('nij,j->ni', lever, p_est) - ba_est - local_g

    if debug_store is not None:
        t_centers = []
        for i in range(0, len(t_valid) - win_size, step_size):
            t_centers.append(t_valid[i + win_size // 2])
        t_centers = np.array(t_centers)
            
        debug_store.update({
            "t_valid": t_centers,
            "angular_norm": np.array(win_omega_norms),
            "alpha_norm": np.array(win_alpha_norms),
            "lever_norm": np.array(win_lever_norms),
            "target_norm": np.linalg.norm(Y_am_global.reshape(-1, 3), axis=1),
            "weights": w_weights,
            "w_time": np.ones_like(W_global),
            "w_omega": w_weights,
            "design": A_full,
            "target": Y_am_global,
            "linear_solution": fake_linear_9d,
            "final_solution": fake_solution_9d,
            "residual_3d": errors_3d.copy(),
            "rms_xyz": rms_xyz,
            "bias_xyz": bias_xyz,
            "g_linear_norm": gravity_norm,
            "g_deviation": 0.0,
            "rank": int(rank),
            "singular": singular,
            "condition": float(singular[0] / singular[-1]),
            "snr_p": float(snr_p),
            "snr_ba": float(snr_ba),
            "scale_factor": float(s_est),
            # 新增可视化用的字段：
            "t_imu": t_valid,
            "accel_compensated": accel_compensated,
            "u_aref": u_aref,
        })

    print("-" * 70 + "\n")

    return {
        "status": "estimated" if result.success else "not_converged",
        "success": bool(result.success),
        "message": str(result.message),
        "p_body_imu_m": p_est.tolist(),
        "accel_bias_imu_m_s2": ba_est.tolist(),
        "gravity_initial_body_m_s2": g_est.tolist(),
        "accel_scale_factor": float(s_est),
        "diagnostics": {
            "model_type": "relative_preintegration_linear_with_scale",
            "window_size_samples": win_size,
            "num_windows": len(A_list),
            "used_cad_prior": use_cad_prior,
        },
        "design_rank": int(rank),
        "design_condition": float(singular[0] / singular[-1]),
        "singular_values": singular.tolist(),
        "num_residuals": int(len(errors)),
        "rms_m_s2": float(np.sqrt(np.mean(errors ** 2))),
        "rms_xyz_m_s2": rms_xyz.tolist(),
        "residual_bias_xyz_m_s2": bias_xyz.tolist(),
        "covariance": covariance.tolist(),
        "translation_std_m": std_p.tolist(),
        "bias_std_m_s2": std_ba.tolist(),
        "gravity_std_m_s2": [0.05, 0.05, 0.05],
        "snr_p": float(snr_p),
        "snr_ba": float(snr_ba),
        "model": "Pre-integrated relative motion with scale factor",
        "attitude_source": "No global integration, gravity canceled out in windows",
        "rotation_time": "fixed at the gyroscope-stage estimate",
    }


def _write_imu_summary(path: Path, payload: Dict) -> None:
    lines = [
        "# IMU Calibration Stage",
        "",
        "## Time initialization",
        f"- coarse offset adjust (s): {payload['time_initialization']['coarse_offset_adjust_sec']}",
        f"- cross-correlation offset (s): {payload['time_initialization']['cross_correlation_offset_sec']}",
        f"- initial time offset (s): {payload['time_initialization']['refined_offset_sec']}",
        f"- final time offset (s): {payload['time_offset_imu_minus_body_sec']}",
        f"- max correlation: {payload['time_initialization']['max_cross_correlation']}",
        "",
        "## Optimization",
        f"- initial RMS: {payload.get('initial_rms')}",
        f"- final RMS: {payload['rms']}",
        f"- Euler xyz body<-imu (deg): {payload['euler_xyz_deg_body_imu']}",
        f"- gyro bias (rad/s): {payload['gyro_bias_imu_rad_s']}",
        f"- translation calibration: {payload.get('translation_calibration', {}).get('status', 'not estimated')}",
        f"- IMU origin in body (m): {payload['p_body_imu_m']}",
        f"- accelerometer bias (m/s^2): {payload['accel_bias_imu_m_s2']}",
        "",
        "## Key figures",
        "- imu_time_alignment.png",
        "- imu_initial_gyro_residual_hist.png",
        "- imu_gyro_residual_hist.png",
        "- debug/accel_01_residual_time.png",
        "- debug/accel_02_residual_vs_omega.png",
        "- debug/accel_03_residual_vs_alpha.png",
        "- debug/accel_04_residual_vs_weight.png",
        "- debug/accel_05_param_compare.png",
        "- debug/accel_06_segment_rms.png",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# === 陀螺仪阶段详细诊断函数 ===
# ============================================================
def _debug_gyro_stage(
    t_ref, w_ref,
    ti, gi,
    x0, result,
    initial_residual, residual, pred,
    rvec0, tau0, bg0,
    tau_window,
    iteration_history,
    out_dir,
    cad_rotvec=None,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    param_names = ["rvec_x", "rvec_y", "rvec_z", "tau", "bg_x", "bg_y", "bg_z"]

    print("\n" + "=" * 78)
    print(">>> Gyro Stage Detailed Diagnostics <<<")
    print("=" * 78)

    print("\n[1. Parameter Changes]")
    print(f"  {'Param':<10} {'Init':>14} {'Final':>14} {'Delta':>14}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14}")
    for i, name in enumerate(param_names):
        delta = result.x[i] - x0[i]
        print(f"  {name:<10} {x0[i]:>+14.6f} {result.x[i]:>+14.6f} {delta:>+14.6f}")

    r_rel = Rot.from_rotvec(x0[:3]).inv() * Rot.from_rotvec(result.x[:3])
    delta_angle = np.linalg.norm(r_rel.as_rotvec()) * 180 / np.pi
    print(f"\n  Rotation Correction = {delta_angle:.4f} deg")

    print(f"  tau init = {tau0:.6f}s, final = {result.x[3]:.6f}s, opt delta = {(result.x[3]-tau0)*1e3:+.4f} ms")
    if abs(result.x[3] - tau0) < 1e-8:
        print("  [WARN] tau didn't move (zero gradient)")
    if abs(abs(result.x[3] - tau0) - tau_window) < 1e-4:
        print("  [WARN] tau hit the boundary!")

    if iteration_history and len(iteration_history) > 1:
        print(f"\n[2. Iteration History] total {len(iteration_history)} steps")
        print(f"  {'Step':>4} {'cost':>16} {'RMS':>13} {'step':>12} {'tau':>12}")
        for rec in iteration_history[:60]:
            x = rec["x"]
            print(f"  {rec['call']:>4} {rec['cost']:>16.6e} {rec['rms']:>13.6e} {rec['step']:>12.3e} {x[3]:>+12.6f}")
        if len(iteration_history) > 60:
            print(f"  ... omitted {len(iteration_history) - 60} steps")

    print("\n[3. Residual Analysis]")
    resid_3d = residual.reshape(-1, 3)
    rms_xyz = np.sqrt(np.mean(resid_3d ** 2, axis=0))
    rms_initial = np.sqrt(np.mean(initial_residual ** 2))
    rms_final = np.sqrt(np.mean(residual ** 2))
    print(f"  Axis RMS:  x={rms_xyz[0]:.6f}  y={rms_xyz[1]:.6f}  z={rms_xyz[2]:.6f}  rad/s")
    print(f"  Total RMS: {rms_final:.6f} rad/s (Initial: {rms_initial:.6f}, Reduct: {(1 - rms_final / max(rms_initial, 1e-9)) * 100:.2f}%)")

    print("\n[4. Observability (Jacobian SVD)]")
    try:
        U, S, Vt = np.linalg.svd(result.jac, full_matrices=False)
        cond = S[0] / max(S[-1], 1e-12)
        effective_rank = int(np.sum(S > 1e-6 * S[0]))
        print(f"  Condition number: {cond:.2e}")
        print(f"  Effective rank:   {effective_rank} / 7")
    except Exception as e:
        print(f"  SVD failed: {e}")

    print("\n[5. Generating Diagnostic Plots]")

    if iteration_history and len(iteration_history) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        hist_x = [r["call"] for r in iteration_history]
        hist_cost = [r["cost"] for r in iteration_history]
        hist_rms = [r["rms"] for r in iteration_history]
        axes[0].semilogy(hist_x, hist_cost, 'b-o', markersize=3, linewidth=1.5)
        axes[0].set_ylabel("Cost")
        axes[0].set_title("Convergence: Cost vs Iteration")
        axes[0].grid(True, alpha=0.3)
        axes[1].semilogy(hist_x, hist_rms, 'r-o', markersize=3, linewidth=1.5)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("RMS (rad/s)")
        axes[1].set_title("Convergence: RMS vs Iteration")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "gyro_convergence.png", dpi=150)
        plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(ti, resid_3d[:, i], 'purple', alpha=0.6, linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.set_ylabel(f"Resid w_{name} (rad/s)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Gyro Residuals over Time")
    plt.tight_layout()
    plt.savefig(out_dir / "gyro_residuals_time.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    for i, (ax, name) in enumerate(zip(axes[:3], ["X", "Y", "Z"])):
        ax.hist(resid_3d[:, i], bins=80, alpha=0.75, color=colors[i])
        ax.set_xlabel(f"Resid w_{name} (rad/s)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} (sigma={np.std(resid_3d[:, i]):.4f})")
        ax.grid(True, alpha=0.3)
    ax = axes[3]
    ax.hist(np.linalg.norm(resid_3d, axis=1), bins=80, alpha=0.75, color='purple')
    ax.set_xlabel("||resid|| (rad/s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Magnitude (sigma={np.std(np.linalg.norm(resid_3d, axis=1)):.4f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "gyro_residuals_hist.png", dpi=150)
    plt.close()

    print("  [OK] Plots saved.")
    print("=" * 78 + "\n")


# ============================================================
# === 加速度计标定阶段详细诊断函数 ===
# ============================================================
def _debug_accel_translation_stage(debug_store: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    print("\n" + "=" * 78)
    print(">>> Accelerometer Stage - Residual Allocation Diagnostics <<<")
    print("=" * 78)

    if not debug_store:
        print("  [skip] debug_store is empty.")
        return

    t = debug_store["t_valid"]
    t_rel = t - t[0]
    res3 = debug_store["residual_3d"]
    res_mag = np.linalg.norm(res3, axis=1)
    omega_norm = debug_store["angular_norm"]
    alpha_norm = debug_store["alpha_norm"]
    lever_norm = debug_store["lever_norm"]
    weights = debug_store["weights"]

    ensure_dir(out_dir / "debug")
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    # 图 1: 逐轴残差时间序列
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
        ax.plot(t_rel, res3[:, i], color=colors[i], lw=0.7, alpha=0.8)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_ylabel(f"res_{name} (m/s^2)")
        ax.set_title(f"{name}: RMS={debug_store['rms_xyz'][i]:.4f}, bias={debug_store['bias_xyz'][i]:+.4f}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t - t0 (s)")
    fig.suptitle("Accel residual time series", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "debug" / "accel_01_residual_time.png", dpi=150)
    plt.close(fig)

    # 图 5: 线性解 vs 非线性解
    lin = debug_store["linear_solution"]
    fin = debug_store["final_solution"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    param_sets = [
        ("p (m)", lin[0:3], fin[0:3], 3),
        ("b_a (m/s^2)", lin[3:6], fin[3:6], 3),
        ("g_B0 (m/s^2)", lin[6:9], fin[6:9], 3),
    ]
    for ax, (title, lin_v, fin_v, n) in zip(axes, param_sets):
        x = np.arange(n)
        ax.bar(x - 0.18, lin_v, width=0.35, label="linear", color="C0", alpha=0.75)
        ax.bar(x + 0.18, fin_v, width=0.35, label="final", color="C3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["x", "y", "z"])
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Linear vs nonlinear parameters (|g|_lin = {debug_store['g_linear_norm']:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "debug" / "accel_05_param_compare.png", dpi=150)
    plt.close(fig)

    print(f"  保存目录: {out_dir / 'debug'}")
    print("=" * 78 + "\n")


def _debug_imu_pipeline(
    t_ref, v_ref, w_ref,
    t_imu, accel, gyro,
    ti, gi,
    tau0, off_cc, tau_final, corr, lags,
    rvec0, rvec_final, bg0, bg_final,
    initial_residual, residual, pred,
    translation_payload,
    ref_acc, v_ref_for_accel,
    stage_cfg,
):
    print("\n" + "=" * 78)
    print(">>> IMU Calibration Pipeline Layered Error Analysis <<<")
    print("=" * 78)

    print("\n[1. Time Alignment]")
    print(f"  Cross-correlation peak off_cc = {off_cc:+.6f} s")
    print(f"  Refined tau0                  = {tau0:+.6f} s")
    print(f"  Optimized tau_final           = {tau_final:+.6f} s")
    print(f"  IMU rate: {1.0 / np.median(np.diff(t_imu)):.1f} Hz")
    print(f"  GT rate:  {1.0 / np.median(np.diff(t_ref)):.1f} Hz")

    print("\n[2. Gyro Extrinsics]")
    print(f"  Kabsch init rvec0     = {rvec0}")
    print(f"  Optimized rvec_final  = {rvec_final}")
    
    print("\n[3. Translation Calibration]")
    tp = translation_payload or {}
    if tp.get("status") == "estimated":
        print(f"  Design Matrix Rank    = {tp['design_rank']} / 7")
        print(f"  Lever Arm Estimate    = {tp['p_body_imu_m']} m")
        print(f"  Accel Bias            = {tp['accel_bias_imu_m_s2']} m/s^2")
        print(f"  Accel Scale Factor    = {tp.get('accel_scale_factor', 1.0):.4f}")
        print(f"  Accel Residual RMS    = {tp['rms_m_s2']:.6f} m/s^2")
    else:
        print(f"  Status: {tp.get('status', 'N/A')}")

    print("\n" + "=" * 78 + "\n")


def run_imu_reference_stage(stage_cfg: Dict, reference_cfg: Dict, output_root: Path, config_dir: Path) -> ImuStageResult:
    name = stage_cfg.get("name", "imu")
    out = output_root / name
    ensure_dir(out)

    ref_path = Path(reference_cfg["twist"]).expanduser()
    inp = stage_cfg.get("inputs") or {}
    imu_path = Path(inp["imu"]).expanduser()
    if not ref_path.is_absolute():
        ref_path = (config_dir / ref_path).resolve()
    if not imu_path.is_absolute():
        imu_path = (config_dir / imu_path).resolve()

    io_cfg = stage_cfg.get("io") or {}
    t_ref, v_ref, w_ref = load_reference_twist(ref_path, delimiter=io_cfg.get("reference_delimiter"))
    t_imu, accel, gyro = load_imu(
        imu_path,
        gyro_cols=inp.get("gyro_cols", "auto"),
        accel_cols=inp.get("accel_cols"),
        delimiter=io_cfg.get("imu_delimiter"),
    )
    preprocessing_cfg = stage_cfg.get("preprocessing") or {}
    trim_edges_sec = float(preprocessing_cfg.get("trim_edges_sec", 0.0))
    edge_mask = edge_trim_mask(t_imu, trim_edges_sec)
    t_imu, gyro = t_imu[edge_mask], gyro[edge_mask]
    if accel is not None:
        accel = accel[edge_mask]

    time_cfg = stage_cfg.get("time") or {}
    ablation = stage_cfg.get("ablation") or {}
    max_lag = float(time_cfg.get("max_lag_sec", 0.5))
    search_range = float(time_cfg.get("search_range_sec", 0.03))
    coarse_start = bool(time_cfg.get("coarse_start", False))
    coarse_adjust = float(time_cfg.get("coarse_offset_adjust_sec", 0.0))
    ref_sig = np.linalg.norm(w_ref, axis=1)
    imu_sig = np.linalg.norm(gyro, axis=1)
    raw_offset = (float(t_imu[0] - t_ref[0]) if coarse_start else 0.0) + coarse_adjust
    t_imu_coarse = t_imu - raw_offset
    if ablation.get("skip_time_init", False):
        off_cc, corr, lags = 0.0, np.zeros(1), np.zeros(1)
        tau0 = raw_offset
    else:
        off_cc, corr, lags = estimate_offset_cc(t_ref, ref_sig, t_imu_coarse, imu_sig, max_lag_sec=max_lag)
        tau0 = raw_offset + _refine_offset(t_ref, ref_sig, t_imu_coarse, imu_sig, off_cc, search_range)

    if ablation.get("skip_rotation_init", False):
        r_bi0, bg0, c_body = np.eye(3), np.zeros(3), np.zeros(3)
    else:
        c_body, valid = _initial_body_bias(t_imu, gyro, t_ref, w_ref, tau0)
        w0 = interp_vectors(t_imu[valid] - tau0, t_ref, w_ref)
        r_bi0 = kabsch_body_from_sensor(gyro[valid], w0)
        bg0 = r_bi0.T @ c_body
    init_cfg = stage_cfg.get("initialization") or {}
    if ablation.get("skip_rotation_init", False):
        rvec0 = np.zeros(3)
    elif "rotvec_body_imu" in init_cfg:
        rvec0 = np.asarray(init_cfg["rotvec_body_imu"], dtype=float)
        r_bi0 = Rot.from_rotvec(rvec0).as_matrix()
    else:
        rvec0 = Rot.from_matrix(r_bi0).as_rotvec()
    if "gyro_bias" in init_cfg:
        bg0 = np.asarray(init_cfg["gyro_bias"], dtype=float)
    x0 = np.r_[rvec0, tau0, bg0]

    opt_cfg = stage_cfg.get("optimization") or {}
    tau_window = float(opt_cfg.get("tau_refine_window_sec", search_range))
    rot_bound = float(opt_cfg.get("rotvec_bound_rad", np.pi))
    bias_bound = float(opt_cfg.get("gyro_bias_bound_rad_s", 2.0))
    lower = np.array([-rot_bound, -rot_bound, -rot_bound, tau0 - tau_window, -bias_bound, -bias_bound, -bias_bound])
    upper = np.array([rot_bound, rot_bound, rot_bound, tau0 + tau_window, bias_bound, bias_bound, bias_bound])
    common_radius = float(ablation.get("common_time_radius_sec", 0.0))
    if common_radius > 0.0:
        lower[3], upper[3] = raw_offset - common_radius, raw_offset + common_radius

    overlap_center = raw_offset if common_radius > 0.0 else tau0
    overlap_radius = common_radius if common_radius > 0.0 else tau_window
    overlap = (t_imu - overlap_center >= t_ref[0] + overlap_radius + 0.01) & (t_imu - overlap_center <= t_ref[-1] - overlap_radius - 0.01)
    ti = t_imu[overlap]
    gi = gyro[overlap]

    iteration_history = []
    _last_x = [None]
    _call_count = [0]

    def _tracked_residual(x, *args):
        _call_count[0] += 1
        r = imu_gyro_residual(x, *args)
        cost = float(0.5 * np.sum(r ** 2))
        rms = float(np.sqrt(np.mean(r ** 2)))
        if _last_x[0] is None or np.linalg.norm(x - _last_x[0]) > 1e-14:
            step = 0.0 if _last_x[0] is None else float(np.linalg.norm(x - _last_x[0]))
            iteration_history.append({
                "call": _call_count[0],
                "x": x.copy(),
                "cost": cost,
                "rms": rms,
                "step": step,
            })
            _last_x[0] = x.copy()
        return r

    result = least_squares(
        _tracked_residual,
        x0,
        args=(ti, gi, t_ref, w_ref),
        method="trf",
        loss=opt_cfg.get("loss", "huber"),
        f_scale=float(opt_cfg.get("f_scale", 0.02)),
        max_nfev=int(opt_cfg.get("max_nfev", 300)),
        bounds=(lower, upper),
        diff_step=1e-3,
    )

    gyro_best_params = []
    gyro_best_costs = []
    best_cost = float("inf")
    for record in iteration_history:
        if record["cost"] < best_cost - 1e-10:
            best_cost = record["cost"]
            gyro_best_params.append(record["x"])
            gyro_best_costs.append(best_cost)

    initial_residual = imu_gyro_residual(x0, ti, gi, t_ref, w_ref)
    residual = imu_gyro_residual(result.x, ti, gi, t_ref, w_ref)
    pred, pred_valid = imu_gyro_prediction(result.x, ti, t_ref, w_ref)
    r_bi_final = Rot.from_rotvec(result.x[:3]).as_matrix()
    bg_final = result.x[4:7]
    gyro_body = (r_bi_final @ (gi - bg_final).T).T
    ref_w_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, w_ref)
    accel_ref_cfg = preprocessing_cfg.get("reference_accel_lowpass") or {}
    v_ref_for_accel, ref_accel_filter_info = smooth_vector_savgol(
        t_ref,
        v_ref,
        enabled=bool(accel_ref_cfg.get("enabled", True)),
        window_sec=float(accel_ref_cfg.get("window_sec", 0.05)),
        polyorder=int(accel_ref_cfg.get("polyorder", 2)),
        edge_mode=str(accel_ref_cfg.get("edge_mode", "interp")),
    )
    ref_acc = _numerical_derivative(t_ref, v_ref_for_accel)

    payload = {
        "stage": name,
        "type": "imu_reference",
        "frame_convention": {
            "rotation": "R_body_imu maps IMU-frame vectors to body frame",
            "time": "t_body = t_imu - tau",
            "residual": "gyro_imu - bias - R_imu_body * omega_body",
            "translation": "accelerometer lever-arm refinement is reserved for the acceleration-enabled stage",
        },
        "inputs": {
            "reference_twist": str(ref_path),
            "imu": str(imu_path),
            "gyro_cols": inp.get("gyro_cols", "auto"),
            "accel_cols": inp.get("accel_cols"),
        },
        "time_initialization": {
            "raw_start_difference_sec": raw_offset,
            "coarse_offset_adjust_sec": coarse_adjust,
            "cross_correlation_offset_sec": off_cc,
            "refined_offset_sec": tau0,
            "max_cross_correlation": float(np.max(corr)),
        },
        "initial_guess": {
            "rotvec_body_imu": rvec0.tolist(),
            "gyro_bias_imu_rad_s": bg0.tolist(),
            "time_offset_sec": tau0,
            "body_bias_proxy": c_body.tolist(),
        },
        "preprocessing": {
            "trim_edges_sec": trim_edges_sec,
            "num_samples_after_edge_trim": int(len(t_imu)),
            "reference_accel_lowpass": ref_accel_filter_info,
        },
        "accel_available": accel is not None,
    }
    payload.update(_summarize(result, residual))

    translation_options = dict(stage_cfg.get("accelerometer_calibration") or {})
    translation_options["skip_linear_init"] = bool(ablation.get("skip_translation_init", False))
    if translation_options.get("enabled", False):
        if accel is None:
            payload["translation_calibration"] = {"status": "unavailable", "reason": "IMU acceleration is required"}
        else:
            smooth_omega, _ = smooth_vector_savgol(t_ref, w_ref, enabled=True,
                                                  window_sec=float(translation_options.get("angular_window_sec", 0.35)),
                                                  polyorder=2, edge_mode="interp")

            accel_debug_store = {}
            translation = estimate_imu_translation(
                ti, accel[overlap], r_bi_final, float(result.x[3]),
                t_ref, v_ref_for_accel, smooth_omega,
                translation_options,
                debug_store=accel_debug_store,
            )
            payload["translation_calibration"] = translation
            if translation["status"] == "estimated":
                np.savez_compressed(
                    out / "imu_process_trace.npz",
                    gyro_params=np.asarray(gyro_best_params),
                    gyro_cost=np.asarray(gyro_best_costs),
                    accel_params=accel_debug_store["accel_best_params"],
                    accel_cost=accel_debug_store["accel_best_costs"],
                    accel_initial=accel_debug_store["accel_linear_initial"],
                    accel_design=accel_debug_store["accel_design_weighted"],
                    accel_target=accel_debug_store["accel_target_weighted"],
                )
                payload["p_body_imu_m"] = translation["p_body_imu_m"]
                payload["accel_bias_imu_m_s2"] = translation["accel_bias_imu_m_s2"]
                payload["frame_convention"]["translation"] = "p_body_imu is the IMU origin expressed in body frame; estimated from acceleration"

            try:
                _debug_accel_translation_stage(accel_debug_store, out)
            except Exception as _dbg3_e:
                print(f"[DEBUG] Accelerometer stage diagnostics exception: {_dbg3_e}")

    payload["T_body_imu"] = make_se3(payload["R_body_imu"], payload["p_body_imu_m"])
    payload["initial_rms"] = float(np.sqrt(np.mean(initial_residual * initial_residual)))

    result_json = out / "imu_result.json"
    data_npz = out / "imu_stage_data.npz"
    save_json(result_json, payload)
    _write_imu_summary(out / "stage_summary.md", payload)
    np.savez_compressed(
        data_npz,
        t_imu=ti,
        gyro=gi,
        gyro_predicted=pred,
        residual=residual.reshape(-1, 3),
        initial_residual=initial_residual.reshape(-1, 3),
        params=result.x,
        initial_params=x0,
        valid=pred_valid,
        accel=accel[overlap] if accel is not None else np.empty((0, 3)),
        accel_reference_velocity=v_ref_for_accel,
        accel_reference_omega=smooth_omega if translation_options.get("enabled", False) and accel is not None else w_ref,
        accel_window_sec=float(translation_options.get("window_sec", 0.1)),
        accel_step_sec=float(translation_options.get("step_sec", 0.05)),
        accel_bias_sigma=float(translation_options.get("bias_sigma", 0.1)),
    )
    np.savetxt(out / "imu_gyro_residuals.csv", residual.reshape(-1, 3), delimiter=",", header="rx,ry,rz", comments="")
    np.savetxt(
        out / "imu_step_metrics.csv",
        np.array([
            ["coarse_offset_adjust_sec", coarse_adjust],
            ["cross_correlation_offset_sec", off_cc],
            ["initial_time_offset_sec", tau0],
            ["final_time_offset_sec", float(result.x[3])],
            ["initial_rms", payload["initial_rms"]],
            ["final_rms", payload["rms"]],
        ], dtype=object),
        delimiter=",",
        fmt="%s",
    )

    if bool((stage_cfg.get("diagnostics") or {}).get("save_plots", True)):
        plot_cross_correlation(out / "imu_cross_correlation.png", lags, corr, off_cc, "IMU/reference angular-speed cross-correlation")
        plot_alignment(out / "imu_time_alignment.png", t_ref, ref_sig, t_imu, imu_sig, float(result.x[3]), "IMU-reference angular-speed alignment", "angular speed (rad/s)")
        plot_vector_components(
            out / "imu_gyro_components_sensor_frame.png",
            ti,
            [
                ("gyro measured", gi, "0.45"),
                ("gyro predicted", pred, "C3"),
            ],
            ["x", "y", "z"],
            "IMU gyroscope component fit",
            "rad/s",
        )
        plot_vector_components(
            out / "imu_gyro_body_alignment_effect.png",
            ti,
            [
                ("gyro rotated to body", gyro_body, "C0"),
                ("reference angular velocity", ref_w_on_imu, "C3"),
            ],
            ["x", "y", "z"],
            "IMU rotation calibration effect",
            "rad/s",
        )
        plot_vector_components(
            out / "reference_twist_components.png",
            t_ref,
            [
                ("reference linear velocity", v_ref, "C0"),
                ("reference angular velocity", w_ref, "C3"),
            ],
            ["x", "y", "z"],
            "Reference body twist components",
            "m/s or rad/s",
        )

        # if accel is not None:
        #     accel_i = accel[overlap]
        #     accel_body = (r_bi_final @ accel_i.T).T
        #     ref_acc_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, ref_acc)
        #     ref_vel_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, v_ref_for_accel)
        #     ref_w_on_acc = interp_vectors(ti - float(result.x[3]), t_ref, w_ref)
        #     ref_body_acc = ref_acc_on_imu + np.cross(ref_w_on_acc, ref_vel_on_imu)
            
        #     # 注意：这里不再减去 np.nanmedian
        #     ref_acc_dyn = ref_body_acc
            
        #     accel_body_gravity_comp_dyn = None
        #     translation = payload.get("translation_calibration") or {}
        #     if translation.get("status") == "estimated":
        #         # 直接用估算出来的重力进行抵消，得到纯动态观测
        #         g_const = np.array(translation["gravity_initial_body_m_s2"])
        #         # 注意这里的符号：通常 IMU 测的是 a - g，所以加回来。具体看你的传感器定义。
        #         # 如果之前 accel_body + g_const 结果是对的，我们就沿用
        #         accel_body_gravity_comp_dyn = accel_body + g_const
            
        #     ref_acc_lsq_scale = _finite_flat_scale(ref_acc_dyn, accel_body_dyn if 'accel_body_dyn' in locals() else accel_body)
            
        #     if translation.get("status") == "estimated" and "accel_scale_factor" in translation:
        #         ref_acc_scale = translation["accel_scale_factor"]
        #     else:
        #         scale_target = accel_body_gravity_comp_dyn if accel_body_gravity_comp_dyn is not None else accel_body
        #         ref_acc_scale = _finite_rms_scale(ref_acc_dyn, scale_target)
            
        #     ref_acc_dyn_scaled = ref_acc_scale * ref_acc_dyn
            
        #     payload["accel_diagnostics"] = {
        #         "reference_accel_rms_scale_to_imu_dynamic": float(ref_acc_scale),
        #         "reference_accel_lsq_scale_to_imu_dynamic": float(ref_acc_lsq_scale),
        #         "imu_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(accel_body ** 2))),
        #         "imu_gravity_comp_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(accel_body_gravity_comp_dyn ** 2))) if accel_body_gravity_comp_dyn is not None else None,
        #         "reference_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(ref_acc_dyn ** 2))),
        #         "reference_scaled_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(ref_acc_dyn_scaled ** 2))),
        #         "reference_accel_model": "dv_body/dt + omega_body cross v_body",
        #         "gravity_compensation": "accel_body + gravity_initial_body_s2" if accel_body_gravity_comp_dyn is not None else None,
        #     }
        #     save_json(result_json, payload)
            
        #     # 画图时，只画去重力后的观测线和真值线
        #     plot_vector_components(
        #         out / "imu_accel_body_dynamic_alignment.png",
        #         ti,
        #         [
        #             *(
        #                 [("accel body gravity compensated", accel_body_gravity_comp_dyn, "C1")]
        #                 if accel_body_gravity_comp_dyn is not None else []
        #             ),
        #             ("reference body acceleration", ref_acc_dyn, "C3"),
        #             (f"reference scaled (x{ref_acc_scale:.2f})", ref_acc_dyn_scaled, "C2"),
        #         ],
        #         ["x", "y", "z"],
        #         "IMU accelerometer dynamic alignment",
        #         "m/s^2",
        #     )

        # if accel is not None:
        #     accel_i = accel[overlap]
        #     accel_body = (r_bi_final @ accel_i.T).T
        #     ref_acc_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, ref_acc)
        #     ref_vel_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, v_ref_for_accel)
        #     ref_w_on_acc = interp_vectors(ti - float(result.x[3]), t_ref, w_ref)
        #     ref_body_acc = ref_acc_on_imu + np.cross(ref_w_on_acc, ref_vel_on_imu)
            
        #     accel_body_gravity_comp_dyn = None
        #     translation = payload.get("translation_calibration") or {}
        #     if translation.get("status") == "estimated":
        #         g_const = np.array(translation["gravity_initial_body_m_s2"])
                
        #         # 1. 先减去常数重力。注意符号：之前是 + 导致 Z 轴飞到 20，说明传感器内部测的包含了向下的重力。
        #         accel_body_gravity_comp = accel_body - g_const
                
        #         # 2. 滑动去趋势（高通滤波），消除由于姿态倾斜引起未扣除干净的重力大波浪
        #         def _sliding_detrend(x, win_sec, dt):
        #             win = int(win_sec / dt)
        #             if win < 3: return x
        #             # 使用卷积计算滑动平均
        #             pad = win // 2
        #             x_pad = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
        #             kernel = np.ones(win) / win
        #             trend = np.zeros_like(x)
        #             for i in range(3):
        #                 trend[:, i] = np.convolve(x_pad[:, i], kernel, mode='valid')
        #             return x - trend
                
        #         dt = np.median(np.diff(ti))
        #         # 用 2 秒的窗口去趋势
        #         accel_body_gravity_comp_dyn = _sliding_detrend(accel_body_gravity_comp, win_sec=2.0, dt=dt)
            
        #     # 真值也做同样的去趋势处理
        #     dt = np.median(np.diff(ti))
        #     def _sliding_detrend_ref(x, win_sec, dt):
        #         win = int(win_sec / dt)
        #         pad = win // 2
        #         x_pad = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
        #         kernel = np.ones(win) / win
        #         trend = np.zeros_like(x)
        #         for i in range(3):
        #             trend[:, i] = np.convolve(x_pad[:, i], kernel, mode='valid')
        #         return x - trend

        #     ref_acc_dyn = _sliding_detrend_ref(ref_body_acc, win_sec=2.0, dt=dt)
            
        #     ref_acc_lsq_scale = _finite_flat_scale(ref_acc_dyn, accel_body_gravity_comp_dyn if accel_body_gravity_comp_dyn is not None else accel_body)
            
        #     if translation.get("status") == "estimated" and "accel_scale_factor" in translation:
        #         ref_acc_scale = translation["accel_scale_factor"]
        #     else:
        #         scale_target = accel_body_gravity_comp_dyn if accel_body_gravity_comp_dyn is not None else accel_body
        #         ref_acc_scale = _finite_rms_scale(ref_acc_dyn, scale_target)
            
        #     ref_acc_dyn_scaled = ref_acc_scale * ref_acc_dyn
            
        #     payload["accel_diagnostics"] = {
        #         "reference_accel_rms_scale_to_imu_dynamic": float(ref_acc_scale),
        #         "reference_accel_lsq_scale_to_imu_dynamic": float(ref_acc_lsq_scale),
        #         "imu_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(accel_body_gravity_comp_dyn ** 2))) if accel_body_gravity_comp_dyn is not None else None,
        #         "reference_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(ref_acc_dyn ** 2))),
        #         "reference_scaled_dynamic_rms_m_s2": float(np.sqrt(np.nanmean(ref_acc_dyn_scaled ** 2))),
        #         "reference_accel_model": "dv_body/dt + omega_body cross v_body",
        #     }
        #     save_json(result_json, payload)
            
        #     plot_vector_components(
        #         out / "imu_accel_body_dynamic_alignment.png",
        #         ti,
        #         [
        #             *(
        #                 [("accel body gravity compensated", accel_body_gravity_comp_dyn, "C1")]
        #                 if accel_body_gravity_comp_dyn is not None else []
        #             ),
        #             ("reference body acceleration", ref_acc_dyn, "C3"),
        #             (f"reference scaled (x{ref_acc_scale:.2f})", ref_acc_dyn_scaled, "C2"),
        #         ],
        #         ["x", "y", "z"],
        #         "IMU accelerometer dynamic alignment (High-pass filtered)",
        #         "m/s^2",
        #     )

        if accel is not None:
            # 直接从 debug_store 读取局部重力补偿的结果
            accel_debug_store = accel_debug_store or {}
            if "accel_compensated" in accel_debug_store:
                t_comp = accel_debug_store["t_imu"]
                accel_comp = accel_debug_store["accel_compensated"]
                ref_acc = accel_debug_store["u_aref"]
                s_est = accel_debug_store["scale_factor"]
            else:
                # 如果没有 debug_store，退回到原来的方式
                accel_i = accel[overlap]
                accel_body = (r_bi_final @ accel_i.T).T
                ref_acc_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, ref_acc)
                ref_vel_on_imu = interp_vectors(ti - float(result.x[3]), t_ref, v_ref_for_accel)
                ref_w_on_acc = interp_vectors(ti - float(result.x[3]), t_ref, w_ref)
                ref_body_acc = ref_acc_on_imu + np.cross(ref_w_on_acc, ref_vel_on_imu)
                t_comp = ti
                accel_comp = accel_body
                ref_acc = ref_body_acc
                s_est = translation.get("accel_scale_factor", 1.0) if translation.get("status") == "estimated" else 1.0
            
            # 直接对比局部重力补偿后的观测值与真值
            plot_vector_components(
                out / "imu_accel_body_dynamic_alignment.png",
                t_comp,
                [
                    ("IMU accel (local gravity compensated)", accel_comp, "C1"),
                    ("Reference body acceleration", ref_acc, "C3"),
                    (f"Reference scaled (x{s_est:.2f})", ref_acc * s_est, "C2"),
                ],
                ["x", "y", "z"],
                "IMU accelerometer dynamic alignment (Local Gravity)",
                "m/s^2",
            )

        plot_residual_hist(out / "imu_initial_gyro_residual_hist.png", initial_residual, "IMU initial gyroscope residuals", "residual (rad/s)")
        plot_residual_hist(out / "imu_gyro_residual_hist.png", residual, "IMU gyroscope residuals", "residual (rad/s)")

    try:
        _debug_imu_pipeline(
            t_ref=t_ref, v_ref=v_ref, w_ref=w_ref,
            t_imu=t_imu, accel=accel, gyro=gyro,
            ti=ti, gi=gi,
            tau0=tau0, off_cc=off_cc, tau_final=float(result.x[3]),
            corr=corr, lags=lags,
            rvec0=rvec0, rvec_final=result.x[:3],
            bg0=bg0, bg_final=result.x[4:7],
            initial_residual=initial_residual, residual=residual, pred=pred,
            translation_payload=payload.get("translation_calibration", {}),
            ref_acc=ref_acc, v_ref_for_accel=v_ref_for_accel,
            stage_cfg=stage_cfg,
        )
    except Exception as _dbg_e:
        print(f"[DEBUG] IMU Pipeline analysis exception: {_dbg_e}")

    try:
        _debug_gyro_stage(
            t_ref=t_ref, w_ref=w_ref,
            ti=ti, gi=gi,
            x0=x0, result=result,
            initial_residual=initial_residual, residual=residual, pred=pred,
            rvec0=rvec0, tau0=tau0, bg0=bg0,
            tau_window=tau_window,
            iteration_history=iteration_history,
            out_dir=out,
            cad_rotvec=None,
        )
    except Exception as _dbg2_e:
        print(f"[DEBUG] Gyro diagnostics exception: {_dbg2_e}")

    return ImuStageResult(name=name, output_dir=out, result_json=result_json, data_npz=data_npz, payload=payload)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as Rot

from .io import ensure_dir, load_event_flow, load_reference_twist, make_se3, save_json
from .math_utils import edge_trim_mask, estimate_offset_cc, interp_vectors
from .plotting import plot_alignment, plot_cross_correlation, plot_residual_hist, plot_scalar_series

EVENT_REPORT_ROTATION_FROM_BODY = np.diag([-1.0, 1.0, -1.0])


def event_model_translation(translation):
    """Map the reported X/Z-reversed lever arm into the flow model convention."""
    return EVENT_REPORT_ROTATION_FROM_BODY @ np.asarray(translation, dtype=float)


def event_pose_in_report_frame(rotation_body_event, translation_body_event):
    """Express the complete event pose in the body frame with X/Z axes reversed."""
    rotation = EVENT_REPORT_ROTATION_FROM_BODY @ np.asarray(rotation_body_event, dtype=float)
    translation = EVENT_REPORT_ROTATION_FROM_BODY @ np.asarray(translation_body_event, dtype=float)
    return {
        "name": "body_xz_reversed",
        "R_report_body": EVENT_REPORT_ROTATION_FROM_BODY.tolist(),
        "R_report_event": rotation.tolist(),
        "p_report_event_m": translation.tolist(),
        "T_report_event": make_se3(rotation, translation),
    }


@dataclass
class EventStageResult:
    name: str
    output_dir: Path
    result_json: Path
    data_npz: Path
    payload: Dict


def _event_activity_by_time(t_event, flow):
    unique_t, idx, count = np.unique(t_event, return_index=True, return_counts=True)
    activity = np.array([np.mean(np.linalg.norm(flow[s:s + c], axis=1)) for s, c in zip(idx, count)])
    return unique_t, activity


def _save_event_slice_diagnostics(out: Path, t_event, flow, scalar):
    unique_t, idx, count = np.unique(t_event, return_index=True, return_counts=True)
    flow_norm = np.linalg.norm(flow, axis=1)
    rows = []
    for t, s, c in zip(unique_t, idx, count):
        vals = flow_norm[s:s + c]
        if scalar is not None:
            sc = scalar[s:s + c]
            rows.append([t, c, np.mean(vals), np.median(vals), np.percentile(vals, 95), np.mean(sc), np.median(sc), np.percentile(sc, 5), np.percentile(sc, 95)])
        else:
            rows.append([t, c, np.mean(vals), np.median(vals), np.percentile(vals, 95), np.nan, np.nan, np.nan, np.nan])
    rows = np.asarray(rows)
    np.savetxt(
        out / "event_slice_statistics.csv",
        rows,
        delimiter=",",
        header="timestamp,count,flow_norm_mean,flow_norm_median,flow_norm_p95,scalar_mean,scalar_median,scalar_p05,scalar_p95",
        comments="",
    )


def _plot_event_residual_by_slice(out: Path, t_event, initial_residual, residual):
    unique_t, idx, count = np.unique(t_event, return_index=True, return_counts=True)
    rows = []
    for t, s, c in zip(unique_t, idx, count):
        ri = initial_residual[s:s + c]
        rf = residual[s:s + c]
        rows.append([t, c, np.sqrt(np.mean(ri * ri)), np.sqrt(np.mean(rf * rf)), np.median(np.abs(rf))])
    rows = np.asarray(rows)
    np.savetxt(
        out / "event_slice_residual_statistics.csv",
        rows,
        delimiter=",",
        header="timestamp,count,initial_rms,final_rms,final_abs_median",
        comments="",
    )
    plot_scalar_series(
        out / "event_residual_by_slice.png",
        [
            ("initial slice RMS", rows[:, 0], rows[:, 2], "0.55"),
            ("final slice RMS", rows[:, 0], rows[:, 3], "C3"),
        ],
        "Event residual by surface slice",
        "residual RMS",
    )
    plot_scalar_series(
        out / "event_slice_activity.png",
        [
            ("slice count", rows[:, 0], rows[:, 1], "0.55"),
            ("flow norm median", rows[:, 0], rows[:, 3], "C3"),
        ],
        "Event slice activity",
        "count / flow norm",
    )


def _refine_offset(t_ref, sig_ref, t_sens, sig_sens, init_offset, search_range):
    def cost(dt):
        aligned = np.interp(t_ref, t_sens - dt, sig_sens, left=np.nan, right=np.nan)
        valid = np.isfinite(aligned)
        if np.count_nonzero(valid) < 20:
            return 1e30
        a = sig_ref[valid] - np.mean(sig_ref[valid])
        b = aligned[valid] - np.mean(aligned[valid])
        denom = max(float(np.std(a) * np.std(b)), 1e-12)
        return float(-np.mean(a * b) / denom)

    res = minimize_scalar(
        cost,
        bounds=(init_offset - search_range, init_offset + search_range),
        method="bounded",
    )
    return float(res.x) if res.success else float(init_offset)


# 法向光流约束 (纯数据，无物理正则化)
def event_residual(params, t_event, x, flow, t_ref, v_ref, w_ref, normal=None):
    rvec = params[:3]
    p_be = event_model_translation(params[3:6])
    tau = params[6]
    r_be = Rot.from_rotvec(rvec).as_matrix()
    r_eb = r_be.T
    tq = t_event - tau
    vb = interp_vectors(tq, t_ref, v_ref)
    wb = interp_vectors(tq, t_ref, w_ref)
    valid = np.isfinite(vb).all(axis=1) & np.isfinite(wb).all(axis=1)
    u_b = vb + np.cross(wb, p_be)
    v_e = (r_eb @ u_b.T).T
    w_e = (r_eb @ wb.T).T
    
    a = np.cross(x, v_e)  # a 是平移引起的切向流方向（平面法向量）
    b = flow - np.cross(x, w_e)
    
    residual = np.einsum("ij,ij->i", a, b)
    
    # ====== 策略二：正交权重 ======
    if normal is not None:
        a_norm = np.linalg.norm(a, axis=1)
        a_unit = a / np.maximum(a_norm[:, None], 1e-12)
        n_norm = np.linalg.norm(normal, axis=1)
        n_unit = normal / np.maximum(n_norm[:, None], 1e-12)
        
        weight = np.abs(np.einsum("ij,ij->i", a_unit, n_unit))
        weight = np.clip(weight, 0.05, 1.0)
        residual = residual * weight
    # =================================

    if not np.all(valid):
        residual = residual.copy()
        residual[~valid] = 1e3
    return residual

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

    try:
        U, S, Vt = np.linalg.svd(result.jac, full_matrices=False)
        svd_info = {
            "singular_values": S.tolist(),
            "condition_number": float(S[0] / max(S[-1], 1e-12)),
            "effective_rank": int(np.sum(S > 1e-6 * S[0])),
            "weakest_direction": Vt[-1].tolist(),
            "param_order": ["rvec_x","rvec_y","rvec_z","p_x","p_y","p_z","tau"],
        }
    except np.linalg.LinAlgError:
        svd_info = None
    rot = Rot.from_rotvec(result.x[:3])

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "num_residuals": int(len(residual)),
        "cost": float(result.cost),
        "rms": rms,
        "sigma2": sigma2,
        "R_body_event": rot.as_matrix().tolist(),
        "rotvec_body_event": result.x[:3].tolist(),
        "euler_xyz_deg_body_event": rot.as_euler("xyz", degrees=True).tolist(),
        "p_body_event_m": result.x[3:6].tolist(),
        "time_offset_event_minus_body_sec": float(result.x[6]),
        "parameter_std": std.tolist() if std is not None else None,
        "covariance": cov.tolist() if cov is not None else None,
        "fim_svd": svd_info,
    }


def _write_event_summary(path: Path, payload: Dict) -> None:
    lines = [
        "# Event Camera Calibration Stage",
        "",
        "## Time initialization",
        f"- coarse offset adjust (s): {payload['time_initialization']['coarse_offset_adjust_sec']}",
        f"- cross-correlation offset (s): {payload['time_initialization']['cross_correlation_offset_sec']}",
        f"- initial time offset (s): {payload['time_initialization']['refined_offset_sec']}",
        f"- final time offset (s): {payload['time_offset_event_minus_body_sec']}",
        f"- max correlation: {payload['time_initialization']['max_cross_correlation']}",
        "",
        "## Data",
        f"- samples after overlap: {payload['initial_guess']['num_samples_after_overlap']} / {payload['initial_guess']['num_samples_before_overlap']}",
        f"- flow scale: {payload['inputs']['flow_scale']}",
        f"- flow norm median: {payload['initial_guess']['flow_norm_median']}",
        "",
        "## Optimization",
        f"- initial RMS: {payload.get('initial_rms')}",
        f"- final RMS: {payload['rms']}",
        f"- Euler xyz body<-event (deg): {payload['euler_xyz_deg_body_event']}",
        f"- p_body_event (m): {payload['p_body_event_m']}",
        f"- p_report_event (m), body_xz_reversed: {payload['reporting_frame']['p_report_event_m']}",
        "",
        "## Key figures",
        "- event_time_alignment.png",
        "- event_initial_residual_hist.png",
        "- event_residual_hist.png",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_event_reference_stage(stage_cfg: Dict, reference_cfg: Dict, output_root: Path, config_dir: Path) -> EventStageResult:
    name = stage_cfg.get("name", "event")
    out = output_root / name
    ensure_dir(out)

    ref_path = Path(reference_cfg["twist"]).expanduser()
    event_path = Path(stage_cfg["inputs"]["event_flow"]).expanduser()
    if not ref_path.is_absolute():
        ref_path = (config_dir / ref_path).resolve()
    if not event_path.is_absolute():
        event_path = (config_dir / event_path).resolve()

    io_cfg = stage_cfg.get("io") or {}
    inp = stage_cfg.get("inputs") or {}
    t_ref, v_ref, w_ref = load_reference_twist(ref_path, delimiter=io_cfg.get("reference_delimiter"))
    
    t_e, x, normal, flow, scalar = load_event_flow(
        event_path,
        x_cols=inp.get("x_cols", "1:4"),
        normal_cols=inp.get("normal_cols", "4:6"),
        normal_speed_col=inp.get("normal_speed_col"),
        alpha_col=inp.get("alpha_col"),
        flow_mode=inp.get("flow_mode", "auto"),
        delimiter=io_cfg.get("event_delimiter"),
    )
    flow_scale = float(inp.get("flow_scale", 1.0))
    flow = flow_scale * flow

    # 检测并切掉大空档
    t = t_e
    dt = np.diff(t)
    big_gaps = np.where(dt > 0.1)[0]
    if len(big_gaps) > 0:
        segments = np.split(np.arange(len(t)), big_gaps + 1)
        longest = max(segments, key=len)
        t_e, x, normal, flow = t_e[longest], x[longest], normal[longest], flow[longest]
        if scalar is not None:
            scalar = scalar[longest]

    preprocessing_cfg = stage_cfg.get("preprocessing") or {}
    trim_edges_sec = float(preprocessing_cfg.get("trim_edges_sec", 0.0))
    edge_mask = edge_trim_mask(t_e, trim_edges_sec)
    t_e, x, normal, flow = t_e[edge_mask], x[edge_mask], normal[edge_mask], flow[edge_mask]
    if scalar is not None:
        scalar = scalar[edge_mask]

    rng = np.random.default_rng(int(stage_cfg.get("seed", 7)))
    max_samples = int(inp.get("max_samples", 50000))
    if max_samples > 0 and len(t_e) > max_samples:
        sel = np.sort(rng.choice(len(t_e), max_samples, replace=False))
        t_e, x, normal, flow = t_e[sel], x[sel], normal[sel], flow[sel]
        if scalar is not None:
            scalar = scalar[sel]

    time_cfg = stage_cfg.get("time") or {}
    ablation = stage_cfg.get("ablation") or {}
    max_lag = float(time_cfg.get("max_lag_sec", 0.5))
    search_range = float(time_cfg.get("search_range_sec", 0.03))
    coarse_start = bool(time_cfg.get("coarse_start", False))
    coarse_adjust = float(time_cfg.get("coarse_offset_adjust_sec", 0.0))
    ref_sig = np.linalg.norm(w_ref, axis=1)
    t_act, event_sig = _event_activity_by_time(t_e, flow)
    initial_offset = time_cfg.get("initial_offset_sec")
    raw_offset = (float(t_act[0] - t_ref[0]) if coarse_start else 0.0) + coarse_adjust
    t_act_coarse = t_act - raw_offset
    if ablation.get("skip_time_init", False):
        off_cc, corr, lags = 0.0, np.zeros(1), np.zeros(1)
        tau0 = raw_offset
    else:
        off_cc, corr, lags = estimate_offset_cc(t_ref, ref_sig, t_act_coarse, event_sig, max_lag_sec=max_lag)
    if not ablation.get("skip_time_init", False) and initial_offset is not None:
        raw_offset = 0.0
        t_act_coarse = t_act
        off_cc = float(initial_offset)
        tau0 = _refine_offset(t_ref, ref_sig, t_act_coarse, event_sig, off_cc, search_range)
    elif not ablation.get("skip_time_init", False):
        tau0 = raw_offset + _refine_offset(t_ref, ref_sig, t_act_coarse, event_sig, off_cc, search_range)

    init_cfg = stage_cfg.get("initialization") or {}
    rvec0 = np.asarray(init_cfg.get("rotvec_body_event", [0.0, 0.0, 0.0]), dtype=float)
    p0 = np.asarray(init_cfg.get("translation_m", [0.0, 0.0, 0.0]), dtype=float)
    
    opt_cfg = stage_cfg.get("optimization") or {}
    tau_window = float(opt_cfg.get("tau_refine_window_sec", search_range))
    rot_bound = float(opt_cfg.get("rotvec_bound_rad", np.pi))
    trans_bound = float(opt_cfg.get("translation_bound_m", 5.0))
    loss_type = opt_cfg.get("loss", "huber")
    f_scale = float(opt_cfg.get("f_scale", 0.2))
    max_nfev = int(opt_cfg.get("max_nfev", 300))

    common_radius = float(ablation.get("common_time_radius_sec", 0.0))
    overlap_center = raw_offset if common_radius > 0.0 else tau0
    overlap_radius = common_radius if common_radius > 0.0 else tau_window
    overlap = (t_e - overlap_center >= t_ref[0] + overlap_radius + 0.01) & (t_e - overlap_center <= t_ref[-1] - overlap_radius - 0.01)
    t_opt = t_e[overlap]
    x_opt = x[overlap]
    normal_opt = normal[overlap]
    flow_opt = flow[overlap]
    if len(t_opt) < 20:
        raise ValueError("Too few event samples overlap the reference twist after time alignment.")

    print("=" * 60)
    print(">>> 开始分步解耦优化 (纯数学，无先验) <<<")
    
    # =========================================================
    # 阶段 1：锁定 p=0，优化 R 和 tau
    # =========================================================
    def res_step1(params):
        full_params = np.r_[params[:3], 0.0, 0.0, 0.0, params[3]]
        return event_residual(full_params, t_opt, x_opt, flow_opt, t_ref, v_ref, w_ref, normal=normal_opt)

    x0_step1 = np.r_[rvec0, tau0]
    lower1 = np.r_[-rot_bound, -rot_bound, -rot_bound, tau0 - tau_window]
    upper1 = np.r_[ rot_bound,  rot_bound,  rot_bound, tau0 + tau_window]

    if ablation.get("skip_rotation_init", False):
        rvec_step1, tau_step1 = rvec0.copy(), tau0
        rms1 = np.sqrt(np.mean(res_step1(x0_step1)**2))
    elif ablation.get("skip_time_init", False):
        res1 = least_squares(
            lambda r: res_step1(np.r_[r, tau0]), rvec0,
            bounds=(lower1[:3], upper1[:3]), method="trf",
            loss=loss_type, f_scale=f_scale, max_nfev=max_nfev, diff_step=5e-2,
        )
        rvec_step1, tau_step1 = res1.x, tau0
        rms1 = np.sqrt(np.mean(res1.fun**2))
    else:
        res1 = least_squares(
            res_step1, x0_step1,
            bounds=(lower1, upper1), method="trf",
            loss=loss_type, f_scale=f_scale, max_nfev=max_nfev,
            diff_step=5e-2,
        )
        rvec_step1, tau_step1 = res1.x[:3], res1.x[3]
        rms1 = np.sqrt(np.mean(res1.fun**2))
    print(f"[Step 1] 纯旋转优化: R_rvec={rvec_step1}, tau={tau_step1:.4f}, RMS={rms1:.4f}")

    # =========================================================
    # 【新增：解析线性求解平移初值】
    # 利用已知的 r_eb, vb, wb 构造 Ap=b，避免从 [0,0,0] 盲搜
    # =========================================================
    r_eb_step1 = Rot.from_rotvec(rvec_step1).as_matrix().T
    tq_opt = t_opt - tau_step1
    vb_opt = interp_vectors(tq_opt, t_ref, v_ref)
    wb_opt = interp_vectors(tq_opt, t_ref, w_ref)
    v_e_opt = (r_eb_step1 @ vb_opt.T).T
    w_e_opt = (r_eb_step1 @ wb_opt.T).T
    df_opt = flow_opt - np.cross(x_opt, w_e_opt)
    
    N = len(x_opt)
    A_mat = np.zeros((N, 3))
    b_vec = np.zeros(N)
    for i in range(N):
        x_i = x_opt[i]
        w_i = wb_opt[i]
        v_i = vb_opt[i]
        df_i = df_opt[i]
        
        xc = np.array([[0, -x_i[2], x_i[1]], [x_i[2], 0, -x_i[0]], [-x_i[1], x_i[0], 0]])
        wc = np.array([[0, -w_i[2], w_i[1]], [w_i[2], 0, -w_i[0]], [-w_i[1], w_i[0], 0]])
        
        A_mat[i, :] = -df_i @ (xc @ r_eb_step1 @ wc)
        b_vec[i] = -df_i @ (xc @ (r_eb_step1 @ v_i))
        
    if ablation.get("skip_translation_init", False):
        p0_lin = np.zeros(3)
    else:
        p0_lin_model, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        p0_lin = event_model_translation(p0_lin_model)
    print(f"[Step 2 Prep] 解析平移初值: p0 = {p0_lin}")

    # =========================================================
    # 阶段 2：固定 R 和 tau，优化 p
    # =========================================================
    def res_step2(params):
        full_params = np.r_[rvec_step1, params, tau_step1]
        return event_residual(full_params, t_opt, x_opt, flow_opt, t_ref, v_ref, w_ref, normal=normal_opt)

    x0_step2 = p0_lin.copy() # 用解析解替代 [0,0,0]
    lower2 = np.array([-trans_bound, -trans_bound, -trans_bound])
    upper2 = np.array([ trans_bound,  trans_bound,  trans_bound])

    if ablation.get("skip_translation_init", False):
        p_step2 = np.zeros(3)
        rms2 = np.sqrt(np.mean(res_step2(p_step2)**2))
    else:
        res2 = least_squares(
            res_step2, x0_step2,
            bounds=(lower2, upper2), method="trf",
            loss=loss_type, f_scale=f_scale, max_nfev=max_nfev,
            diff_step=5e-2,
        )
        p_step2 = res2.x
        rms2 = np.sqrt(np.mean(res2.fun**2))
    print(f"[Step 2] 纯平移优化: p={p_step2}, RMS={rms2:.4f}")

    # =========================================================
    # 【一维扫描】扩大扫描范围 [-0.5, 0.5]
    # =========================================================
    print("\n>>> 开始 p_x 一维扫描 (固定 R 和 tau) <<<")
    p_x_candidates = np.linspace(-0.5, 0.5, 21) 
    for px in p_x_candidates:
        p_test = np.array([px, p_step2[1], p_step2[2]])
        full_params = np.r_[rvec_step1, p_test, tau_step1]
        res_test = event_residual(full_params, t_opt, x_opt, flow_opt, t_ref, v_ref, w_ref, normal=normal_opt)
        cost_test = np.mean(res_test ** 2)
        print(f"  p_x = {px:+.3f} -> RMS = {np.sqrt(cost_test):.4f}")
    print("=" * 60)
    
    # =========================================================
    # 阶段 3：联合微调
    # =========================================================
    print("[Step 3] 联合微调...")
    x0_joint = np.r_[rvec_step1, p_step2, tau_step1]
    lower_joint = np.array([-rot_bound, -rot_bound, -rot_bound, -trans_bound, -trans_bound, -trans_bound, tau0 - tau_window])
    upper_joint = np.array([ rot_bound,  rot_bound,  rot_bound,  trans_bound,  trans_bound,  trans_bound, tau0 + tau_window])
    if common_radius > 0.0:
        lower_joint[6], upper_joint[6] = raw_offset - common_radius, raw_offset + common_radius

    result = least_squares(
        event_residual, x0_joint,
        args=(t_opt, x_opt, flow_opt, t_ref, v_ref, w_ref, normal_opt),
        bounds=(lower_joint, upper_joint), method="trf",
        loss=loss_type, f_scale=f_scale, max_nfev=max_nfev,
        diff_step=5e-2,
    )
    print(f"[Step 3] 联合结果: RMS={np.sqrt(np.mean(result.fun**2)):.4f}")
    print("=" * 60)

    # 后续数据保存逻辑不变
    initial_residual = event_residual(x0_joint, t_opt, x_opt, flow_opt, t_ref, v_ref, w_ref, normal=normal_opt)
    residual = event_residual(result.x, t_opt, x_opt, flow_opt, t_ref, v_ref, w_ref, normal=normal_opt)
    
    payload = {
        "stage": name,
        "type": "event_reference",
        "frame_convention": {
            "rotation": "R_body_event maps event-camera frame to body frame",
            "translation": "p_body_event is event-camera origin expressed in body frame",
            "time": "t_body = t_event - tau",
            "residual": "dot(cross(x, v_event), normal_flow - cross(x, omega_event))",
        },
        "inputs": {
            "reference_twist": str(ref_path),
            "event_flow": str(event_path),
            "x_cols": inp.get("x_cols", "1:4"),
            "normal_cols": inp.get("normal_cols", "4:6"),
            "normal_speed_col": inp.get("normal_speed_col"),
            "alpha_col": inp.get("alpha_col"),
            "flow_mode": inp.get("flow_mode", "auto"),
            "flow_scale": flow_scale,
            "max_samples": inp.get("max_samples", 50000),
            "seed": int(stage_cfg.get("seed", 7)),
        },
        "time_initialization": {
            "raw_start_difference_sec": raw_offset,
            "coarse_offset_adjust_sec": coarse_adjust,
            "initial_offset_sec": float(initial_offset) if initial_offset is not None else None,
            "cross_correlation_offset_sec": off_cc,
            "refined_offset_sec": tau0,
            "max_cross_correlation": float(np.max(corr)),
        },
        "initial_guess": {
            "rotvec_body_event": rvec0.tolist(),
            "p_body_event_m": p0.tolist(),
            "time_offset_sec": tau0,
            "num_samples_before_overlap": int(len(t_e)),
            "num_samples_after_overlap": int(len(t_opt)),
            "flow_norm_median": float(np.median(np.linalg.norm(flow_opt, axis=1))),
            "flow_norm_p95": float(np.percentile(np.linalg.norm(flow_opt, axis=1), 95)),
        },
        "preprocessing": {
            "trim_edges_sec": trim_edges_sec,
            "num_samples_after_edge_trim": int(len(t_e)),
        },
        "staged_results": {
            "step1_rotation_tau_rms": float(rms1),
            "step2_translation_rms": float(rms2),
            "step1_rvec": rvec_step1.tolist(),
            "step1_tau": float(tau_step1),
            "step2_p": p_step2.tolist(),
            "p0_lin": p0_lin.tolist(),
        }
    }
    payload.update(_summarize(result, residual))
    payload["T_body_event"] = make_se3(payload["R_body_event"], payload["p_body_event_m"])
    payload["translation_parameter_convention"] = "reported X/Z-reversed; event_model_translation maps to the flow-model lever arm"
    payload["p_body_event_model_m"] = event_model_translation(payload["p_body_event_m"]).tolist()
    payload["reporting_frame"] = {
        "name": "event_translation_parameter",
        "p_report_event_m": payload["p_body_event_m"],
    }
    payload["initial_rms"] = float(np.sqrt(np.mean(initial_residual * initial_residual)))

    result_json = out / "event_result.json"
    data_npz = out / "event_stage_data.npz"
    save_json(result_json, payload)
    _save_event_slice_diagnostics(out, t_opt, flow_opt, scalar[overlap] if scalar is not None else None)
    _plot_event_residual_by_slice(out, t_opt, initial_residual, residual)
    _write_event_summary(out / "stage_summary.md", payload)
    np.savez_compressed(
        data_npz,
        t_event=t_opt,
        x=x_opt,
        normal=normal_opt,
        flow=flow_opt,
        residual=residual,
        initial_residual=initial_residual,
        params=result.x,
        initial_params=x0_joint,
    )
    np.savetxt(out / "event_residuals.csv", residual, delimiter=",", header="residual", comments="")
    np.savetxt(out / "event_initial_residuals.csv", initial_residual, delimiter=",", header="residual", comments="")
    np.savetxt(
        out / "event_step_metrics.csv",
        np.array([
            ["coarse_offset_adjust_sec", coarse_adjust],
            ["cross_correlation_offset_sec", off_cc],
            ["initial_time_offset_sec", tau0],
            ["final_time_offset_sec", float(result.x[6])],
            ["flow_scale", flow_scale],
            ["flow_norm_median", payload["initial_guess"]["flow_norm_median"]],
            ["initial_rms", payload["initial_rms"]],
            ["final_rms", payload["rms"]],
        ], dtype=object),
        delimiter=",",
        fmt="%s",
    )

    if bool((stage_cfg.get("diagnostics") or {}).get("save_plots", True)):
        plot_cross_correlation(out / "event_cross_correlation.png", lags, corr, off_cc, "Event/reference activity cross-correlation")
        plot_alignment(out / "event_time_alignment.png", t_ref, ref_sig, t_act, event_sig, float(result.x[6]), "Event-reference timing cue alignment", "activity")
        plot_residual_hist(out / "event_initial_residual_hist.png", initial_residual, "Event initial normal-flow residuals", "residual")
        plot_residual_hist(out / "event_residual_hist.png", residual, "Event normal-flow residuals", "residual")

    return EventStageResult(name=name, output_dir=out, result_json=result_json, data_npz=data_npz, payload=payload)

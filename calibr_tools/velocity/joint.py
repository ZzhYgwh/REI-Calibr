from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from .event import event_residual
from .imu import imu_accel_window_residual, imu_gyro_residual
from .io import ensure_dir, load_reference_twist, make_se3, save_json
from .radar import radar_body_residual


@dataclass
class JointStageResult:
    name: str
    output_dir: Path
    result_json: Path
    data_npz: Optional[Path]
    payload: Dict


def _load_stage_records(stage_cfg: Dict, stage_records: List[Dict]) -> List[Dict]:
    requested = set((stage_cfg.get("inputs") or {}).get("stages", []))
    selected = []
    for record in stage_records:
        if requested and record.get("name") not in requested:
            continue
        item = dict(record)
        result_path = item.get("result_json")
        if result_path and Path(result_path).exists():
            with open(result_path, "r", encoding="utf-8") as f:
                item["result"] = json.load(f)
        selected.append(item)
    return selected


def run_joint_manifest_stage(stage_cfg: Dict, stage_records: List[Dict], output_root: Path) -> JointStageResult:
    name = stage_cfg.get("name", "joint")
    out = output_root / name
    ensure_dir(out)
    selected = _load_stage_records(stage_cfg, stage_records)

    payload = {
        "stage": name,
        "type": "joint_manifest",
        "status": "manifest_only",
        "note": "This stage collects independent calibration outputs. Use joint_refine for nonlinear joint residual optimization.",
        "selected_stages": selected,
    }
    result_json = out / "joint_manifest.json"
    save_json(result_json, payload)
    return JointStageResult(name=name, output_dir=out, result_json=result_json, data_npz=None, payload=payload)


def _stage_kind(result: Dict) -> Optional[str]:
    if "R_body_radar" in result:
        return "radar"
    if "R_body_event" in result:
        return "event"
    if "R_body_imu" in result:
        return "imu"
    return None


def _make_block(record: Dict, t_ref, v_ref, w_ref, cfg: Dict) -> Dict | None:
    result = record.get("result") or {}
    data_path = record.get("data_npz")
    if not data_path or not Path(data_path).exists():
        return None
    data = np.load(data_path)
    kind = _stage_kind(result)
    if kind == "radar":
        params = np.asarray(data["params"], dtype=float)
        scale_mode = result.get("scale_mode", "global")
        weight = float(cfg.get("weights", {}).get(record["name"], cfg.get("weights", {}).get("radar", 1.0)))
        return {
            "name": record["name"],
            "kind": kind,
            "params": params,
            "weight": weight,
            "scale_mode": scale_mode,
            "data": {
                "t": np.asarray(data["t_radar"], dtype=float),
                "v": np.asarray(data["radar_velocity"], dtype=float),
            },
            "bounds": _radar_bounds(params, cfg, scale_mode),
        }
    if kind == "event":
        params = np.asarray(data["params"], dtype=float)
        weight = float(cfg.get("weights", {}).get(record["name"], cfg.get("weights", {}).get("event", 1.0)))
        return {
            "name": record["name"],
            "kind": kind,
            "params": params,
            "weight": weight,
            "data": {
                "t": np.asarray(data["t_event"], dtype=float),
                "x": np.asarray(data["x"], dtype=float),
                "flow": np.asarray(data["flow"], dtype=float),
            },
            "bounds": _event_bounds(params, cfg),
        }
    if kind == "imu":
        params = np.asarray(data["params"], dtype=float)
        weight = float(cfg.get("weights", {}).get(record["name"], cfg.get("weights", {}).get("imu", 1.0)))
        block = {
            "name": record["name"],
            "kind": kind,
            "params": params,
            "weight": weight,
            "data": {
                "t": np.asarray(data["t_imu"], dtype=float),
                "gyro": np.asarray(data["gyro"], dtype=float),
            },
        }
        accel_cfg = cfg.get("imu_accel") or {}
        if accel_cfg.get("enabled", False):
            calibration = result.get("translation_calibration") or {}
            required = {"accel", "accel_reference_velocity", "accel_reference_omega",
                        "accel_window_sec", "accel_step_sec", "accel_bias_sigma"}
            if calibration.get("status") != "estimated" or not required.issubset(data.files):
                raise ValueError(f"{record['name']} has no estimated acceleration state or matching stage data")
            accel = np.asarray(data["accel"], dtype=float)
            if len(accel) != len(block["data"]["t"]):
                raise ValueError(f"{record['name']} acceleration timestamps do not match samples")
            block["data"].update({
                "accel": accel,
                "v_accel_ref": np.asarray(data["accel_reference_velocity"], dtype=float),
                "w_accel_ref": np.asarray(data["accel_reference_omega"], dtype=float),
            })
            block["accel_cfg"] = {
                "window_sec": float(data["accel_window_sec"]),
                "step_sec": float(data["accel_step_sec"]),
                "bias_sigma": float(data["accel_bias_sigma"]),
            }
            block["accel_weight"] = float(accel_cfg.get("weight", 1.0))
            block["params"] = np.r_[params, result["p_body_imu_m"],
                                    result["accel_bias_imu_m_s2"], calibration["accel_scale_factor"]]
        block["bounds"] = _imu_bounds(block["params"], cfg)
        return block
    return None


def _radar_bounds(params, cfg, scale_mode):
    rot_delta = float(cfg.get("rot_delta_bound_rad", 0.35))
    trans_delta = float(cfg.get("translation_delta_bound_m", 0.30))
    tau_delta = float(cfg.get("tau_delta_bound_sec", 0.03))
    log_scale_delta = float(cfg.get("log_scale_delta_bound", 0.35))
    lower = params.copy()
    upper = params.copy()
    lower[:3] -= rot_delta
    upper[:3] += rot_delta
    lower[3:6] -= trans_delta
    upper[3:6] += trans_delta
    lower[6] -= tau_delta
    upper[6] += tau_delta
    if scale_mode == "global":
        lower[7] -= log_scale_delta
        upper[7] += log_scale_delta
    else:
        lower[7] = -1e-12
        upper[7] = 1e-12
    return lower, upper


def _event_bounds(params, cfg):
    rot_delta = float(cfg.get("rot_delta_bound_rad", 0.35))
    trans_delta = float(cfg.get("translation_delta_bound_m", 0.30))
    tau_delta = float(cfg.get("tau_delta_bound_sec", 0.03))
    lower = params.copy()
    upper = params.copy()
    lower[:3] -= rot_delta
    upper[:3] += rot_delta
    lower[3:6] -= trans_delta
    upper[3:6] += trans_delta
    lower[6] -= tau_delta
    upper[6] += tau_delta
    return lower, upper


def _imu_bounds(params, cfg):
    rot_delta = float(cfg.get("rot_delta_bound_rad", 0.35))
    tau_delta = float(cfg.get("tau_delta_bound_sec", 0.03))
    bias_delta = float(cfg.get("gyro_bias_delta_bound_rad_s", 0.20))
    lower = params.copy()
    upper = params.copy()
    lower[:3] -= rot_delta
    upper[:3] += rot_delta
    lower[3] -= tau_delta
    upper[3] += tau_delta
    lower[4:7] -= bias_delta
    upper[4:7] += bias_delta
    if len(params) > 7:
        lower[7:10] -= float(cfg.get("translation_delta_bound_m", 0.30))
        upper[7:10] += float(cfg.get("translation_delta_bound_m", 0.30))
        lower[10:13] -= float(cfg.get("accel_bias_delta_bound_m_s2", 0.20))
        upper[10:13] += float(cfg.get("accel_bias_delta_bound_m_s2", 0.20))
        scale_delta = float(cfg.get("accel_scale_delta_bound", 0.35))
        lower[13] = max(0.5, params[13] - scale_delta)
        upper[13] = min(5.0, params[13] + scale_delta)
    return lower, upper


def _imu_accel_residual(block: Dict, params, t_ref, regularize=True):
    d = block["data"]
    return imu_accel_window_residual(
        params, d["t"], d["accel"], t_ref, d["v_accel_ref"], d["w_accel_ref"],
        **block["accel_cfg"], regularize=regularize,
    )


def _block_residual(block: Dict, params, t_ref, v_ref, w_ref):
    d = block["data"]
    if block["kind"] == "radar":
        return radar_body_residual(params, d["t"], d["v"], t_ref, v_ref, w_ref, block["scale_mode"])
    if block["kind"] == "event":
        return event_residual(params, d["t"], d["x"], d["flow"], t_ref, v_ref, w_ref)
    if block["kind"] == "imu":
        gyro = imu_gyro_residual(params, d["t"], d["gyro"], t_ref, w_ref)
        if "accel_cfg" not in block:
            return gyro
        accel = _imu_accel_residual(block, params, t_ref)
        return np.r_[gyro, np.sqrt(block["accel_weight"]) * accel]
    raise ValueError(f"Unknown joint block kind: {block['kind']}")


def _split_params(x, blocks):
    out = []
    offset = 0
    for block in blocks:
        n = len(block["params"])
        out.append(x[offset:offset + n])
        offset += n
    return out


def _joint_residual(x, blocks, t_ref, v_ref, w_ref):
    chunks = []
    for block, params in zip(blocks, _split_params(x, blocks)):
        chunks.append(np.sqrt(block["weight"]) * _block_residual(block, params, t_ref, v_ref, w_ref))
    return np.concatenate(chunks) if chunks else np.empty(0)


def _rms(residual):
    return float(np.sqrt(np.mean(np.asarray(residual) ** 2))) if len(residual) else float("nan")


def _summarize_block(block, before_params, after_params, t_ref, v_ref, w_ref):
    rb = _block_residual(block, before_params, t_ref, v_ref, w_ref)
    ra = _block_residual(block, after_params, t_ref, v_ref, w_ref)
    item = {
        "name": block["name"],
        "kind": block["kind"],
        "weight": block["weight"],
        "before_rms": _rms(rb),
        "after_rms": _rms(ra),
        "delta_norm": float(np.linalg.norm(after_params - before_params)),
        "params_before": before_params.tolist(),
        "params_after": after_params.tolist(),
    }
    if block["kind"] == "radar":
        rot = Rot.from_rotvec(after_params[:3])
        item.update({
            "R_body_radar": rot.as_matrix().tolist(),
            "p_body_radar_m": after_params[3:6].tolist(),
            "T_body_radar": make_se3(rot.as_matrix(), after_params[3:6]),
            "time_offset_radar_minus_body_sec": float(after_params[6]),
            "velocity_scale": float(np.exp(after_params[7])) if block.get("scale_mode") == "global" else 1.0,
            "euler_xyz_deg_body_radar": rot.as_euler("xyz", degrees=True).tolist(),
        })
    elif block["kind"] == "event":
        rot = Rot.from_rotvec(after_params[:3])
        item.update({
            "R_body_event": rot.as_matrix().tolist(),
            "p_body_event_m": after_params[3:6].tolist(),
            "T_body_event": make_se3(rot.as_matrix(), after_params[3:6]),
            "time_offset_event_minus_body_sec": float(after_params[6]),
            "euler_xyz_deg_body_event": rot.as_euler("xyz", degrees=True).tolist(),
        })
    elif block["kind"] == "imu":
        rot = Rot.from_rotvec(after_params[:3])
        has_accel = "accel_cfg" in block
        position = after_params[7:10].tolist() if has_accel else [0.0, 0.0, 0.0]
        item.update({
            "R_body_imu": rot.as_matrix().tolist(),
            "p_body_imu_m": position,
            "T_body_imu": make_se3(rot.as_matrix(), position),
            "time_offset_imu_minus_body_sec": float(after_params[3]),
            "gyro_bias_imu_rad_s": after_params[4:7].tolist(),
            "euler_xyz_deg_body_imu": rot.as_euler("xyz", degrees=True).tolist(),
            "gyro_before_rms": _rms(imu_gyro_residual(before_params, block["data"]["t"], block["data"]["gyro"], t_ref, w_ref)),
            "gyro_after_rms": _rms(imu_gyro_residual(after_params, block["data"]["t"], block["data"]["gyro"], t_ref, w_ref)),
        })
        if has_accel:
            item.update({
                "accel_bias_imu_m_s2": after_params[10:13].tolist(),
                "accel_scale_factor": float(after_params[13]),
                "accel_before_rms": _rms(_imu_accel_residual(block, before_params, t_ref, regularize=False)),
                "accel_after_rms": _rms(_imu_accel_residual(block, after_params, t_ref, regularize=False)),
            })
    return item


def _write_joint_summary(path: Path, payload: Dict):
    lines = [
        "# Joint Refinement Stage",
        "",
        f"- status: {payload['status']}",
        f"- initial joint RMS: {payload['initial_joint_rms']}",
        f"- final joint RMS: {payload['final_joint_rms']}",
        f"- optimizer message: {payload['message']}",
        "",
        "| Stage | Type | Before RMS | After RMS | Delta norm |",
        "|---|---|---:|---:|---:|",
    ]
    for item in payload["blocks"]:
        lines.append(
            f"| {item['name']} | {item['kind']} | {item['before_rms']:.6g} | {item['after_rms']:.6g} | {item['delta_norm']:.6g} |"
        )
    lines += ["", "## Refined SE(3)", ""]
    for item in payload["blocks"]:
        key = f"T_body_{item['kind']}"
        if key not in item:
            continue
        lines += [f"### {item['name']}: `{key}`", "", "```text"]
        for row in item[key]:
            lines.append("  [" + ", ".join(f"{float(v): .9f}" for v in row) + "]")
        lines += ["```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_joint_refine_stage(stage_cfg: Dict, reference_cfg: Dict, stage_records: List[Dict], output_root: Path, config_dir: Path) -> JointStageResult:
    name = stage_cfg.get("name", "joint_refine")
    out = output_root / name
    ensure_dir(out)

    ref_path = Path(reference_cfg["twist"]).expanduser()
    if not ref_path.is_absolute():
        ref_path = (config_dir / ref_path).resolve()
    t_ref, v_ref, w_ref = load_reference_twist(ref_path)

    selected = _load_stage_records(stage_cfg, stage_records)
    opt_cfg = stage_cfg.get("optimization") or {}
    blocks = []
    for record in selected:
        block = _make_block(record, t_ref, v_ref, w_ref, opt_cfg)
        if block is not None:
            blocks.append(block)
    if not blocks:
        raise ValueError("joint_refine has no valid input stages with stage_data.npz.")

    x0 = np.concatenate([b["params"] for b in blocks])
    lower = np.concatenate([b["bounds"][0] for b in blocks])
    upper = np.concatenate([b["bounds"][1] for b in blocks])
    initial_residual = _joint_residual(x0, blocks, t_ref, v_ref, w_ref)
    if any("accel_cfg" in block for block in blocks):
        # The fixed-reference objective is a sum of independent sensor blocks.
        subresults = [
            least_squares(
                lambda x, b=block: np.sqrt(b["weight"]) * _block_residual(b, x, t_ref, v_ref, w_ref),
                block["params"], method="trf", loss=opt_cfg.get("loss", "huber"),
                f_scale=float(opt_cfg.get("f_scale", 0.2)),
                max_nfev=int(opt_cfg.get("max_nfev", 200)), bounds=block["bounds"],
            ) for block in blocks
        ]
        final_params = np.concatenate([res.x for res in subresults])
        success = all(res.success for res in subresults)
        message = "; ".join(f"{block['name']}: {res.message}" for block, res in zip(blocks, subresults))
        cost = float(sum(res.cost for res in subresults))
        nfev = {block["name"]: int(res.nfev) for block, res in zip(blocks, subresults)}
    else:
        result = least_squares(
            _joint_residual,
            x0,
            args=(blocks, t_ref, v_ref, w_ref),
            method="trf",
            loss=opt_cfg.get("loss", "huber"),
            f_scale=float(opt_cfg.get("f_scale", 0.2)),
            max_nfev=int(opt_cfg.get("max_nfev", 200)),
            bounds=(lower, upper),
        )
        final_params = result.x
        success, message, cost = bool(result.success), str(result.message), float(result.cost)
        nfev = {"joint": int(result.nfev)}
    final_residual = _joint_residual(final_params, blocks, t_ref, v_ref, w_ref)
    before_chunks = _split_params(x0, blocks)
    after_chunks = _split_params(final_params, blocks)
    block_summaries = [
        _summarize_block(block, b0, b1, t_ref, v_ref, w_ref)
        for block, b0, b1 in zip(blocks, before_chunks, after_chunks)
    ]
    payload = {
        "stage": name,
        "type": "joint_refine",
        "status": "optimized" if success else "failed",
        "success": success,
        "message": message,
        "cost": cost,
        "function_evaluations": nfev,
        "initial_joint_rms": _rms(initial_residual),
        "final_joint_rms": _rms(final_residual),
        "num_residuals": int(len(final_residual)),
        "num_parameters": int(len(final_params)),
        "blocks": block_summaries,
        "note": "Joint residual refinement from stage results; no direct sensor-to-sensor pairwise constraints or CAD priors. IMU acceleration is included only when optimization.imu_accel.enabled is true.",
    }

    result_json = out / "joint_result.json"
    data_npz = out / "joint_stage_data.npz"
    save_json(result_json, payload)
    _write_joint_summary(out / "joint_summary.md", payload)
    np.savez_compressed(
        data_npz,
        initial_params=x0,
        final_params=final_params,
        initial_residual=initial_residual,
        final_residual=final_residual,
    )
    np.savetxt(
        out / "joint_block_metrics.csv",
        np.asarray([[b["name"], b["kind"], b["before_rms"], b["after_rms"], b["delta_norm"]] for b in block_summaries], dtype=object),
        delimiter=",",
        fmt="%s",
        header="stage,type,before_rms,after_rms,delta_norm",
        comments="",
    )
    return JointStageResult(name=name, output_dir=out, result_json=result_json, data_npz=data_npz, payload=payload)

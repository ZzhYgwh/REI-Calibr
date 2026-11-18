from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from .io import ensure_dir


def _load_result(record):
    p = record.get("result_json")
    if not p or not Path(p).exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _sensor_pose(result):
    if not result:
        return None
    if "p_body_radar_m" in result:
        return "Radar", np.asarray(result["p_body_radar_m"], dtype=float), result.get("time_offset_radar_minus_body_sec")
    if "p_body_event_m" in result:
        return "Event", np.asarray(result["p_body_event_m"], dtype=float), result.get("time_offset_event_minus_body_sec")
    if "p_body_imu_m" in result:
        return "IMU", np.asarray(result["p_body_imu_m"], dtype=float), result.get("time_offset_imu_minus_body_sec")
    return None


def _small_time_offset(result):
    ti = result.get("time_initialization", {})
    final = (
        result.get("time_offset_radar_minus_body_sec")
        or result.get("time_offset_event_minus_body_sec")
        or result.get("time_offset_imu_minus_body_sec")
    )
    raw = ti.get("raw_start_difference_sec", 0.0)
    if final is None:
        return None
    return float(final) - float(raw)


def _fmt_num(value, digits=6):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_vec(values):
    if values is None:
        return "n/a"
    return "[" + ", ".join(_fmt_num(v, 6) for v in values) + "]"


def _fmt_matrix(matrix):
    if matrix is None:
        return "n/a"
    lines = []
    for row in matrix:
        lines.append("  [" + ", ".join(f"{float(v): .9f}" for v in row) + "]")
    return "\n".join(lines)


def _quality_label(value, good, weak):
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v <= good:
        return "good"
    if v <= weak:
        return "weak"
    return "poor"


def _radar_quality(result):
    obs = result.get("translation_observability") or {}
    rms = result.get("rms")
    lever_p95 = obs.get("lever_speed_p95")
    lever_med = obs.get("lever_speed_median")
    omega_p95 = obs.get("omega_norm_p95")
    cond = obs.get("numeric_condition", obs.get("analytic_condition"))
    corr = (result.get("time_initialization") or {}).get("max_cross_correlation")
    ratio = None
    if rms is not None and lever_p95 is not None:
        ratio = float(rms) / max(float(lever_p95), 1e-9)
    return {
        "omega_p95": omega_p95,
        "lever_median": lever_med,
        "lever_p95": lever_p95,
        "condition": cond,
        "corr": corr,
        "rms_to_lever_p95": ratio,
        "rank_label": _quality_label(cond, 10.0, 50.0),
        "lever_label": "good" if lever_p95 is not None and lever_p95 >= 0.10 else ("weak" if lever_p95 is not None and lever_p95 >= 0.04 else "poor"),
        "corr_label": "good" if corr is not None and corr >= 0.60 else ("weak" if corr is not None and corr >= 0.30 else "poor"),
    }


def _result_fields(result):
    if "T_body_radar" in result:
        return {
            "sensor": "radar",
            "rotation": result.get("euler_xyz_deg_body_radar"),
            "translation": result.get("p_body_radar_m"),
            "time": result.get("time_offset_radar_minus_body_sec"),
            "small_time": _small_time_offset(result),
            "scale": result.get("velocity_scale"),
            "se3_name": "T_body_radar",
            "se3": result.get("T_body_radar"),
            "figures": [
                "radar_velocity_lowpass.png",
                "radar_time_alignment.png",
                "radar_body_velocity_fit.png",
                "radar_extrinsic_alignment_effect.png",
                "radar_translation_observability.png",
                "radar_residual_hist.png",
            ],
        }
    if "T_body_event" in result:
        return {
            "sensor": "event",
            "rotation": result.get("euler_xyz_deg_body_event"),
            "translation": result.get("p_body_event_m"),
            "time": result.get("time_offset_event_minus_body_sec"),
            "small_time": _small_time_offset(result),
            "scale": None,
            "se3_name": "T_body_event",
            "se3": result.get("T_body_event"),
            "figures": [
                "event_time_alignment.png",
                "event_residual_by_slice.png",
                "event_slice_activity.png",
                "event_residual_hist.png",
            ],
        }
    if "T_body_imu" in result:
        return {
            "sensor": "imu",
            "rotation": result.get("euler_xyz_deg_body_imu"),
            "translation": result.get("p_body_imu_m"),
            "time": result.get("time_offset_imu_minus_body_sec"),
            "small_time": _small_time_offset(result),
            "scale": None,
            "se3_name": "T_body_imu",
            "se3": result.get("T_body_imu"),
            "figures": [
                "imu_time_alignment.png",
                "imu_gyro_body_alignment_effect.png",
                "imu_gyro_components_sensor_frame.png",
                "imu_accel_body_dynamic_alignment.png",
                "imu_gyro_residual_hist.png",
            ],
        }
    return None


def _cad_results():
    path = Path(__file__).resolve().parents[2] / "experiments" / "comb3" / "sensor_to_gt.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return {record["stage"]: record for record in records if record.get("method") == "CAD"}


def write_session_summary(output_root: Path, stage_records):
    lines = [
        "# Velocity-Domain Calibration Summary",
        "",
        f"- Output directory: `{output_root}`",
        "- Spatial convention: `x_GT = R_GT_sensor @ x_sensor + p_GT_sensor`; the estimator's `body` frame is GT.",
        "- `R_GT_sensor` maps sensor axes into GT; `p_GT_sensor` runs from the GT origin to the sensor origin and is expressed in GT.",
        "- Time convention: `t_GT = t_sensor - tau`.",
        "",
        "## Stage Results",
        "",
        "| Stage | Type | RMS | tau (s) | tau after epoch alignment (s) | p_GT_sensor (m) | Euler xyz sensor -> GT (deg) | Scale |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    loaded = []
    for record in stage_records:
        result = _load_result(record)
        if result is None:
            continue
        fields = _result_fields(result)
        if fields is None:
            continue
        loaded.append((record, result, fields))
        lines.append(
            "| {stage} | {typ} | {rms} | {tau} | {small_tau} | `{trans}` | `{rot}` | {scale} |".format(
                stage=record.get("name", ""),
                typ=fields["sensor"],
                rms=_fmt_num(result.get("rms"), 5),
                tau=_fmt_num(fields["time"], 8),
                small_tau=_fmt_num(fields["small_time"], 5),
                trans=_fmt_vec(fields["translation"]),
                rot=_fmt_vec(fields["rotation"]),
                scale=_fmt_num(fields["scale"], 5) if fields["scale"] is not None else "n/a",
            )
        )

    cad = _cad_results()
    cad_rows = []
    for record, _, fields in loaded:
        reference = cad.get(record.get("name"))
        if reference is None:
            continue
        euler = Rotation.from_matrix(reference["R_GT_sensor"]).as_euler("xyz", degrees=True)
        cad_rows.append(
            f"| {record['name']} | `{_fmt_vec(reference['p_GT_sensor_m'])}` | `{_fmt_vec(euler)}` |"
        )
    if cad_rows:
        lines += [
            "",
            "### CAD Reference",
            "",
            "CAD and stage extrinsics both map sensor coordinates into GT. "
            "Source: `baseline/src/CAD-Manual/CAD.txt` (structured record: "
            "`experiments/comb3/sensor_to_gt.json`). CAD provides no time offset, "
            "velocity scale, or fitting RMS.",
            "",
            "| Stage | p_GT_sensor (m) | Euler xyz sensor -> GT (deg) |",
            "|---|---|---|",
            *cad_rows,
        ]

    lines += [
        "",
        "## Data Quality and Motion Excitation",
        "",
        "These diagnostics indicate whether the recorded motion can support translation and time estimation. For radar translation, the important signal is the lever-arm velocity $\\|\\omega \\times p\\|$; if it is small compared with the residual, translation is numerically fragile even when the Jacobian is full rank.",
        "",
        "| Stage | omega p95 (rad/s) | lever median (m/s) | lever p95 (m/s) | trans. cond. | time corr. | RMS/lever p95 | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record, result, fields in loaded:
        if fields["sensor"] != "radar":
            continue
        q = _radar_quality(result)
        diagnosis = f"rank {q['rank_label']}, lever {q['lever_label']}, time {q['corr_label']}"
        lines.append(
            "| {stage} | {omega} | {lever_med} | {lever_p95} | {cond} | {corr} | {ratio} | {diag} |".format(
                stage=record.get("name", ""),
                omega=_fmt_num(q["omega_p95"], 5),
                lever_med=_fmt_num(q["lever_median"], 5),
                lever_p95=_fmt_num(q["lever_p95"], 5),
                cond=_fmt_num(q["condition"], 5),
                corr=_fmt_num(q["corr"], 5),
                ratio=_fmt_num(q["rms_to_lever_p95"], 5),
                diag=diagnosis,
            )
        )
    lines += [
        "",
        "Practical reading: `rank good` means the angular directions are not degenerate. `lever weak/poor` means the angular excitation or actual lever arm does not create enough velocity difference for robust translation estimation. A large `RMS/lever p95` means the translation signal is close to the noise/residual floor.",
        "",
        "## SE(3) Extrinsics",
        "",
        "Each matrix maps sensor coordinates into GT; its translation runs from the GT origin to the sensor origin, in GT coordinates. Source JSON calls GT `body`:",
        "",
        "\\[",
        "{}^{\\mathcal{GT}}\\mathbf T_{\\mathcal S} =",
        "\\begin{bmatrix}",
        "{}^{\\mathcal{GT}}\\mathbf R_{\\mathcal S} & {}^{\\mathcal{GT}}\\mathbf p_{\\mathcal S} \\\\",
        "\\mathbf 0^\\top & 1",
        "\\end{bmatrix}.",
        "\\]",
        "",
    ]
    for record, result, fields in loaded:
        lines += [
            f"### {record.get('name', '')}: `{fields['se3_name'].replace('T_body_', 'T_GT_')}`",
            "",
            "```text",
            _fmt_matrix(fields["se3"]),
            "```",
            "",
        ]

    lines += [
        "## Figures To Inspect",
        "",
        "### Session Overview",
        "",
        "- `paper_figures/spatial_layout_3d.png`: estimated 3-D sensor placement.",
        "- `paper_figures/spatiotemporal_overview.png`: spatial layout and relative timing overview.",
        "",
    ]
    joint_records = [r for r in stage_records if r.get("type") == "joint_refine"]
    if joint_records:
        lines += ["### Joint Refinement", ""]
        for record in joint_records:
            name = record.get("name", "")
            lines.append(f"- `{name}/joint_summary.md`: before/after residuals and refined SE(3).")
            lines.append(f"- `{name}/joint_result.json`: machine-readable joint refinement result.")
            lines.append(f"- `{name}/joint_block_metrics.csv`: compact per-block before/after metrics.")
        lines.append("")
    for record, _, fields in loaded:
        lines += [f"### {record.get('name', '')}", ""]
        out_dir = Path(record.get("output_dir", ""))
        for fig in fields["figures"]:
            if (out_dir / fig).exists():
                rel = (Path(record.get("name", "")) / fig).as_posix()
                lines.append(f"- `{rel}`")
        lines.append("")

    lines += [
        "## CSV Files",
        "",
        "- Radar stages: `radar_aligned_debug.csv` contains measured body-frame velocity, predicted body-frame velocity, and xyz residuals.",
        "- IMU stages: `imu_gyro_residuals.csv` contains xyz gyroscope residuals.",
        "- Event stages: `event_residuals.csv` contains scalar normal-flow residuals; event observations are not direct xyz velocity measurements.",
        "",
    ]
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_session_overview(output_root: Path, stage_records):
    fig_dir = output_root / "paper_figures"
    ensure_dir(fig_dir)
    sensors = []
    for record in stage_records:
        result = _load_result(record)
        pose = _sensor_pose(result)
        if pose is not None:
            sensors.append((record.get("name", pose[0]), result, *pose))
    if not sensors:
        write_session_summary(output_root, stage_records)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.scatter([0], [0], s=90, color="black", label="Body B", zorder=3)
    ax.text(0.03, 0.03, "Body B", fontsize=9, weight="bold")
    colors = {"Radar": "C0", "Event": "C3", "IMU": "C2"}
    for stage_name, result, sensor_type, p, _ in sensors:
        c = colors.get(sensor_type, "0.35")
        ax.arrow(0, 0, p[0], p[1], head_width=0.025, length_includes_head=True, color=c, alpha=0.9)
        ax.scatter([p[0]], [p[1]], s=70, color=c, zorder=4)
        ax.text(p[0] + 0.025, p[1] + 0.025, f"{stage_name}\n({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) m", fontsize=8)
    ax.set_title("Estimated Spatial Layout")
    ax.set_xlabel("body x (m)")
    ax.set_ylabel("body y (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    ax = axes[1]
    labels = []
    vals = []
    rms = []
    for stage_name, result, sensor_type, _, _ in sensors:
        dt = _small_time_offset(result)
        if dt is None:
            continue
        labels.append(stage_name)
        vals.append(dt)
        rms.append(result.get("rms", np.nan))
    y = np.arange(len(labels))
    ax.axvline(0.0, color="black", lw=0.8)
    ax.barh(y, vals, color="0.30", alpha=0.8)
    for yi, val, r in zip(y, vals, rms):
        ax.text(val, yi, f" {val:.3f}s, RMS {r:.3g}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("sensor-to-body offset after epoch alignment (s)")
    ax.set_title("Estimated Temporal Offsets")
    ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Velocity-Domain Spatio-Temporal Calibration Overview", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "spatiotemporal_overview.png", dpi=260)
    fig.savefig(fig_dir / "spatiotemporal_overview.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], s=90, color="black", label="Body B", depthshade=False)
    ax.text(0.02, 0.02, 0.02, "Body B", fontsize=9, weight="bold")
    axis_len = 0.25
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", linewidth=1.5)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", linewidth=1.5)
    ax.text(axis_len, 0, 0, "x_B", color="r", fontsize=8)
    ax.text(0, axis_len, 0, "y_B", color="g", fontsize=8)
    ax.text(0, 0, axis_len, "z_B", color="b", fontsize=8)

    points = [np.zeros(3)]
    for stage_name, result, sensor_type, p, _ in sensors:
        c = colors.get(sensor_type, "0.35")
        points.append(p)
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=c, linewidth=1.4, alpha=0.85)
        ax.scatter([p[0]], [p[1]], [p[2]], s=80, color=c, depthshade=False)
        ax.text(p[0], p[1], p[2], f" {stage_name}\n [{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}] m", fontsize=8)
        if sensor_type in {"Radar", "Event"}:
            size = 0.055
            verts = [
                [
                    (p[0] - size, p[1] - size, p[2]),
                    (p[0] + size, p[1] - size, p[2]),
                    (p[0] + size, p[1] + size, p[2]),
                    (p[0] - size, p[1] + size, p[2]),
                ]
            ]
            poly = Poly3DCollection(verts, alpha=0.16, facecolor=c, edgecolor=c)
            ax.add_collection3d(poly)

    pts = np.vstack(points)
    center = np.mean(pts, axis=0)
    span = max(float(np.max(np.ptp(pts, axis=0))), 0.5)
    half = span * 0.65
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("body x (m)")
    ax.set_ylabel("body y (m)")
    ax.set_zlabel("body z (m)")
    ax.set_title("3D Estimated Sensor Layout")
    ax.view_init(elev=24, azim=-54)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "spatial_layout_3d.png", dpi=260)
    fig.savefig(fig_dir / "spatial_layout_3d.pdf")
    plt.close(fig)
    write_session_summary(output_root, stage_records)

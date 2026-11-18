from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import apply_overrides, load_yaml, save_yaml
from .io import ensure_dir, save_json
from .event import run_event_reference_stage
from .imu import run_imu_reference_stage
from .joint import run_joint_manifest_stage, run_joint_refine_stage
from .radar import run_radar_reference_stage
from .visualization import make_session_overview


DEFAULT_DATA_FILES = {
    "reference.twist": "gt.twist",
    "radar_main.inputs.radar_velocity": "radar_vel.tum",
    "radar_main.inputs.radar_debug": "details/radar_lsq_debug_main.csv",
    "aid_radar_main.inputs.radar_velocity": "aid_radar_vel.tum",
    "aid_radar_main.inputs.radar_debug": "details/radar_lsq_debug_aid.csv",
    "event_main.inputs.event_flow": "debug_flows.csv",
    "livox_imu_main.inputs.imu": "livox_imu.csv",
    "dvs_imu_main.inputs.imu": "dvs_imu.csv",
}


def apply_data_dir(config: Dict, data_dir: Optional[Path]) -> Dict:
    if data_dir is None:
        return config
    data_dir = data_dir.expanduser().resolve()
    out = json.loads(json.dumps(config))
    out.setdefault("reference", {})["twist"] = str(data_dir / DEFAULT_DATA_FILES["reference.twist"])
    for stage in out.get("stages", []):
        name = stage.get("name")
        if name == "radar_main":
            stage.setdefault("inputs", {})["radar_velocity"] = str(data_dir / DEFAULT_DATA_FILES["radar_main.inputs.radar_velocity"])
            debug_path = data_dir / DEFAULT_DATA_FILES["radar_main.inputs.radar_debug"]
            if debug_path.exists():
                stage.setdefault("inputs", {})["radar_debug"] = str(debug_path)
        elif name == "aid_radar_main":
            stage.setdefault("inputs", {})["radar_velocity"] = str(data_dir / DEFAULT_DATA_FILES["aid_radar_main.inputs.radar_velocity"])
            debug_path = data_dir / DEFAULT_DATA_FILES["aid_radar_main.inputs.radar_debug"]
            if debug_path.exists():
                stage.setdefault("inputs", {})["radar_debug"] = str(debug_path)
        elif name == "event_main":
            stage.setdefault("inputs", {})["event_flow"] = str(data_dir / DEFAULT_DATA_FILES["event_main.inputs.event_flow"])
        elif name == "livox_imu_main":
            stage.setdefault("inputs", {})["imu"] = str(data_dir / DEFAULT_DATA_FILES["livox_imu_main.inputs.imu"])
        elif name == "dvs_imu_main":
            stage.setdefault("inputs", {})["imu"] = str(data_dir / DEFAULT_DATA_FILES["dvs_imu_main.inputs.imu"])
    return out


def apply_output(config: Dict, output_dir: Optional[Path]) -> Dict:
    if output_dir is None:
        return config
    out = json.loads(json.dumps(config))
    out.setdefault("project", {})["output_dir"] = str(output_dir.expanduser())
    return out


def run_pipeline(config: Dict, config_path: Path, only_stage: Optional[str] = None) -> Dict:
    config_dir = config_path.parent.resolve()
    project_cfg = config.get("project", {})
    output_root = Path(project_cfg.get("output_dir", "experiments/velocity_calib")).expanduser()
    if not output_root.is_absolute():
        output_root = (config_dir / output_root).resolve()
    ensure_dir(output_root)

    copied_config = output_root / "resolved_config.yaml"
    save_yaml(copied_config, config)

    reference_cfg = config.get("reference", {})
    if "twist" not in reference_cfg:
        raise ValueError("Config requires reference.twist")

    stage_records: List[Dict] = []
    for stage_cfg in config.get("stages", []):
        if not stage_cfg.get("enabled", True):
            continue
        name = stage_cfg.get("name")
        if only_stage and name != only_stage:
            continue
        stage_type = stage_cfg.get("type")
        if stage_type == "radar_reference":
            result = run_radar_reference_stage(stage_cfg, reference_cfg, output_root, config_dir)
            stage_records.append({
                "name": result.name,
                "type": stage_type,
                "output_dir": str(result.output_dir),
                "result_json": str(result.result_json),
                "data_npz": str(result.data_npz),
                "rms": result.payload.get("rms"),
                "time_offset_sec": result.payload.get("time_offset_radar_minus_body_sec"),
            })
        elif stage_type == "event_reference":
            result = run_event_reference_stage(stage_cfg, reference_cfg, output_root, config_dir)
            stage_records.append({
                "name": result.name,
                "type": stage_type,
                "output_dir": str(result.output_dir),
                "result_json": str(result.result_json),
                "data_npz": str(result.data_npz),
                "rms": result.payload.get("rms"),
                "time_offset_sec": result.payload.get("time_offset_event_minus_body_sec"),
            })
        elif stage_type == "imu_reference":
            result = run_imu_reference_stage(stage_cfg, reference_cfg, output_root, config_dir)
            stage_records.append({
                "name": result.name,
                "type": stage_type,
                "output_dir": str(result.output_dir),
                "result_json": str(result.result_json),
                "data_npz": str(result.data_npz),
                "rms": result.payload.get("rms"),
                "time_offset_sec": result.payload.get("time_offset_imu_minus_body_sec"),
            })
        elif stage_type == "joint_refine":
            result = run_joint_refine_stage(stage_cfg, reference_cfg, stage_records, output_root, config_dir)
            stage_records.append({
                "name": result.name,
                "type": result.payload.get("type"),
                "output_dir": str(result.output_dir),
                "result_json": str(result.result_json),
                "data_npz": str(result.data_npz) if result.data_npz else None,
                "status": result.payload.get("status"),
                "initial_joint_rms": result.payload.get("initial_joint_rms"),
                "final_joint_rms": result.payload.get("final_joint_rms"),
            })
        elif stage_type in {"joint", "joint_manifest"}:
            result = run_joint_manifest_stage(stage_cfg, stage_records, output_root)
            stage_records.append({
                "name": result.name,
                "type": result.payload.get("type"),
                "output_dir": str(result.output_dir),
                "result_json": str(result.result_json),
                "data_npz": None,
                "status": result.payload.get("status"),
            })
        else:
            raise ValueError(f"Unknown or unsupported stage type: {stage_type}")

    manifest = {
        "project": project_cfg.get("name", "velocity_calibration"),
        "config": str(copied_config),
        "output_dir": str(output_root),
        "stages": stage_records,
    }
    save_json(output_root / "calibration_session.json", manifest)
    make_session_overview(output_root, stage_records)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Modular velocity-domain calibration runner.")
    parser.add_argument("--config", required=True, type=Path, help="YAML calibration config.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing gt.twist, radar_vel.tum, aid_radar_vel.tum, debug_flows.csv, livox_imu.csv, and dvs_imu.csv.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory for this run.")
    parser.add_argument("--stage", default=None, help="Run only one stage name.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set project.output_dir=experiments/run1",
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = apply_overrides(load_yaml(args.config), args.overrides)
    config = apply_data_dir(config, args.data_dir)
    config = apply_output(config, args.output)
    manifest = run_pipeline(config, args.config.resolve(), only_stage=args.stage)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

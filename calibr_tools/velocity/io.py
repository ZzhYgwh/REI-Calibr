from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_table(path: Path, delimiter: Optional[str] = None) -> np.ndarray:
    path = Path(path)
    if delimiter is None and path.suffix.lower() == ".csv":
        delimiter = ","
    attempts = []
    for skip_header in range(0, 6):
        attempts.extend([
            (delimiter, skip_header),
            (None, skip_header),
            (",", skip_header),
        ])
    last_error = None
    data = None
    for delim, skip_header in attempts:
        try:
            data = np.loadtxt(path, delimiter=delim, skiprows=skip_header)
            break
        except ValueError as exc:
            last_error = exc
    if data is None:
        raise ValueError(f"Failed to load numeric table {path}: {last_error}") from last_error
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[np.isfinite(data).all(axis=1)]
    # return data[np.argsort(data[:, 0])]
    # 新
    order = np.argsort(data[:, 0], kind="stable")
    return data[order]


def parse_cols(cols: str) -> Tuple[int, int]:
    parts = [int(c) for c in str(cols).split(":")]
    if len(parts) != 2:
        raise ValueError(f"Column slice must be like '1:4', got: {cols}")
    return parts[0], parts[1]


def load_reference_twist(path: Path, delimiter: Optional[str] = None):
    data = load_table(path, delimiter=delimiter)
    if data.shape[1] < 7:
        raise ValueError(f"Reference twist must have columns t,vx,vy,vz,wx,wy,wz: {path}")
    return data[:, 0], data[:, 1:4], data[:, 4:7]


def load_odometry_pose(path: Path, delimiter: Optional[str] = None):
    data = load_table(path, delimiter=delimiter)
    if data.shape[1] < 8:
        raise ValueError(f"Odometry pose must have columns t,px,py,pz,qx,qy,qz,qw: {path}")
    return data[:, 0], data[:, 4:8]


def load_vector_series(path: Path, cols="auto", delimiter: Optional[str] = None):
    data = load_table(path, delimiter=delimiter)
    if data.shape[1] < 4:
        raise ValueError(f"Vector series must have timestamp plus three values: {path}")
    if cols == "auto":
        if data.shape[1] >= 7:
            n1 = np.nanmedian(np.linalg.norm(data[:, 1:4], axis=1))
            n2 = np.nanmedian(np.linalg.norm(data[:, 4:7], axis=1))
            start, end = (4, 7) if n2 > 3.0 * max(n1, 1e-12) else (1, 4)
        else:
            start, end = 1, 4
    else:
        start, end = parse_cols(cols)
    return data[:, 0], data[:, start:end]


def _ensure_dim3(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 3:
        return values
    if values.shape[1] == 2:
        return np.column_stack([values, np.zeros(len(values))])
    raise ValueError(f"Expected 2D or 3D vectors, got shape {values.shape}")


def load_event_flow(
    path: Path,
    x_cols="1:4",
    normal_cols="4:6",
    normal_speed_col=None,
    alpha_col=None,
    flow_mode="auto",
    delimiter: Optional[str] = None,
):
    data = load_table(path, delimiter=delimiter)
    xs, xe = parse_cols(x_cols)
    ns, ne = parse_cols(normal_cols)
    x = _ensure_dim3(data[:, xs:xe])
    normal = _ensure_dim3(data[:, ns:ne])
    if flow_mode == "vector":
        flow = normal
    elif normal_speed_col is not None:
        flow = normal * data[:, int(normal_speed_col):int(normal_speed_col) + 1]
    elif alpha_col is not None:
        flow = normal * data[:, int(alpha_col):int(alpha_col) + 1]
    else:
        flow = normal
    scalar = None
    if normal_speed_col is not None:
        scalar = data[:, int(normal_speed_col)]
    elif alpha_col is not None:
        scalar = data[:, int(alpha_col)]
    return data[:, 0], x, normal, flow, scalar


def load_imu(path: Path, gyro_cols="auto", accel_cols=None, delimiter: Optional[str] = None):
    data = load_table(path, delimiter=delimiter)
    if gyro_cols == "auto":
        if data.shape[1] >= 7:
            accel = data[:, 1:4]
            gyro = data[:, 4:7]
        elif data.shape[1] >= 4:
            accel = None
            gyro = data[:, 1:4]
        else:
            raise ValueError(f"IMU file has too few columns: {path}")
    else:
        gs, ge = parse_cols(gyro_cols)
        gyro = data[:, gs:ge]
        accel = None
    if accel_cols:
        accs, acce = parse_cols(accel_cols)
        accel = data[:, accs:acce]
    return data[:, 0], accel, gyro


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_se3(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform.tolist()

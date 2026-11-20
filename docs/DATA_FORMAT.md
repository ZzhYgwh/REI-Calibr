# Input Data Format

## Data releases

- **Raw recordings:** [Google Drive](https://drive.google.com/drive/folders/11vWLF8X6Cd7w-J2NAtEPvszvG5DYo0Hy?usp=drive_link)
- **Frontend-processed calibration inputs:** [Google Drive](https://drive.google.com/drive/folders/1pBN6QWuz3XEVE6UZTKoemDToQXk3U_Nt?usp=drive_link)

The raw recordings are original sensor logs and are not stored in this repository. The calibration-input bundle contains the timestamped files expected by `configs/full_pipeline.yaml`. To regenerate frontend observations from raw logs, use the [REI-Calibr-Frontend](https://github.com/ZzhYgwh/REI-Calibr-Frontend) submodule and its `launch/out.launch`.

The pipeline expects one directory per recording. A typical directory looks like this:

```text
data/sequence/
├── gt.twist
├── radar_vel.tum
├── aid_radar_vel.tum        # optional second radar
├── debug_flows.csv          # event-frontend normal-flow observations
├── livox_imu.csv
├── dvs_imu.csv
└── details/
    ├── radar_lsq_debug_main.csv
    └── radar_lsq_debug_aid.csv
```

Blank lines and comment/header rows are accepted by the table loader. Timestamps are expressed in seconds.

## Reference twist

`gt.twist` supplies the reference body twist:

```text
t vx vy vz wx wy wz
```

- `vx, vy, vz`: linear velocity in the reference/body frame, in m/s.
- `wx, wy, wz`: angular velocity in the reference/body frame, in rad/s.

## Radar velocity

`radar_vel.tum` and `aid_radar_vel.tum` contain timestamped velocity observations in the radar frame:

```text
t vx vy vz
```

Set `radar_cols` when the three velocity columns are not indices `1:4`. The `.tum` suffix is retained for compatibility with the experimental tooling; it is not a pose file in this project.

The optional `details/radar_lsq_debug_*.csv` files feed the radar quality filter. When they are absent, the corresponding quality mask is disabled and the other preprocessing options still apply.

## Event normal flow

Event observations are produced by the event frontend and stored in a CSV table. The `debug_flows.csv` written by the bundled frontend has this layout:

```text
process_time,inlier_idx,pixel_x,pixel_y,pixel_z,
norm_in_pixel_x,norm_in_pixel_y,norm_in_pixel_z,
grad_x,grad_y,grad_z,normal_norm
```

The exact meaning of each column is configurable through:

```yaml
x_cols: "2:5"
normal_cols: "8:11"
flow_mode: auto
normal_speed_col: "11"
alpha_col: null
```

- `x_cols` identifies the normalized event coordinates.
- `normal_cols` identifies the normal-flow direction or vector columns.
- `normal_speed_col` or `alpha_col` scales the normal to obtain the flow.
- If the flow vector is already stored directly, set `flow_mode: vector` and leave the speed columns as `null`.

The bundled `configs/full_pipeline.yaml` matches this export: event coordinates are columns `2:5`, the flow vector is `8:11`, and the flow magnitude is column `11`. Adapt these four fields if your frontend uses a different layout.

## IMU

IMU files contain accelerometer and gyroscope measurements:

```text
t ax ay az gx gy gz
```

- Accelerometer values are in m/s².
- Gyroscope values are in rad/s.

The default column selection is `accel_cols: "1:4"` and `gyro_cols: "4:7"`. A gyro-only file such as `t gx gy gz` is also supported.

The calibration-input release contains the synchronized IMU exports used in the reported experiment (`livox_imu.csv` and `dvs_imu.csv`). The ROS frontend writes IMU-dependent motion estimates and event/radar diagnostics; the dataset bundle should be treated as the canonical source when exact reproduction of the paper inputs matters.

## Outputs

Each stage writes a `stage_summary.md` file and a machine-readable result. The main artifacts are:

| File | Description |
| --- | --- |
| `calibration_session.json` | Manifest of stages, run configuration, and generated artifacts |
| `resolved_config.yaml` | Fully resolved configuration used for the run |
| `summary.md` | Human-readable spatial and temporal calibration summary |
| `<stage>/<stage>_result.json` | Extrinsics, timing, scale/bias, residuals, and covariance data |
| `<stage>/stage_summary.md` | Per-stage summary |
| `<stage>/*_residuals.csv` | Residuals for diagnostics and plotting |
| `<stage>/*_debug.csv` | Synchronized observations and predictions |
| `<stage>/*.png` | Alignment, residual, and optimizer diagnostics |

The `data/` and `outputs/` directories are ignored by Git so that recordings and process artifacts are not published.

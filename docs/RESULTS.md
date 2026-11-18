# Representative Calibration Result

This page keeps only a compact, text-based result snapshot. Raw recordings, intermediate CSV files, plots, and the manuscript are intentionally not part of this repository.

## Provenance

- Source: the final five-stage configuration used for the internal representative calibration run dated 2026-11-13.
- Run type: one representative sequence, one optimization run. The values below are not a multi-sequence benchmark.
- Spatial convention: `x_GT = R_GT_sensor @ x_sensor + p_GT_sensor`.
- Time convention: `t_GT = t_sensor - tau`.
- Residual RMS values are reported in the native unit of each stage and should be compared within a modality, not across modalities.

## Stage estimates

| Stage | Modality | Residual RMS | `tau` (s) | `p_GT_sensor` (m) | Euler xyz sensor-to-GT (deg) |
| --- | --- | ---: | ---: | --- | --- |
| `radar_main` | Radar | 0.1511 | -0.532916 | `[0.135189, -0.0310358, 0.037504]` | `[3.26007, 4.84847, 2.81825]` |
| `aid_radar_main` | Auxiliary radar | 0.12071 | -0.534487 | `[0.135417, -0.0372424, 0.075781]` | `[-87.1734, 3.75463, 0.944139]` |
| `event_main` | Event camera | 0.11232 | -0.562357 | `[0.105259, 0.00630348, 0.0155143]` | `[-92.1411, 3.11444, -89.0439]` |
| `livox_imu_main` | Livox IMU | 0.063022 | -0.005838 | `[0.0819858, 0.00821753, 0.105339]` | `[3.0046, 26.1563, 2.9992]` |
| `dvs_imu_main` | DVS IMU | 0.060671 | -0.768396 | `[0.0918803, 0.00665832, 0.0244585]` | `[-95.1746, 3.53366, -87.2211]` |

## CAD reference

The CAD values are useful as a sanity check but are not treated as ground truth by the optimizer.

| Stage | `p_GT_sensor` (m) | Euler xyz sensor-to-GT (deg) |
| --- | --- | --- |
| `radar_main` | `[0.0865, -0.0418, 0.0766]` | `[0, 0, 0]` |
| `aid_radar_main` | `[0.0865, -0.0233, 0.11318]` | `[-90, 0, 0]` |
| `event_main` | `[0.0569, 0.01265, 0.05068]` | `[-90, 0, -90]` |
| `livox_imu_main` | `[0.0197342, 0.01609, 0.112643]` | `[0, 24.8579, 0]` |
| `dvs_imu_main` | `[0.0569, 0.01265, 0.05068]` | `[-90, 0, -90]` |

## Joint refinement

The optional joint refinement reduced the aggregate residual RMS from `0.0761444` to `0.0759848`. The optimizer reached its evaluation limit (`max_nfev`), so the result is reported as a small refinement rather than a fully converged joint solution.

## Motion diagnostics

For this sequence, the radar stages had sufficient angular excitation for the lever-arm term:

| Stage | `omega` p95 (rad/s) | Lever-arm velocity p95 (m/s) | Translation condition number | Time correlation |
| --- | ---: | ---: | ---: | ---: |
| `radar_main` | 2.9304 | 0.2393 | 1.748 | 0.909 |
| `aid_radar_main` | 2.9167 | 0.3087 | 1.512 | 0.927 |

These diagnostics matter: without enough angular motion, the translation and time offset can be numerically fragile even when the optimizer reports a solution.

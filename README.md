<p align="center">
  <img src="assets/banner.png" alt="REI-Calibr: Radar–Event–IMU velocity-domain calibration" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ZzhYgwh/REI-Calibr/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ZzhYgwh/REI-Calibr/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-2EA043?style=flat-square"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B?style=flat-square&logo=arxiv&logoColor=white"></a>
  <a href="#video"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-coming%20soon-FF0000?style=flat-square&logo=youtube&logoColor=white"></a>
</p>

# REI-Calibr

**REI-Calibr** is the execution code for velocity-domain spatiotemporal calibration of a **radar**, an **event camera**, and an **IMU** against a fixed reference twist. It estimates each sensor's rotation, translation, time offset, and modality-specific nuisance parameters without requiring sensor-to-sensor feature associations.

This repository is intentionally lightweight: it contains the runnable pipeline, the final reference configurations, and a compact result summary. Raw recordings, generated experiment folders, intermediate files, and the manuscript are **not** included.

## Highlights

- Rigid-body velocity residual with the lever-arm term: `R_BR v_R = s (v_B + ω_B × p_BR)`.
- Independent radar, event, and IMU stages followed by an optional joint refinement.
- Coarse-to-fine time-offset estimation and robust Huber/soft-L1 optimization.
- Reusable YAML configuration with CLI overrides and one-stage execution.
- Reproducible manifests (`calibration_session.json`, `resolved_config.yaml`) and per-stage diagnostics.

## Method

For a sensor `S ∈ {R, E, I}` rigidly attached to the calibration body `B`, the pipeline relates the sensor-frame velocity observation to the reference body twist:

```text
v_S = s_S · R_BRᵀ (v_B + ω_B × p_BS) + b_S
t_B = t_S - tau_S
```

The unknowns are estimated from angular excitation in the reference trajectory:

- `R_BS`, `p_BS`: spatial extrinsic parameters.
- `tau_S`: sensor-minus-body time offset.
- `s_S`: optional Doppler scale.
- `b_S`: optional radar bias or IMU gyro bias.

The radar, event, and IMU stages share the same reference twist but keep modality-specific residuals. See `docs/RESULTS.md` for the spatial/time conventions used in the reported output.

## Repository Layout

```text
.
├── assets/                 # Project banner and logo
├── calibr_tools/velocity/  # Modular calibration pipeline
├── configs/                # Sanitized final configurations
├── docs/                   # Input format and compact results
├── scripts/                # Synthetic example, single-sequence/batch runners, aggregation
├── CITATION.cff            # Citation metadata
├── LICENSE                 # MIT
├── pyproject.toml          # Package metadata
└── requirements.txt        # Runtime dependencies
```

## Installation

The code requires Python 3.8 or newer and works on CPU only.

```bash
git clone https://github.com/ZzhYgwh/REI-Calibr.git
cd REI-Calibr

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional editable install:

```bash
python3 -m pip install -e .
```

Core dependencies: `numpy`, `scipy`, `PyYAML`, and `matplotlib`.

## Data

Place one recording in `data/sequence/`. The minimum radar-only setup is:

```text
data/sequence/
├── gt.twist          # reference twist
└── radar_vel.tum     # radar-frame velocity
```

The full setup also uses `aid_radar_vel.tum`, `debug_flows.csv`, `livox_imu.csv`, and `dvs_imu.csv`. See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the column conventions and optional radar diagnostics.

The recordings used in the paper are not bundled here. The paper's project page is [ZzhYgwh/TwistEstimator](https://github.com/ZzhYgwh/TwistEstimator); use the data release and format notes there if you are reproducing the complete five-sensor experiment.

## Running

### Radar-only quick start

No private recording is needed for a first smoke test. Generate a small synthetic radar sequence:

```bash
python3 scripts/make_synthetic_example.py --output data/example
```

Then run:

```bash
python3 -m calibr_tools.velocity.pipeline \
  --config configs/radar_only.yaml \
  --data-dir data/example \
  --output outputs/example
```

Or use the wrapper:

```bash
./scripts/run_sequence.sh configs/radar_only.yaml data/example outputs/example
```

### Full radar–event–IMU pipeline

```bash
python3 -m calibr_tools.velocity.pipeline \
  --config configs/full_pipeline.yaml \
  --data-dir data/sequence \
  --output outputs/full_pipeline
```

### Run a single stage

```bash
python3 -m calibr_tools.velocity.pipeline \
  --config configs/full_pipeline.yaml \
  --data-dir data/sequence \
  --output outputs/radar_only \
  --stage radar_main
```

### Batch processing

```bash
./scripts/run_batch.sh data configs/full_pipeline.yaml outputs
```

Each subdirectory of `data/` that contains `gt.twist` is treated as one sequence.

### Override configuration without editing YAML

```bash
python3 -m calibr_tools.velocity.pipeline \
  --config configs/radar_only.yaml \
  --data-dir data/sequence \
  --output outputs/sequence \
  --set stages.0.optimization.scale_mode=global \
  --set stages.0.time.max_lag_sec=1.0 \
  --set stages.0.optimization.max_nfev=500
```

## Configuration

- `configs/radar_only.yaml`: minimal single-radar calibration and the recommended starting point.
- `configs/full_pipeline.yaml`: final five-stage configuration plus joint refinement, using neutral `data/sequence/` placeholders.

Important configuration groups:

| Group | Purpose |
| --- | --- |
| `reference` | Reference twist file (`t vx vy vz wx wy wz`). |
| `inputs` | Sensor files and column slices. |
| `preprocessing` | Edge trimming, low-pass filtering, radar quality masks, and outlier rejection. |
| `time` | Coarse alignment, search range, and optional initial offset. |
| `optimization` | Robust loss, residual scale, iteration limit, scale mode, and parameter bounds. |
| `diagnostics` | Plot and summary generation. |

## Outputs

The main output directory contains:

```text
outputs/<run>/
├── calibration_session.json
├── resolved_config.yaml
├── summary.md
└── <stage>/
    ├── <stage>_result.json
    ├── stage_summary.md
    ├── *_residuals.csv
    ├── *_debug.csv
    └── *.png
```

The result JSON records the estimated sensor-to-body transform, time offset, scale/bias, residual statistics, and the frame convention. `summary.md` is the compact human-readable report.

To combine several completed run roots:

```bash
python3 scripts/aggregate_results.py outputs/run_a outputs/run_b \
  --continue-seq \
  -o outputs/aggregate.md
```

## Representative Result

The following single-sequence snapshot is included for orientation only; raw data and intermediate files are not distributed. Full details and the CAD sanity check are in [`docs/RESULTS.md`](docs/RESULTS.md).

| Stage | Residual RMS | `tau` (s) | `p_GT_sensor` (m) |
| --- | ---: | ---: | --- |
| `radar_main` | 0.1511 | -0.532916 | `[0.135189, -0.0310358, 0.037504]` |
| `aid_radar_main` | 0.12071 | -0.534487 | `[0.135417, -0.0372424, 0.075781]` |
| `event_main` | 0.11232 | -0.562357 | `[0.105259, 0.00630348, 0.0155143]` |
| `livox_imu_main` | 0.063022 | -0.005838 | `[0.0819858, 0.00821753, 0.105339]` |
| `dvs_imu_main` | 0.060671 | -0.768396 | `[0.0918803, 0.00665832, 0.0244585]` |

Residual RMS is in the native unit of each stage. Compare values within a modality rather than across modalities.

## Video

<a id="video"></a>

YouTube: _coming soon_. Add the final illustrative video URL here and in the badge at the top of this file.

## Citation

If you use this code, please cite the accompanying paper. The arXiv identifier is a placeholder until the preprint record is assigned.

**arXiv reference format**

> Y. Lyu, Z. Zou, Y. Li, X. Guo, C. Zhao, and Q. Pan, “Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation,” arXiv preprint arXiv:XXXX.XXXXX, 2026.

**BibTeX for arXiv**

```bibtex
@misc{lyu2026radareventarxiv,
  author        = {Lyu, Yang and Zou, Zhenghao and Li, Yanfeng and Guo, Xiaohu and Zhao, Chunhui and Pan, Quan},
  title         = {Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {Preprint}
}
```

**Published version**

```bibtex
@article{lyu2026radarevent,
  author  = {Lyu, Yang and Zou, Zhenghao and Li, Yanfeng and Guo, Xiaohu and Zhao, Chunhui and Pan, Quan},
  title   = {Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation},
  journal = {IEEE/ASME Transactions on Mechatronics},
  year    = {2026},
  doi     = {10.1109/TMECH.2026.3662880}
}
```

Paper DOI: [10.1109/TMECH.2026.3662880](https://doi.org/10.1109/TMECH.2026.3662880). Update the arXiv link, arXiv reference, BibTeX `eprint`, and `CITATION.cff` when the preprint identifier is available.

## License

This project is released under the [MIT License](LICENSE).

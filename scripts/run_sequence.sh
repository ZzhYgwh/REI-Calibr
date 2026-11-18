#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/radar_only.yaml}"
DATA_DIR="${2:-data/sequence}"
OUTPUT_DIR="${3:-outputs/sequence}"

python3 -m calibr_tools.velocity.pipeline \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  "${@:4}"

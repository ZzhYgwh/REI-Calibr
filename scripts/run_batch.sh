#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${1:-data}"
CONFIG="${2:-configs/radar_only.yaml}"
OUTPUT_ROOT="${3:-outputs}"

for sequence_dir in "$DATA_ROOT"/*; do
  [ -d "$sequence_dir" ] || continue
  [ -f "$sequence_dir/gt.twist" ] || continue
  sequence_name="$(basename "$sequence_dir")"
  echo "==> $sequence_name"
  python3 -m calibr_tools.velocity.pipeline \
    --config "$CONFIG" \
    --data-dir "$sequence_dir" \
    --output "$OUTPUT_ROOT/$sequence_name"
done

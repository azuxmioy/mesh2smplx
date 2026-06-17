#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_starter.sh <config.yaml> <keypoints_3d.npy> [frame_indices]

Examples:
  scripts/run_starter.sh local_configs/my_sequence.yaml /path/to/my_sequence/keypoints_3d.npy 0
  MAX_STEPS_PER_STAGE= scripts/run_starter.sh local_configs/my_sequence.yaml /path/to/my_sequence/keypoints_3d.npy 0-30

Environment:
  PYTHON                 Python executable to use (default: python)
  MAX_STEPS_PER_STAGE    Smoke-test cap per fitting stage (default: 5; empty for full schedule)
  SCAN_SURFACE_SAMPLES   Mesh samples for scan target (default: 2000)
  BODY_VERTEX_SAMPLES    Body vertices sampled for mesh loss (default: 2000)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
CONFIG="$1"
KEYPOINTS3D="$2"
FRAME_INDICES="${3:-${FRAME_INDICES:-0}}"
MAX_STEPS_PER_STAGE="${MAX_STEPS_PER_STAGE-5}"
SCAN_SURFACE_SAMPLES="${SCAN_SURFACE_SAMPLES:-2000}"
BODY_VERTEX_SAMPLES="${BODY_VERTEX_SAMPLES:-2000}"

echo "Inspecting mesh sequence..."
"${PYTHON_BIN}" -m mesh2smplx inspect --config "${CONFIG}"

FIT_ARGS=(
  -m mesh2smplx fit-full
  --config "${CONFIG}"
  --keypoints3d "${KEYPOINTS3D}"
  --frame-indices "${FRAME_INDICES}"
  --scan-surface-samples "${SCAN_SURFACE_SAMPLES}"
  --body-vertex-samples "${BODY_VERTEX_SAMPLES}"
)

if [[ -n "${MAX_STEPS_PER_STAGE}" ]]; then
  FIT_ARGS+=(--max-steps-per-stage "${MAX_STEPS_PER_STAGE}")
fi

echo "Running fit for frame indices: ${FRAME_INDICES}"
"${PYTHON_BIN}" "${FIT_ARGS[@]}"

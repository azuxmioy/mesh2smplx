#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_gpu_full_pipeline.sh <config.yaml> [keypoints_3d.npy]

Examples:
  scripts/run_gpu_full_pipeline.sh configs/server_full_pipeline.yaml
  scripts/run_gpu_full_pipeline.sh configs/server_full_pipeline.yaml data/keypoints_3d.npy
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENTRYPOINT="${REPO_ROOT}/main.py"
CONFIG="$1"
KEYPOINTS3D="${2:-}"

FIT_ARGS=(
  "${ENTRYPOINT}"
  --config "${CONFIG}"
)

if [[ -n "${KEYPOINTS3D}" ]]; then
  if [[ "${KEYPOINTS3D}" != *.npy ]]; then
    echo "Expected optional keypoints path to end with .npy, got: ${KEYPOINTS3D}" >&2
    exit 2
  fi
  FIT_ARGS+=(--keypoints3d "${KEYPOINTS3D}")
fi

echo "Running full mesh2smplx pipeline for all meshes in sorted order"
"${PYTHON_BIN}" "${FIT_ARGS[@]}"

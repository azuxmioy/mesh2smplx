#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_local_debug_aitviewer.sh <config.yaml> [frame_indices]
  scripts/run_local_debug_aitviewer.sh <config.yaml> <keypoints_3d.npy> [frame_indices]

Examples:
  scripts/run_local_debug_aitviewer.sh configs/local_aitviewer.yaml
  scripts/run_local_debug_aitviewer.sh configs/local_aitviewer.yaml 0-5
  scripts/run_local_debug_aitviewer.sh configs/local_aitviewer.yaml data/keypoints_3d.npy 0
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
KEYPOINTS3D=""
FRAME_INDICES="0"

if [[ "${2:-}" == *.npy ]]; then
  KEYPOINTS3D="$2"
  FRAME_INDICES="${3:-${FRAME_INDICES}}"
elif [[ -n "${2:-}" ]]; then
  FRAME_INDICES="$2"
fi

FIT_ARGS=(
  "${ENTRYPOINT}"
  --config "${CONFIG}"
  --frame-indices "${FRAME_INDICES}"
  --aitviewer-launch
)

if [[ -n "${KEYPOINTS3D}" ]]; then
  FIT_ARGS+=(--keypoints3d "${KEYPOINTS3D}")
fi

echo "Running local AITviewer debug fit for frame indices: ${FRAME_INDICES}"
"${PYTHON_BIN}" "${FIT_ARGS[@]}"

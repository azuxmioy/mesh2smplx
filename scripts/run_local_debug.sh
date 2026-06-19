#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_local_debug.sh <config.yaml> [frame_indices] [--tracking true|false] [--betas body_shape.npy]
  scripts/run_local_debug.sh <config.yaml> <keypoints_3d.npy> [frame_indices] [--tracking true|false] [--betas body_shape.npy]

Examples:
  scripts/run_local_debug.sh configs/cpu.yaml
  scripts/run_local_debug.sh configs/cpu.yaml 0-5
  scripts/run_local_debug.sh configs/cpu.yaml data/keypoints_3d.npy 0
  scripts/run_local_debug.sh configs/cpu.yaml --tracking false --betas data/body_shape.npy
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="$1"
shift
KEYPOINTS3D=""
FRAME_INDICES="0"
FRAME_INDICES_SET=0

FIT_ARGS=(
  --config "${CONFIG}"
  --aitviewer-launch
)

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tracking)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --tracking; expected true or false." >&2
        exit 2
      fi
      FIT_ARGS+=(--tracking "$2")
      shift 2
      ;;
    --tracking=*)
      FIT_ARGS+=(--tracking "${1#*=}")
      shift
      ;;
    --betas)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --betas." >&2
        exit 2
      fi
      FIT_ARGS+=(--betas "$2")
      shift 2
      ;;
    --betas=*)
      FIT_ARGS+=(--betas "${1#*=}")
      shift
      ;;
    *.npy)
      if [[ -n "${KEYPOINTS3D}" ]]; then
        echo "Only one positional keypoints .npy path is supported." >&2
        exit 2
      fi
      KEYPOINTS3D="$1"
      FIT_ARGS+=(--keypoints3d "${KEYPOINTS3D}")
      shift
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "${FRAME_INDICES_SET}" -ne 0 ]]; then
        echo "Only one frame index/range argument is supported." >&2
        exit 2
      fi
      FRAME_INDICES="$1"
      FRAME_INDICES_SET=1
      shift
      ;;
  esac
done

FIT_ARGS+=(--frame-indices "${FRAME_INDICES}")

echo "Running local AITviewer debug fit for frame indices: ${FRAME_INDICES}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m mesh2smplx.main "${FIT_ARGS[@]}"

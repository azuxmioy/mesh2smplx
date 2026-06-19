#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_gpu.sh <config.yaml> [keypoints_3d.npy] [--tracking true|false] [--betas body_shape.npy]

Examples:
  scripts/run_gpu.sh configs/gpu.yaml
  scripts/run_gpu.sh configs/gpu.yaml data/keypoints_3d.npy
  scripts/run_gpu.sh configs/gpu.yaml --tracking false --betas data/body_shape.npy
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

FIT_ARGS=(
  --config "${CONFIG}"
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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "Running full mesh2smplx pipeline for all meshes in sorted order"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m mesh2smplx.main "${FIT_ARGS[@]}"

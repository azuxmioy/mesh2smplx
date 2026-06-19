#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_example_data.sh [--force] [--keep-archive]

Downloads mesh2smplx_example_data.zip from:
  https://huggingface.co/datasets/hohs/mesh2smplx

and extracts it into ./data.
USAGE
}

FORCE=0
KEEP_ARCHIVE=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --keep-archive)
      KEEP_ARCHIVE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ID="hohs/mesh2smplx"
ARCHIVE_NAME="mesh2smplx_example_data.zip"
ARCHIVE_PATH="${REPO_ROOT}/${ARCHIVE_NAME}"
DATA_DIR="${REPO_ROOT}/data"

if [[ -d "${DATA_DIR}/meshes" && "${FORCE}" -ne 1 ]]; then
  echo "data/meshes already exists. Use --force to overwrite the example data." >&2
  exit 2
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to extract ${ARCHIVE_NAME}." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if command -v hf >/dev/null 2>&1; then
  hf download "${DATASET_ID}" "${ARCHIVE_NAME}" \
    --repo-type dataset \
    --local-dir "${REPO_ROOT}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${DATASET_ID}" "${ARCHIVE_NAME}" \
    --repo-type dataset \
    --local-dir "${REPO_ROOT}"
elif command -v curl >/dev/null 2>&1; then
  curl -L \
    "https://huggingface.co/datasets/${DATASET_ID}/resolve/main/${ARCHIVE_NAME}" \
    -o "${ARCHIVE_PATH}"
else
  echo "Install huggingface_hub, huggingface-cli, or curl to download example data." >&2
  exit 1
fi

unzip -o "${ARCHIVE_PATH}" -d "${REPO_ROOT}"

if [[ "${KEEP_ARCHIVE}" -ne 1 ]]; then
  rm -f "${ARCHIVE_PATH}"
fi

echo "Example data is ready in ${DATA_DIR}"

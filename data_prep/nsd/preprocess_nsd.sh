#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBJECT=1
DATA_SPACE="nativesurface"
ROI_MAPPING="combined"
DS_DIR=""
OUTPUT_DIR="${REPO_ROOT}/data_prep_outputs"
STIM_CSV=""
KEEP_REPS="true"
DEBUG_MODE="false"

usage() {
    cat <<'EOF'
Usage: data_prep/nsd/preprocess_nsd.sh --ds-dir PATH [OPTIONS]

Run the NSD preprocessing example for one subject.

Options:
  --ds-dir PATH          Root NSD dataset directory. Required.
  --subject N            Subject id. Default: 1.
  --data-space NAME      Data space. Default: nativesurface.
  --roi-mapping NAME     ROI mapping. Default: combined.
  --output-dir PATH      Output directory.
  --stim-csv PATH        Stimulus mapping CSV path.
  --debug-mode           Run a smaller debug preprocessing job.
  --no-keep-reps         Do not pass --keep-reps to preprocess_nsd.py.
  -h, --help             Show this help message.
EOF
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "Missing value for ${flag}" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ds-dir) require_value "$1" "${2:-}"; DS_DIR="$2"; shift 2 ;;
        --subject) require_value "$1" "${2:-}"; SUBJECT="$2"; shift 2 ;;
        --data-space) require_value "$1" "${2:-}"; DATA_SPACE="$2"; shift 2 ;;
        --roi-mapping) require_value "$1" "${2:-}"; ROI_MAPPING="$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --stim-csv) require_value "$1" "${2:-}"; STIM_CSV="$2"; shift 2 ;;
        --debug-mode) DEBUG_MODE="true"; shift ;;
        --no-keep-reps) KEEP_REPS="false"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${DS_DIR}" ]]; then
    echo "Missing required option: --ds-dir" >&2
    usage >&2
    exit 2
fi

STIM_CSV="${STIM_CSV:-${OUTPUT_DIR}/nsd_stim_mapping.csv}"

mkdir -p "${OUTPUT_DIR}"

args=(
  --subject "${SUBJECT}"
  --data-space "${DATA_SPACE}"
  --roi-mapping "${ROI_MAPPING}"
  --ds-dir "${DS_DIR}"
  --stim-csv-path "${STIM_CSV}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${KEEP_REPS}" == "true" ]]; then
  args+=(--keep-reps)
fi

if [[ "${DEBUG_MODE}" == "true" ]]; then
  args+=(--debug-mode)
fi

python "${SCRIPT_DIR}/preprocess_nsd.py" "${args[@]}"

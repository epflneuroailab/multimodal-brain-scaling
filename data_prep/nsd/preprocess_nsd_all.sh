#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBJECTS=(1 2 3 4 5 6 7 8)
DATA_SPACES=("func1pt8mm" "nativesurface" "fsaverage")
ROI_MAPPING=("combined")
DS_DIR=""
OUTPUT_DIR="${REPO_ROOT}/data_prep_outputs"
STIM_CSV=""
APPLY_NSDGENERAL_MASK="true"
DEBUG_MODE="false"

usage() {
    cat <<'EOF'
Usage: data_prep/nsd/preprocess_nsd_all.sh --ds-dir PATH [OPTIONS]

Run the NSD preprocessing wrapper across subjects, data spaces, and ROI mappings.

Options:
  --ds-dir PATH                Root NSD dataset directory. Required.
  --subjects "LIST"            Space-separated subject ids.
  --data-spaces "LIST"         Space-separated data spaces.
  --roi-mappings "LIST"        Space-separated ROI mappings.
  --output-dir PATH            Output directory.
  --stim-csv PATH              Stimulus mapping CSV path.
  --debug-mode                 Run smaller debug preprocessing jobs.
  --no-apply-nsdgeneral-mask   Do not pass --apply-nsdgeneral-mask.
  -h, --help                   Show this help message.
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
        --subjects) require_value "$1" "${2:-}"; read -r -a SUBJECTS <<< "$2"; shift 2 ;;
        --data-spaces) require_value "$1" "${2:-}"; read -r -a DATA_SPACES <<< "$2"; shift 2 ;;
        --roi-mappings) require_value "$1" "${2:-}"; read -r -a ROI_MAPPING <<< "$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --stim-csv) require_value "$1" "${2:-}"; STIM_CSV="$2"; shift 2 ;;
        --debug-mode) DEBUG_MODE="true"; shift ;;
        --no-apply-nsdgeneral-mask) APPLY_NSDGENERAL_MASK="false"; shift ;;
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

for roi_mapping in "${ROI_MAPPING[@]}"; do
    for data_space in "${DATA_SPACES[@]}"; do
        for subject in "${SUBJECTS[@]}"; do

            args=(
              --subject "${subject}"
              --data-space "${data_space}"
              --roi-mapping "${roi_mapping}"
              --ds-dir "${DS_DIR}"
              --stim-csv-path "${STIM_CSV}"
              --output-dir "${OUTPUT_DIR}"
            )

            if [[ "${APPLY_NSDGENERAL_MASK}" == "true" ]]; then
              args+=(--apply-nsdgeneral-mask)
            fi

            if [[ "${DEBUG_MODE}" == "true" ]]; then
              args+=(--debug-mode)
            fi

            python "${SCRIPT_DIR}/preprocess_nsd.py" "${args[@]}"

        done

    done

done

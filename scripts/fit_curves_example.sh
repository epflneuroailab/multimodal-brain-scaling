#!/usr/bin/env bash
# End-to-end scaling-curve fitting example.
#
# Expects pretraining result tables already restored locally via
# `mbs-download-artifacts`. Pass options on the command line to point at
# custom inputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_CSV="./artifacts/pretraining_results_with_metadata.csv"
EXPERIMENT_CONFIG="configs/analysis/scaling_compute/architecture_average/benchmark_average.yaml"
OUTPUT_DIR="./outputs/curve_fits"
ARTIFACT_DIR="./outputs/curve_fit_bootstraps"
NUM_WORKERS="8"
NUM_BOOTSTRAPS="100"

usage() {
    cat <<'EOF'
Usage: scripts/fit_curves_example.sh [OPTIONS]

Run the scaling-curve fitting example.

Options:
  --results-csv PATH          Input results CSV.
  --experiment-config PATH    Scaling experiment config YAML.
  --output-dir PATH           Directory for fitted curve summaries.
  --artifact-dir PATH         Directory for bootstrap artifacts.
  --num-workers N             Number of worker processes.
  --num-bootstraps N          Number of bootstrap samples.
  -h, --help                  Show this help message.
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
        --results-csv) require_value "$1" "${2:-}"; RESULTS_CSV="$2"; shift 2 ;;
        --experiment-config) require_value "$1" "${2:-}"; EXPERIMENT_CONFIG="$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --artifact-dir) require_value "$1" "${2:-}"; ARTIFACT_DIR="$2"; shift 2 ;;
        --num-workers) require_value "$1" "${2:-}"; NUM_WORKERS="$2"; shift 2 ;;
        --num-bootstraps) require_value "$1" "${2:-}"; NUM_BOOTSTRAPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

MBS_EXTRAS="training evaluation analysis"
MBS_FORCE_SYNC="false"
source "${SCRIPT_DIR}/env.sh"

mbs_run mbs-fit-curves \
    --experiment-config "$EXPERIMENT_CONFIG" \
    --results-csv "$RESULTS_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --num-workers "$NUM_WORKERS" \
    --num-bootstraps "$NUM_BOOTSTRAPS" \
    --overwrite

#!/usr/bin/env bash
# End-to-end scaling-curve fitting example.
#
# Expects pretraining result tables already restored locally via
# `mbs-download-artifacts`. Override variables on the command line or in
# your environment to point at custom inputs.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

RESULTS_CSV="${RESULTS_CSV:-./artifacts/pretraining_results_with_metadata.csv}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/analysis/scaling_compute/architecture_average/benchmark_average.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/curve_fits}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./outputs/curve_fit_bootstraps}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_BOOTSTRAPS="${NUM_BOOTSTRAPS:-100}"

mbs_run mbs-fit-curves \
    --experiment-config "$EXPERIMENT_CONFIG" \
    --results-csv "$RESULTS_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --num-workers "$NUM_WORKERS" \
    --num-bootstraps "$NUM_BOOTSTRAPS" \
    --overwrite

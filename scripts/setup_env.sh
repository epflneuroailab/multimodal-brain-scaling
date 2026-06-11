#!/usr/bin/env bash
# Install the extras needed by the example workflows in one uv sync.
# Example scripts call this logic automatically when the environment is missing;
# run this script directly when you want an explicit setup or refresh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBS_EXTRAS="${MBS_EXTRAS:-${EXTRAS:-training evaluation analysis}}"
MBS_FORCE_SYNC="${MBS_FORCE_SYNC:-true}"

source "${SCRIPT_DIR}/env.sh"
mbs_prepare_env

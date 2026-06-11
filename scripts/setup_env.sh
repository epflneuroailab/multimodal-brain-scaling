#!/usr/bin/env bash
# Install the extras needed by the example workflows in one uv sync.
# Example scripts call this logic automatically when the environment is missing;
# run this script directly when you want an explicit setup or refresh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBS_EXTRAS="training evaluation analysis"
MBS_FORCE_SYNC="false"

usage() {
    cat <<'EOF'
Usage: scripts/setup_env.sh [OPTIONS]

Install the optional dependencies needed by the example workflows.

Options:
  --extras EXTRAS     Space-separated extras to install.
  --all-extras        Install every optional dependency group.
  --force-sync        Refresh the environment even if it is marked ready.
  -h, --help          Show this help message.
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
        --extras)
            require_value "$1" "${2:-}"
            MBS_EXTRAS="$2"
            shift 2
            ;;
        --all-extras)
            MBS_EXTRAS="all-extras"
            shift
            ;;
        --force-sync)
            MBS_FORCE_SYNC="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

source "${SCRIPT_DIR}/env.sh"
mbs_prepare_env

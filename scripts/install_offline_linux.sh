#!/usr/bin/env bash
set -euo pipefail

REQ_FILE="${1:-requirements.txt}"
WHEEL_DIR="artifacts/wheels/linux"

export PIP_NO_INDEX=1
export PIP_FIND_LINKS="${WHEEL_DIR}"

python -m pip install --no-index --find-links "${WHEEL_DIR}" -r "${REQ_FILE}"

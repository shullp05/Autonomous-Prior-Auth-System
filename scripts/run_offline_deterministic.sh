#!/usr/bin/env bash
set -euo pipefail

# Offline deterministic run: no sockets allowed, no LLM usage.
export PA_USE_DETERMINISTIC="true"
export PA_LETTER_MODE="deterministic"
export PA_OFFLINE_MODE="true"
export PA_OFFLINE_ALLOW_LOCALHOST="false"
export PA_OFFLINE_STRICT_UNKNOWN_HOST="true"

# Enforce offline + telemetry-off envs (explicit, not implicit).
export HF_HUB_OFFLINE="1"
export HF_HUB_DISABLE_TELEMETRY="1"
export TRANSFORMERS_OFFLINE="1"
export LANGCHAIN_TRACING_V2="false"
export LANGSMITH_TRACING="false"
export LANGSMITH_DISABLED="true"
export ANONYMIZED_TELEMETRY="false"

python batch_runner.py "$@"

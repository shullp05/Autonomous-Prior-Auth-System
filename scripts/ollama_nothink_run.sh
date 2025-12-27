#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: scripts/ollama_nothink_run.sh <model> <prompt...>}"
shift || true
PROMPT="${*:-}"

# Hard disable thinking at runtime
# (Better than relying on "/no_think" suffix alone.)
if [[ -n "$PROMPT" ]]; then
  ollama run "$MODEL" --think=false "$PROMPT"
else
  ollama run "$MODEL" --think=false
fi

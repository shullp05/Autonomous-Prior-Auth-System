#!/usr/bin/env bash
set -euo pipefail

# Generate requirements.lock from active environment
# Filters out file-based or editable installs which might break reproducible builds
# Adds the PyTorch CPU index so torch==*+cpu can resolve in CI.
echo "Freezing dependencies..."

{
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  pip freeze | grep -v "file://" | grep -v "-e "
} > requirements.lock

echo "Dependencies locked to requirements.lock"
cat requirements.lock | head -n 5
echo "..."

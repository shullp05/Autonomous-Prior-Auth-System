#!/bin/bash
set -e

# Generate requirements.lock from active environment
# Filters out file-based or editable installs which might break reproducible builds
echo "Freezing dependencies..."

pip freeze | grep -v "file://" | grep -v "-e " > requirements.lock

echo "Dependencies locked to requirements.lock"
cat requirements.lock | head -n 5
echo "..."

#!/bin/bash
# Build all PA-Audit models from Modelfiles
# Run from project root: ./models/build_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  Building PA-Audit Models"
echo "=========================================="

# Check if base models are available
echo ""
echo "Checking base models..."

check_model() {
    if ollama list | grep -q "$1"; then
        echo "  ✓ $1 found"
        return 0
    else
        echo "  ✗ $1 not found - pulling..."
        ollama pull "$1"
    fi
}

check_model "qwen2.5:14b-instruct-q4_K_M"
check_model "qwen3:14b"
check_model "mistral-nemo:latest"

# Build custom models
echo ""
echo "Building custom PA-Audit models..."

echo ""
echo ">>> Building pa-audit-qwen25..."
ollama create pa-audit-qwen25 -f "$SCRIPT_DIR/pa-audit-qwen25/Modelfile"
echo "  ✓ pa-audit-qwen25 created"

echo ""
echo ">>> Building pa-audit-qwen3..."
ollama create pa-audit-qwen3 -f "$SCRIPT_DIR/pa-audit-qwen3/Modelfile"
echo "  ✓ pa-audit-qwen3 created"

echo ""
echo ">>> Building pa-audit-mistral..."
ollama create pa-audit-mistral -f "$SCRIPT_DIR/pa-audit-mistral/Modelfile"
echo "  ✓ pa-audit-mistral created"

echo ""
echo "=========================================="
echo "  All models built successfully!"
echo "=========================================="
echo ""
echo "Available PA-Audit models:"
ollama list | grep "pa-audit"
echo ""
echo "To use a model, set in .env:"
echo "  PA_AUDIT_MODEL_FLAVOR=qwen25   # (recommended - best accuracy)"
echo "  PA_AUDIT_MODEL_FLAVOR=mistral  # (fastest)"
echo "  PA_AUDIT_MODEL_FLAVOR=qwen3    # (slow - thinking mode)"

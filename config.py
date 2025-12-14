"""
config.py - Centralized Configuration for PriorAuth Agent

This module defines all environment variables, model names, and system thresholds.
It is the single source of truth for configuration to prevent "magic strings" 
and hardcoded values scattered across the codebase.
"""

import os
from typing import Dict, Any

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
USE_RAW_MODELS = os.getenv("PA_USE_RAW_MODELS", "false").lower() == "true"
AUDIT_MODEL_FLAVOR = os.getenv("PA_AUDIT_MODEL_FLAVOR", "qwen25")
USE_DETERMINISTIC = os.getenv("PA_USE_DETERMINISTIC", "false").lower() == "true"

# -----------------------------------------------------------------------------
# Repository discovery configuration
# -----------------------------------------------------------------------------
# MAX_SEARCH_DEPTH controls how deep file discovery traverses the tree when
# listing repository contents for audits and orphan detection. Hidden directories
# and files (those starting with a dot) are excluded via the pattern '*/.*'.
# This aligns with zero‑trust principles by avoiding scanning of hidden VCS or
# environment folders.
MAX_SEARCH_DEPTH: int = int(os.getenv("PA_MAX_SEARCH_DEPTH", "5"))

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
# Models are built from Modelfiles with optimized settings:
#   - Context window: 4096 tokens (RAM efficient)
#   - Temperature/top_p/num_predict baked in
#   - System prompts for clinical PA evaluation
# 
# Build models: ./models/build_models.sh

MODEL_MAP_CUSTOM: Dict[str, Dict[str, Any]] = {
    "mistral": {
        "name": "pa-audit-mistral",
        "options": {},  # All settings baked into Modelfile
        "ram_gb": 8,    # Approximate RAM requirement
    },
    "qwen25": {
        "name": "pa-audit-qwen25",
        "options": {},
        "ram_gb": 10,
    },
    "qwen3": {
        "name": "pa-audit-qwen3",
        "options": {},
        "ram_gb": 12,
    },
}

MODEL_MAP_RAW: Dict[str, Dict[str, Any]] = {
    "mistral": {
        "name": "mistral-nemo:12b-instruct-2407-q4_K_M",
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 512, "num_ctx": 4096},
        "ram_gb": 8,
    },
    "qwen25": {
        "name": "qwen2.5:14b-instruct-q4_K_M",
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 768, "num_ctx": 4096},
        "ram_gb": 10,
    },
    "qwen3": {
        "name": "qwen3:latest",
        "options": {"temperature": 0.25, "top_p": 0.95, "num_predict": 512, "num_ctx": 4096},
        "ram_gb": 12,
    },
}

# Select map based on environment
MODEL_MAP = MODEL_MAP_RAW if USE_RAW_MODELS else MODEL_MAP_CUSTOM

# Fallback to 'mistral' if unknown flavor requested
if AUDIT_MODEL_FLAVOR not in MODEL_MAP:
    print(f"[WARN] Unknown model flavor '{AUDIT_MODEL_FLAVOR}', defaulting to 'qwen25'")
    AUDIT_MODEL_FLAVOR = "qwen25"

# Active Model Config
ACTIVE_MODEL_CONFIG = MODEL_MAP[AUDIT_MODEL_FLAVOR]
AUDIT_MODEL_NAME = ACTIVE_MODEL_CONFIG["name"]
AUDIT_MODEL_OPTIONS = ACTIVE_MODEL_CONFIG["options"]
AUDIT_MODEL_RAM_GB = ACTIVE_MODEL_CONFIG["ram_gb"]

# Specialized Models (can be overridden if needed)
APPEAL_MODEL_NAME = os.getenv("PA_APPEAL_MODEL", AUDIT_MODEL_NAME)
EMBED_MODEL_NAME = os.getenv("PA_EMBED_MODEL", "kronos483/MedEmbed-large-v0.1:latest")
PA_RERANK_MODEL = os.getenv("PA_RERANK_MODEL", "maidalun1020/bce-reranker-base_v1")
PA_ENABLE_RERANK = os.getenv("PA_ENABLE_RERANK", "true").lower() == "true"
PA_RERANK_DEVICE = os.getenv("PA_RERANK_DEVICE", "cuda").lower()
PA_RAG_K_VECTOR = int(os.getenv("PA_RAG_K_VECTOR", "25"))
PA_RAG_TOP_K_DOCS = int(os.getenv("PA_RAG_TOP_K_DOCS", "8"))
PA_RAG_SCORE_FLOOR = float(os.getenv("PA_RAG_SCORE_FLOOR", "0.35"))
PA_RAG_MIN_DOCS = int(os.getenv("PA_RAG_MIN_DOCS", "3"))
POLICY_ID = os.getenv("PA_POLICY_ID", "RX-WEG-2025")



# =============================================================================
# SYSTEM THRESHOLDS
# =============================================================================
FNR_ALERT_THRESHOLD = 0.10  # Governance audit: 10% discrepancy triggers alert
CLAIM_VALUE_USD = 1350.00   # Average claim value for revenue calculations
DEFAULT_CLAIM_RATE = 0.15   # Default claim rate for chaos_monkey scenario generation

# =============================================================================
# CLINICAL CODING CONSTANTS
# =============================================================================
LOINC_BMI = "39156-5"
LOINC_WEIGHT = "29463-7"
LOINC_HEIGHT = "8302-2"

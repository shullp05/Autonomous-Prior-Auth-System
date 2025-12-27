"""
config.py - Centralized Configuration for PriorAuth Agent

Single source of truth for environment variables, model names, and thresholds.
"""

import os
from typing import Any

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

def _as_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

# IMPORTANT: True means use RAW upstream Ollama model names; False means use custom Modelfile-built aliases.
USE_RAW_MODELS = _as_bool("PA_USE_RAW_MODELS", "false")  # FIXED (was inverted)
AUDIT_MODEL_FLAVOR = os.getenv("PA_AUDIT_MODEL_FLAVOR", "nemo8b").strip()
USE_DETERMINISTIC = _as_bool("PA_USE_DETERMINISTIC", "false")
OFFLINE_MODE = _as_bool("PA_OFFLINE_MODE", "false")

# -----------------------------------------------------------------------------
# Repository discovery configuration
# -----------------------------------------------------------------------------
MAX_SEARCH_DEPTH: int = int(os.getenv("PA_MAX_SEARCH_DEPTH", "8"))


# =============================================================================
# PROVIDER / PRACTICE CONTEXT (PCP OFFICE, PROVIDER-FACING)
# =============================================================================
PA_PROVIDER_NAME = os.getenv("PA_PROVIDER_NAME", "Peter Shull").strip()
PA_PROVIDER_CREDENTIALS = os.getenv("PA_PROVIDER_CREDENTIALS", "PharmD").strip()
PA_PRACTICE_NAME = os.getenv("PA_PRACTICE_NAME", "Clearview Medical Group").strip()
PA_PROVIDER_NPI = os.getenv("PA_PROVIDER_NPI", "1234567890").strip()
PA_PRACTICE_PHONE = os.getenv("PA_PRACTICE_PHONE", "555-555-5555").strip()
PA_PRACTICE_FAX = os.getenv("PA_PRACTICE_FAX", "555-555-5556").strip()
PA_PRACTICE_ADDRESS = os.getenv("PA_PRACTICE_ADDRESS", "123 Main St, Anytown, USA").strip()

# Recipient defaults for payer-ready letters (override per client/demo)
PA_UM_RECIPIENT_ORG = os.getenv("PA_UM_RECIPIENT_ORG", "Utilization Management").strip() or "Utilization Management"
PA_UM_RECIPIENT_DEPT = os.getenv("PA_UM_RECIPIENT_DEPT", "Utilization Management Department").strip() or "Utilization Management Department"
PA_UM_ATTENTION = os.getenv("PA_UM_ATTENTION", "Medical Director").strip() or "Medical Director"

def require_provider_context() -> dict[str, str]:
    """
    Enforce that we have enough real-world identifiers to generate payer-ready letters
    WITHOUT placeholders. Fail fast if missing.
    """
    required = {
        "PA_PROVIDER_NAME": PA_PROVIDER_NAME,
        "PA_PRACTICE_NAME": PA_PRACTICE_NAME,
        "PA_PROVIDER_NPI": PA_PROVIDER_NPI,
        "PA_PRACTICE_PHONE": PA_PRACTICE_PHONE,
        "PA_PRACTICE_FAX": PA_PRACTICE_FAX,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Provider context missing required fields: {missing}. "
            "Set PA_PROVIDER_* and PA_PRACTICE_* env vars to generate payer-ready letters without placeholders."
        )
    return {
        "provider_name": PA_PROVIDER_NAME,
        "provider_credentials": PA_PROVIDER_CREDENTIALS,
        "practice_name": PA_PRACTICE_NAME,
        "npi": PA_PROVIDER_NPI,
        "phone": PA_PRACTICE_PHONE,
        "fax": PA_PRACTICE_FAX,
        "address": PA_PRACTICE_ADDRESS,
    }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
# Build custom models: ./models/build_models.sh

MODEL_MAP_CUSTOM: dict[str, dict[str, Any]] = {
    "mistral": {"name": "pa-audit-mistral", "options": {}, "ram_gb": 8},
    "qwen25": {
        "name": "pa-audit-qwen25",
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 768, "num_ctx": 4096},
        "ram_gb": 10,
    },
    "nemo8b": {
        "name": "pa-audit-nemotron-cascade8b",
        "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.1, "repeat_last_n": 256, "num_predict": 768, "num_ctx": 4096, "seed": 42},
        "ram_gb": 8,
        "stop": "<im_end>",
        "start": "<im_start>",
    },
    "nemo4b": {
        "name": "pa-audit-nemo-cascade4b",
        "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.1, "repeat_last_n": 256, "num_predict": 768, "num_ctx": 4096, "seed": 42},
        "ram_gb": 6,
        "stop": "<im_end>",
        "start": "<im_start>",
    },
    "qwen3": {"name": "pa-audit-qwen3", "options": {}, "ram_gb": 12},
}

MODEL_MAP_RAW: dict[str, dict[str, Any]] = {
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
    "nemo8b": {
        "name": "hf.co/bartowski/nvidia_Nemotron-Cascade-8B-GGUF:Q6_K",
        "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.1, "repeat_last_n": 256, "num_predict": 768, "num_ctx": 4096, "seed": 42},
        "ram_gb": 8,
        "stop": "<im_end>",
        "start": "<im_start>",
    },
    "nemo4b": {
        "name": "hf.co/bartowski/nvidia_Nemotron-Cascade-8B-GGUF:Q4_K_M",
        "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.1, "repeat_last_n": 256, "num_predict": 768, "num_ctx": 4096, "seed": 42},
        "ram_gb": 6,
        "stop": "<im_end>",
        "start": "<im_start>",
    },
    "qwen3": {
        "name": "qwen3:latest",
        "options": {"temperature": 0.25, "top_p": 0.95, "num_predict": 512, "num_ctx": 4096},
        "ram_gb": 12,
    },
}

MODEL_MAP = MODEL_MAP_RAW if USE_RAW_MODELS else MODEL_MAP_CUSTOM

if AUDIT_MODEL_FLAVOR not in MODEL_MAP:
    # Avoid print-on-import in libraries; prefer raising or logging in the entrypoint.
    AUDIT_MODEL_FLAVOR = "nemo8b"

ACTIVE_MODEL_CONFIG = MODEL_MAP[AUDIT_MODEL_FLAVOR]
AUDIT_MODEL_NAME = ACTIVE_MODEL_CONFIG["name"]
AUDIT_MODEL_OPTIONS = ACTIVE_MODEL_CONFIG["options"]
AUDIT_MODEL_RAM_GB = ACTIVE_MODEL_CONFIG["ram_gb"]

# Specialized Models (can be overridden if needed)
APPEAL_MODEL_NAME = os.getenv("PA_APPEAL_MODEL", AUDIT_MODEL_NAME).strip()
EMBED_MODEL_NAME = os.getenv("PA_EMBED_MODEL", "kronos483/MedEmbed-large-v0.1:latest").strip()

PA_RERANK_MODEL = os.getenv("PA_RERANK_MODEL", "maidalun1020/bce-reranker-base_v1").strip()
PA_ENABLE_RERANK = _as_bool("PA_ENABLE_RERANK", "true")
PA_RERANK_DEVICE = os.getenv("PA_RERANK_DEVICE", "cuda").strip().lower()

PA_RAG_K_VECTOR = int(os.getenv("PA_RAG_K_VECTOR", "25"))
PA_RAG_TOP_K_DOCS = int(os.getenv("PA_RAG_TOP_K_DOCS", "8"))
PA_RAG_SCORE_FLOOR = float(os.getenv("PA_RAG_SCORE_FLOOR", "0.35"))
PA_RAG_MIN_DOCS = int(os.getenv("PA_RAG_MIN_DOCS", "3"))

POLICY_ID = os.getenv("PA_POLICY_ID", "RX-WEG-2025").strip()


# =============================================================================
# SYSTEM THRESHOLDS
# =============================================================================
FNR_ALERT_THRESHOLD = 0.10
CLAIM_VALUE_USD = 1350.00
DEFAULT_CLAIM_RATE = 0.15


# =============================================================================
# CLINICAL CODING CONSTANTS
# =============================================================================
LOINC_BMI = "39156-5"
LOINC_WEIGHT = "29463-7"
LOINC_HEIGHT = "8302-2"

# SNOMED-CT (Conditions/Findings)
SNOMED_OBESE = "162864005"       # Obesity (BMI 30+)
SNOMED_OVERWEIGHT = "162863004"  # Overweight (BMI 25-29)

# ICD-10-CM (Diagnosis Codes)
ICD10_OVERWEIGHT = "E66.3"       # Overweight
ICD10_OBESITY = "E66.9"          # Obesity, unspecified
ICD10_MORBID = "E66.01"          # Morbid Obesity

# Strict Validation Lists
VALID_OBESITY_DX_CODES = {
    ICD10_OVERWEIGHT,
    ICD10_OBESITY,
    ICD10_MORBID,
    "E66.09", "E66.1", "E66.2", "E66.8", "E66.0"
}
VALID_BMI_Z_PREFIX = "Z68"

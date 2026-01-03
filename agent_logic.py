"""
agent_logic.py - LangGraph-based Clinical Prior Authorization Agent

This module implements an autonomous agent for evaluating Wegovy (semaglutide)
prior authorization requests using a combination of:

1. **RAG (Retrieval-Augmented Generation)**: Policy retrieval from ChromaDB
2. **LLM Reasoning**: Clinical narration / audit via local LLM with structured JSON output
3. **Python Guardrails**: Deterministic policy engine is the single source of truth for eligibility
4. **Pydantic Validation**: Schema enforcement on LLM outputs

Architecture:
    retrieve_policy → clinical_audit → make_decision → END

Key Functions:
    - build_agent(): Constructs the LangGraph workflow
    - retrieve_policy(): RAG retrieval with optional reranking
    - clinical_audit(): Runs deterministic engine + optional LLM narration
    - generate_approved_letter(): Creates PA request letters for approvals
    - generate_appeal_letter(): Creates provider-facing documentation/clarification letters for denials/flags

Configuration (Environment Variables):
    - PA_AUDIT_MODEL: LLM for clinical narration (default from config)
    - PA_APPEAL_MODEL: LLM for letter drafting (default from config)
    - PA_EMBED_MODEL: Embedding model for RAG (default from config)
    - PA_ENABLE_RERANK: Enable/disable reranking (default: true)
    - PA_RAG_SCORE_FLOOR: BCE score threshold (default from config)
    - PA_RAG_MIN_DOCS: Minimum docs to keep even if scores are low (default from config)
    - PA_PROVIDER_*: Provider context for payer-ready letters (required for structured letter mode)

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, TypedDict, Union

import pandas as pd
import psutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

# IMPORTANT: load .env BEFORE importing config (config may read env at import time)
load_dotenv()

logger = logging.getLogger(__name__)

# ---- Policy snapshot / deterministic engine (single source of truth) ----
from policy_utils import normalize, format_criteria_list
from letter_service import LetterResult, generate_approved_letter, generate_appeal_letter
from audit_logger import get_audit_logger  # noqa: E402

# ---- Optional reranker (safe-to-import wrapper) ----
# This module is designed to not blow up imports if BCEmbedding isn't installed.
from bce_reranker import rerank_bce  # noqa: E402

# ---- Central config ----
from config import (  # noqa: E402
    APPEAL_MODEL_NAME,
    AUDIT_MODEL_FLAVOR,
    AUDIT_MODEL_NAME,
    AUDIT_MODEL_OPTIONS,
    AUDIT_MODEL_RAM_GB,
    EMBED_MODEL_NAME,
    PA_RAG_K_VECTOR,
    PA_RAG_MIN_DOCS,
    PA_RAG_SCORE_FLOOR,
    PA_RAG_TOP_K_DOCS,
    PA_RERANK_DEVICE,
    PA_RERANK_MODEL,
)
from config import (
    POLICY_ID as ACTIVE_POLICY_ID,
)
from policy_constants import (  # noqa: E402
    AMBIGUOUS_APPEAL_TERMS,
    BMI_OBESE_THRESHOLD,
    BMI_OVERWEIGHT_THRESHOLD,
)
from policy_engine import _parse_bmi, evaluate_eligibility  # noqa: E402
from policy_snapshot import SNAPSHOT_PATH, load_policy_snapshot  # noqa: E402
from schema_validation import validate_policy_snapshot  # noqa: E402

# =========================
# CONFIG (easy model swap)
# =========================

AUDIT_MODEL = AUDIT_MODEL_NAME
AUDIT_MODEL_OPTS = AUDIT_MODEL_OPTIONS or {}
AUDIT_MODEL_RAM = AUDIT_MODEL_RAM_GB

APPEAL_MODEL = APPEAL_MODEL_NAME
EMBED_MODEL = EMBED_MODEL_NAME
RERANK_MODEL = PA_RERANK_MODEL
RERANK_DEVICE_DEFAULT = (PA_RERANK_DEVICE or "cpu").lower()

from config import PA_ENABLE_RERANK as ENABLE_RERANK

# RAG scoring knobs: config defaults, env override supported
RAG_SCORE_FLOOR = float(os.getenv("PA_RAG_SCORE_FLOOR", str(PA_RAG_SCORE_FLOOR)))
RAG_MIN_DOCS = int(os.getenv("PA_RAG_MIN_DOCS", str(PA_RAG_MIN_DOCS)))

SNAPSHOT = load_policy_snapshot(SNAPSHOT_PATH, ACTIVE_POLICY_ID)
validate_policy_snapshot(SNAPSHOT)

# --- RAG CONFIG ---
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "priorauth_policies"
_STATIC_POLICY_FALLBACK = json.dumps(SNAPSHOT, indent=2, ensure_ascii=True)

# --- DATA LOADING (lazy) ---
df_patients: pd.DataFrame | None = None
df_meds: pd.DataFrame | None = None
df_conditions: pd.DataFrame | None = None
df_obs: pd.DataFrame | None = None
_REFERENCE_YEAR: int | None = None

_DATA_DIR = Path(os.getenv("ETL_OUTPUT_DIR", "output"))
_PAT_PATH = _DATA_DIR / "data_patients.csv"
_MED_PATH = _DATA_DIR / "data_medications.csv"
_COND_PATH = _DATA_DIR / "data_conditions.csv"
_OBS_PATH = _DATA_DIR / "data_observations.csv"


def _coerce_pid(x: Any) -> str:
    return str(x).strip()


def _clean_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        out.append(s)
    return out


def _env_reference_year() -> int | None:
    ref_date = os.getenv("PA_REFERENCE_DATE", "").strip()
    if ref_date:
        try:
            return int(pd.to_datetime(ref_date, errors="raise").year)
        except Exception:
            logger.warning("Invalid PA_REFERENCE_DATE=%s; ignoring.", ref_date)
    ref_year = os.getenv("PA_REFERENCE_YEAR", "").strip()
    if ref_year:
        try:
            return int(ref_year)
        except Exception:
            logger.warning("Invalid PA_REFERENCE_YEAR=%s; ignoring.", ref_year)
    return None


def _resolve_reference_year(candidates: list[tuple[pd.DataFrame | None, str]] | None = None) -> int:
    env_year = _env_reference_year()
    if env_year is not None:
        return env_year
    years: list[int] = []
    if candidates:
        for df, col in candidates:
            if df is None or getattr(df, "empty", True) or col not in df.columns:
                continue
            dates = pd.to_datetime(df[col], errors="coerce")
            if dates.empty:
                continue
            max_year = dates.dt.year.max()
            if pd.notna(max_year):
                years.append(int(max_year))
    if years:
        return max(years)
    return datetime.now(UTC).year


def _current_year() -> int:
    return _REFERENCE_YEAR or _resolve_reference_year()


def _filter_current_year_rows(
    df: pd.DataFrame | None,
    date_col: str,
    reference_year: int | None = None,
) -> pd.DataFrame | None:
    if df is None or getattr(df, "empty", True) or date_col not in df.columns:
        return df
    if reference_year is None:
        reference_year = _REFERENCE_YEAR or _resolve_reference_year([(df, date_col)])
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return df.loc[dates.dt.year == reference_year].copy()


def _latest_bmi_observation(p_obs: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None]:
    if p_obs.empty or "date" not in p_obs.columns:
        return None, None
    bmi_rows = p_obs[p_obs["type"] == "BMI"].copy()
    if bmi_rows.empty:
        return None, None
    bmi_rows["date_parsed"] = pd.to_datetime(bmi_rows["date"], errors="coerce")
    bmi_rows = bmi_rows.dropna(subset=["date_parsed"]).sort_values("date_parsed", ascending=False)
    if bmi_rows.empty:
        return None, None
    try:
        return bmi_rows.iloc[0]["date_parsed"], float(bmi_rows.iloc[0]["value"])
    except Exception:
        return None, None


def _latest_height_weight_pair(p_obs: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None, float | None]:
    if p_obs.empty or "date" not in p_obs.columns:
        return None, None, None
    obs = p_obs[p_obs["type"].isin(["Height", "Weight"])].copy()
    if obs.empty:
        return None, None, None
    obs["date_parsed"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date_parsed"])
    if obs.empty:
        return None, None, None
    height_rows = obs[obs["type"] == "Height"]
    weight_rows = obs[obs["type"] == "Weight"]
    if height_rows.empty or weight_rows.empty:
        return None, None, None
    height_by_date = height_rows.sort_values("date_parsed").groupby("date_parsed").tail(1)
    weight_by_date = weight_rows.sort_values("date_parsed").groupby("date_parsed").tail(1)
    common_dates = set(height_by_date["date_parsed"]) & set(weight_by_date["date_parsed"])
    if not common_dates:
        return None, None, None
    latest_date = max(common_dates)
    try:
        height_cm = float(height_by_date[height_by_date["date_parsed"] == latest_date].iloc[0]["value"])
        weight_kg = float(weight_by_date[weight_by_date["date_parsed"] == latest_date].iloc[0]["value"])
    except Exception:
        return None, None, None
    return latest_date, height_cm, weight_kg


def _ensure_data_loaded() -> None:
    """Lazy load data to allow module import without files present."""
    global df_patients, df_meds, df_conditions, df_obs, _REFERENCE_YEAR
    if df_patients is not None:
        return

    required = [_PAT_PATH, _MED_PATH, _COND_PATH, _OBS_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(
            "Data files not found (%s). Run etl_pipeline.py and/or chaos_monkey.py to generate outputs.",
            ", ".join(missing),
        )
        return

    try:
        df_patients = pd.read_csv(_PAT_PATH)
        df_meds = pd.read_csv(_MED_PATH)
        df_conditions = pd.read_csv(_COND_PATH)
        df_obs = pd.read_csv(_OBS_PATH)

        _REFERENCE_YEAR = _resolve_reference_year(
            [
                (df_meds, "date"),
                (df_conditions, "onset_date"),
                (df_obs, "date"),
            ]
        )
        df_meds = _filter_current_year_rows(df_meds, "date", _REFERENCE_YEAR)
        df_conditions = _filter_current_year_rows(df_conditions, "onset_date", _REFERENCE_YEAR)
        df_obs = _filter_current_year_rows(df_obs, "date", _REFERENCE_YEAR)

        # CRITICAL: normalize patient_id to string across all frames for consistent joins/filters
        for df in (df_patients, df_meds, df_conditions, df_obs):
            if df is not None and not getattr(df, "empty", True) and "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].astype(str)

        # Normalize observation type casing if present (defensive)
        if df_obs is not None and not getattr(df_obs, "empty", True) and "type" in df_obs.columns:
            df_obs["type"] = df_obs["type"].astype(str)

    except Exception as e:
        logger.error("Data Load Error: %s", e)


def write_model_trace(model_name: str, role: str, params: dict, required_ram_gb: float | None = None) -> None:
    trace = {
        "model_name": model_name,
        "role": role,
        "params": params,
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
        "ram_required_gb": required_ram_gb or "unknown",
    }
    try:
        with open(".last_model_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
    except Exception as e:
        logger.warning("Could not write model trace: %s", e)


def _make_llm(
    model: str,
    temperature: float = 0.0,
    prefer_json: bool = False,
    options: dict | None = None,
    format_schema: dict | None = None,
) -> ChatOllama:
    """
    Create ChatOllama instance with optional format and low-level tuning.
    """
    try:
        from langchain_ollama import ChatOllama
    except Exception as e:
        raise RuntimeError(f"ChatOllama import failed: {e}") from e
    if options is None:
        options = {}
    kwargs = {"model": model, "temperature": temperature, **options}
    if format_schema is not None:
        kwargs["format"] = format_schema
    elif prefer_json:
        kwargs["format"] = "json"
    try:
        return ChatOllama(**kwargs)
    except TypeError:
        # Legacy fallback if older langchain_ollama doesn't accept `format`
        kwargs.pop("format", None)
        return ChatOllama(**kwargs)


# --- BMI CALCULATION ---
def calculate_bmi_if_missing(patient_id: str, df_obs_in: pd.DataFrame | None) -> str:
    """
    Retrieve or calculate BMI for a patient.

    Returns:
        - "<value> (Source: EMR)"  if BMI observation exists
        - "<value> (Calculated)"   if calculated from height/weight
        - "MISSING_DATA"           if cannot determine
    """
    if df_obs_in is None or getattr(df_obs_in, "empty", True):
        return "MISSING_DATA"

    pid = _coerce_pid(patient_id)
    p_obs = df_obs_in[df_obs_in["patient_id"].astype(str) == pid].copy()
    p_obs = _filter_current_year_rows(p_obs, "date")
    if p_obs.empty:
        return "MISSING_DATA"

    # 1) Choose the most recent visit data (BMI row or height/weight pair)
    bmi_date, bmi_val = _latest_bmi_observation(p_obs)
    hw_date, height_cm, weight_kg = _latest_height_weight_pair(p_obs)

    if bmi_val is None and height_cm is None:
        return "MISSING_DATA"

    if bmi_date is not None and bmi_val is not None and (hw_date is None or bmi_date >= hw_date):
        return f"{round(bmi_val, 1)} (Source: EMR)"

    # 2) Calculate from height/weight
    try:
        if height_cm is None or weight_kg is None:
            return "MISSING_DATA"

        height_m = height_cm / 100.0
        if height_m <= 0:
            return "MISSING_DATA"

        calculated_bmi = round(weight_kg / (height_m**2), 1)
        return f"{calculated_bmi} (Calculated)"
    except Exception:
        return "MISSING_DATA"


# --- PATIENT LOOKUP ---
def look_up_patient_data(patient_id: str) -> dict | None:
    """
    Retrieve patient data from loaded CSVs. Returns None if data not loaded or patient missing.
    """
    _ensure_data_loaded()

    if df_patients is None or getattr(df_patients, "empty", True):
        logger.error("Patient data not loaded (run etl_pipeline.py then chaos_monkey.py).")
        return None

    pid = _coerce_pid(patient_id)

    pat_rows = df_patients[df_patients["patient_id"].astype(str) == pid].to_dict("records")
    if not pat_rows:
        return None

    pat = pat_rows[0]

    meds: list[str] = []
    if df_meds is not None and not getattr(df_meds, "empty", True):
        meds_series = df_meds[df_meds["patient_id"].astype(str) == pid]["medication_name"]
        meds = _clean_str_list(meds_series.dropna().astype(str).tolist())

    conds: list[dict] = []
    if df_conditions is not None and not getattr(df_conditions, "empty", True):
        c_rows = df_conditions[df_conditions["patient_id"].astype(str) == pid]
        # Pass full condition dicts so code-level checks can see ICD anchors.
        conds = c_rows.to_dict(orient="records")

    latest_bmi = calculate_bmi_if_missing(pid, df_obs)

    return {
        "patient_id": pid,  # IMPORTANT: include for downstream letter generation / auditing
        "name": str(pat.get("name", "")).strip(),
        "dob": str(pat.get("dob", "")).strip(),
        "meds": meds,
        "conditions": conds,
        "latest_bmi": latest_bmi,
    }


# --- STATE DEFINITION ---
class AgentState(TypedDict, total=False):
    patient_id: str
    drug_requested: str
    patient_data: dict
    deterministic_decision: dict
    policy_text: str
    policy_docs: list
    audit_findings: dict
    final_decision: str
    reasoning: str
    appeal_letter: str
    appeal_note: str
    audit_model_flavor: str
    policy_path: str
    decision_type: str
    safety_exclusion_code: str
    ambiguity_code: str


# --- PYDANTIC SCHEMA FOR LLM OUTPUT VALIDATION ---
class AuditResult(BaseModel):
    """Strict-ish schema for LLM audit output validation (LLM output is advisory only)."""
    model_config = ConfigDict(extra="ignore")

    bmi_numeric: Optional[float] = None
    safety_flag: Literal["CLEAR", "DETECTED"] = "CLEAR"
    comorbidity_category: Literal["NONE", "HYPERTENSION", "LIPIDS", "DIABETES", "OSA", "CVD"] = "NONE"
    evidence_quoted: str = ""
    verdict: Literal[
        "APPROVED",
        "DENIED_SAFETY",
        "DENIED_CLINICAL",
        "DENIED_MISSING_INFO",
        "MANUAL_REVIEW",
        "DENIED_BENEFIT_EXCLUSION",
        "DENIED_OTHER",
        "CDI_REQUIRED",
        "SAFETY_SIGNAL_NEEDS_REVIEW",
    ] = "MANUAL_REVIEW"
    reasoning: str = ""

    @field_validator("comorbidity_category", mode="before")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        if isinstance(v, str) and v.upper() == "CARDIOVASCULAR_DISEASE":
            return "CVD"
        return v


def _audit_result_schema() -> dict:
    schema = AuditResult.model_json_schema()
    schema["required"] = list(AuditResult.model_fields.keys())
    schema["additionalProperties"] = False
    return schema


AUDIT_RESULT_SCHEMA = _audit_result_schema()


def _merge_deterministic_over_llm(det: dict, llm_obj: Optional[dict]) -> dict:
    """
    Deterministic engine is the source of truth for:
      verdict, bmi_numeric, safety_flag, comorbidity_category, evidence_quoted,
      policy_path, decision_type, safety_exclusion_code, ambiguity_code, reasoning.

    LLM output (if available) is retained as:
      llm_verdict, llm_reasoning, llm_evidence_quoted.
    """
    out = dict(det or {})
    if isinstance(llm_obj, dict):
        out["llm_verdict"] = str(llm_obj.get("verdict", "")).strip()
        out["llm_reasoning"] = str(llm_obj.get("reasoning", "")).strip()
        out["llm_evidence_quoted"] = str(llm_obj.get("evidence_quoted", "")).strip()

        # If deterministic reasoning is empty (shouldn't be), fall back to LLM
        if not str(out.get("reasoning") or "").strip() and out.get("llm_reasoning"):
            out["reasoning"] = out["llm_reasoning"]

    # Ensure required keys always exist for downstream code
    out.setdefault("policy_path", "UNKNOWN")
    out.setdefault("decision_type", out.get("verdict", "UNKNOWN"))
    out.setdefault("safety_exclusion_code", None)
    out.setdefault("ambiguity_code", None)
    return out


def _build_policy_query(det_result: Any, patient_data: dict, drug_requested: str) -> str:
    bmi = getattr(det_result, "bmi_numeric", None)
    verdict = getattr(det_result, "verdict", "UNKNOWN")
    policy_path = getattr(det_result, "policy_path", "UNKNOWN")
    comorbidity = getattr(det_result, "comorbidity_category", "NONE")
    evidence = getattr(det_result, "evidence_quoted", "")

    conds = patient_data.get("conditions", []) or []
    if conds and isinstance(conds[0], dict):
        cond_names = [str(c.get("condition_name", "")).strip() for c in conds if str(c.get("condition_name", "")).strip()]
    else:
        cond_names = [str(c).strip() for c in conds if str(c).strip()]

    cond_snippet = ", ".join(cond_names[:8])
    bmi_str = f"{bmi:.1f}" if isinstance(bmi, (int, float)) else "unknown"

    return (
        f"{drug_requested} prior authorization policy. "
        f"BMI {bmi_str}. Verdict {verdict}. Policy path {policy_path}. "
        f"Comorbidity {comorbidity}. Evidence {evidence}. "
        f"Conditions: {cond_snippet}"
    ).strip()


def _format_policy_evidence(docs: list[Document]) -> str:
    if not docs:
        return _STATIC_POLICY_FALLBACK

    chunks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        section = meta.get("section", "unknown")
        policy_id = meta.get("policy_id", "")
        header = f"[{idx}] section={section}"
        if policy_id:
            header += f" policy_id={policy_id}"
        content = str(doc.page_content or "").strip()
        chunks.append(f"{header}\n{content}".strip())
    return "\n\n".join(chunks).strip()


def _extract_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 1)[1]
        if "```" in raw:
            raw = raw.split("```", 1)[0]
        raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    candidate = (
        raw[first_brace : last_brace + 1].strip()
        if (first_brace != -1 and last_brace != -1 and last_brace > first_brace)
        else raw
    )

    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Parsed JSON is not an object.", candidate, 0)
    return obj


def _apply_score_floor(scored_docs: list[tuple[Document, float]], floor: float, min_docs: int) -> list[Document]:
    if not scored_docs:
        return []
    docs, scores = zip(*scored_docs)
    docs = list(docs)
    scores = list(scores)

    filtered: list[Document] = []
    for d, s in zip(docs, scores):
        if float(s) >= floor:
            filtered.append(d)
    if len(filtered) < min_docs:
        filtered = docs[:min_docs]
    return filtered


def _filter_docs_for_policy_path(docs: list[Document], policy_path: str | None) -> list[Document]:
    if not docs or not policy_path:
        return docs

    def section(d: Document) -> str:
        return str((d.metadata or {}).get("section") or "")

    if policy_path == "BMI30_OBESITY":
        allowed_prefixes = (
            "documentation:requirements",
            "eligibility:pathway1",
            "diagnosis:obesity_strings",
        )
        priority_order = [
            "eligibility:pathway1",
            "diagnosis:obesity_strings",
            "documentation:requirements",
        ]
    elif policy_path == "BMI27_COMORBIDITY":
        allowed_prefixes = (
            "documentation:requirements",
            "eligibility:pathway2",
            "diagnosis:overweight_strings",
            "comorbidity:",
        )
        priority_order = [
            "eligibility:pathway2",
            "comorbidity:hypertension",
            "diagnosis:overweight_strings",
            "documentation:requirements",
        ]
    elif policy_path == "SAFETY_EXCLUSION":
        allowed_prefixes = (
            "safety_exclusions:",
            "drug_conflicts:glp1_glp1_gip",
        )
        priority_order = [
            "safety_exclusions:mtc_men2",
            "safety_exclusions:pregnancy_nursing",
            "safety_exclusions:concurrent_glp1",
            "drug_conflicts:glp1_glp1_gip",
        ]
    elif policy_path == "AMBIGUITY_MANUAL_REVIEW":
        allowed_prefixes = (
            "ambiguity:",
            "eligibility:",
            "diagnosis:obesity_strings",
            "diagnosis:overweight_strings",
            "documentation:requirements",
        )
        priority_order = [
            "ambiguity:",
            "eligibility:pathway1",
            "eligibility:pathway2",
            "diagnosis:obesity_strings",
            "diagnosis:overweight_strings",
            "documentation:requirements",
        ]
    else:
        return docs

    filtered = [d for d in docs if section(d).startswith(allowed_prefixes)]
    if not filtered:
        return docs

    def priority_key(doc: Document) -> int:
        sec = section(doc)
        for idx, prefix in enumerate(priority_order):
            if sec.startswith(prefix):
                return idx
        return len(priority_order)

    return sorted(filtered, key=priority_key)


def _ensure_vectorstore():
    if not os.path.isdir(PERSIST_DIR):
        raise RuntimeError(
            f"ChromaDB directory '{PERSIST_DIR}' not found. Run setup_rag.py to build the policy index."
        )
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def retrieve_policy(state: AgentState) -> dict:
    logger.info("[RAG] Retrieving policy context for %s", state.get("patient_id", "unknown"))

    patient_id = state.get("patient_id")
    p_data = state.get("patient_data") or (look_up_patient_data(patient_id) if patient_id else None)
    if not p_data:
        return {"patient_data": None, "policy_text": _STATIC_POLICY_FALLBACK, "policy_docs": []}

    det_result = evaluate_eligibility(p_data)
    drug_requested = str(state.get("drug_requested") or "Wegovy")
    query = _build_policy_query(det_result, p_data, drug_requested)

    docs: list[Document] = []
    try:
        vectordb = _ensure_vectorstore()
        docs = vectordb.similarity_search(
            query,
            k=PA_RAG_K_VECTOR,
            filter={"policy_id": ACTIVE_POLICY_ID},
        )
        if ENABLE_RERANK and docs:
            scored = rerank_bce(query, docs)
            if not scored:
                scored = [(d, 0.0) for d in docs]
            filtered_docs = _apply_score_floor(scored, RAG_SCORE_FLOOR, RAG_MIN_DOCS)
            docs = _filter_docs_for_policy_path(filtered_docs, getattr(det_result, "policy_path", None))
    except Exception as e:
        logger.warning("Policy retrieval failed; using static fallback: %s", e)
        docs = []

    docs = docs[:PA_RAG_TOP_K_DOCS] if docs else []
    policy_text = _format_policy_evidence(docs)

    return {
        "patient_data": p_data,
        "policy_docs": docs,
        "policy_text": policy_text,
    }


# --- CLINICAL AUDIT NODE ---
def clinical_audit(state: AgentState) -> dict:
    logger.info("[Audit] Checking Patient %s", state.get("patient_id", ""))

    patient_id = state.get("patient_id")
    p_data = state.get("patient_data") or (look_up_patient_data(patient_id) if patient_id else None)

    if not p_data:
        audit_result = {
            "bmi_numeric": None,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "verdict": "MANUAL_REVIEW",
            "reasoning": "Patient record could not be found; route to manual review.",
            "policy_path": "UNKNOWN",
            "decision_type": "FLAGGED_AMBIGUITY",
            "safety_exclusion_code": None,
            "ambiguity_code": None,
        }
        return {"patient_data": None, "audit_findings": audit_result}

    # Deterministic decision ALWAYS computed (single source of truth)
    det_result_obj = evaluate_eligibility(p_data)
    det_decision = det_result_obj.to_dict()

    # Optional LLM advisory audit (kept for transparency / debugging / narration)
    policy_text = str(state.get("policy_text") or _STATIC_POLICY_FALLBACK)

    llm_audit_dict: dict | None = None
    try:
        det_bmi = _parse_bmi(p_data.get("latest_bmi"))
        det_bmi_str = str(det_bmi) if det_bmi is not None else "null"

        llm = _make_llm(
            model=AUDIT_MODEL,
            temperature=0,
            prefer_json=True,
            options=AUDIT_MODEL_OPTS,
            format_schema=AUDIT_RESULT_SCHEMA,
        )

        system_prompt = """
You are a Senior Utilization Review Medical Director. Your job is to summarize the eligibility logic and evidence.
IMPORTANT:
- The deterministic engine is the source of truth for the decision.
- You may disagree, but you must still output your own JSON assessment for audit/debug.

Return ONE strict JSON object. No markdown. No extra text.
"""

        user_payload = {
            "policy_evidence": policy_text,
            "patient": {
                "patient_id": p_data.get("patient_id"),
                "bmi_raw": p_data.get("latest_bmi"),
                "bmi_numeric_verified": det_bmi_str,
                "conditions": p_data.get("conditions", []),
                "meds": p_data.get("meds", []),
            },
            "deterministic_decision_source_of_truth": det_decision,
            "output_schema": {
                "bmi_numeric": "number|null",
                "safety_flag": "CLEAR|DETECTED",
                "comorbidity_category": "NONE|HYPERTENSION|LIPIDS|DIABETES|OSA|CVD",
                "evidence_quoted": "string",
                "verdict": "APPROVED|DENIED_SAFETY|DENIED_CLINICAL|DENIED_MISSING_INFO|MANUAL_REVIEW|DENIED_BENEFIT_EXCLUSION|DENIED_OTHER|CDI_REQUIRED|SAFETY_SIGNAL_NEEDS_REVIEW",
                "reasoning": "string",
            },
        }

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ]
        )
        content = str(response.content or "").strip()
        raw_json = _extract_json_object(content)

        try:
            validated = AuditResult(**raw_json)
            llm_audit_dict = validated.model_dump()
        except ValidationError as ve:
            logger.warning("[Audit] LLM schema validation failed (advisory only): %s", str(ve)[:300])
            llm_audit_dict = None

    except Exception as e:
        # advisory-only failure: do not affect deterministic decision
        logger.warning("[Audit] LLM advisory audit failed: %s", e)
        llm_audit_dict = None

    # Merge deterministic truth over LLM advisory output
    audit_result = _merge_deterministic_over_llm(det_decision, llm_audit_dict)

    write_model_trace(
        model_name=AUDIT_MODEL,
        role="clinical_audit",
        params=AUDIT_MODEL_OPTS,
        required_ram_gb=AUDIT_MODEL_RAM,
    )

    state["audit_model_flavor"] = AUDIT_MODEL_FLAVOR
    return {
        "patient_data": p_data,
        "audit_findings": audit_result,
        "audit_model_flavor": AUDIT_MODEL_FLAVOR,
    }


# --- DECISION NODE ---
def make_decision(state: AgentState) -> dict:
    """
    Turn findings into:
      - final_decision: APPROVED / DENIED / FLAGGED / PROVIDER_ACTION_REQUIRED
      - reasoning: human-readable summary
      - appeal_letter: letter/template when applicable
      - appeal_note: provider note for MANUAL_REVIEW or missing info
    """
    f = state.get("audit_findings", {}) or {}
    p_data = state.get("patient_data", {}) or {}
    verdict = str(f.get("verdict", "MANUAL_REVIEW")).upper()
    model_used = state.get("audit_model_flavor", "unknown")

    appeal_letter: str | None = None
    appeal_note: str | None = None
    final_status = "DENIED"

    bmi = None
    raw_bmi = f.get("bmi_numeric", None)
    try:
        if raw_bmi is not None:
            bmi = float(raw_bmi)
    except Exception:
        bmi = None

    evidence = str(f.get("evidence_quoted") or "").strip()
    # safety_flag = str(f.get("safety_flag", "CLEAR")).upper()  # Unused variable removed
    reasoning_src = str(f.get("reasoning") or "").strip()

    def with_bmi_prefix(text: str) -> str:
        if bmi is not None:
            return f"BMI {bmi:.2f}. {text}".strip()
        return text

    def _apply_letter_result(letter_res: Any) -> None:
        nonlocal appeal_letter, appeal_note
        if isinstance(letter_res, LetterResult):
            if letter_res.letter:
                appeal_letter = letter_res.letter
            if letter_res.note:
                appeal_note = letter_res.note
        elif isinstance(letter_res, str) and letter_res.strip():
            appeal_letter = letter_res

    if verdict == "APPROVED":
        final_status = "APPROVED"
        reason = with_bmi_prefix(reasoning_src or "Meets coverage criteria under policy.")

        # Generate a payer-ready PA request letter for approvals
        approval_reasoning = reasoning_src or reason
        approved_letter = generate_approved_letter(p_data, approval_reasoning, f)
        _apply_letter_result(approved_letter)

    elif verdict == "DENIED_SAFETY":
        final_status = "DENIED"
        base = reasoning_src or "Denied due to a documented safety exclusion per policy."
        reason = with_bmi_prefix(f"HARD STOP: Safety exclusion. {base}")

    elif verdict == "DENIED_MISSING_INFO":
        final_status = "PROVIDER_ACTION_REQUIRED"
        if bmi is None:
            reason = (
                "Provider action required. BMI is not documented and could not be calculated "
                "from recent height/weight. Please document a current BMI so this request can be processed."
            )
        else:
            reason = with_bmi_prefix(f"Provider action required. {reasoning_src}".strip())

        appeal_result = generate_appeal_letter(p_data, reason, f)
        _apply_letter_result(appeal_result)

    elif verdict == "DENIED_CLINICAL":
        final_status = "DENIED"
        base = reasoning_src or "Denied because BMI and/or qualifying comorbidities do not meet policy criteria."
        reason = with_bmi_prefix(f"Denied. {base}")

        # Detailed criteria list for Denial
        criteria_list = format_criteria_list(
            bmi,
            f.get("found_diagnosis_string"),
            f.get("found_e66_code"),
            f.get("found_z68_code"),
            evidence,
            f.get("missing_anchor_code"),
            f.get("ambiguity_code")
        )

        # For BMI 27–29.9 with non-qualifying/ambiguous evidence, generate documentation guidance
        # For BMI 27–29.9 with non-qualifying/ambiguous evidence, generate documentation guidance
        if bmi is not None and BMI_OVERWEIGHT_THRESHOLD <= bmi < BMI_OBESE_THRESHOLD:
             # Just use the template for consistency
             appeal_letter = f"""PRIOR AUTHORIZATION DENIAL NOTIFICATION
DENIAL REASON:
{reason}

{criteria_list}

APPEAL RIGHTS:
You may appeal this determination by submitting additional clinical documentation that addresses the criteria not met.
"""
        else:
             # Standard denial template
             appeal_letter = f"""PRIOR AUTHORIZATION DENIAL NOTIFICATION
DENIAL REASON:
{reason}

{criteria_list}
"""

    elif verdict == "MANUAL_REVIEW":
        final_status = "FLAGGED"
        base = reasoning_src or "Manual review required due to ambiguity per policy."
        detail = with_bmi_prefix(base)
        if evidence:
            detail += f" Evidence term: '{evidence}'."
        reason = detail.strip()

        # Detailed criteria list
        criteria_list = format_criteria_list(
            bmi,
            f.get("found_diagnosis_string"),
            f.get("found_e66_code"),
            f.get("found_z68_code"),
            evidence,
            f.get("missing_anchor_code"),
            f.get("ambiguity_code")
        )

        ambiguous_hit = evidence and any(term in (evidence.lower()) for term in (AMBIGUOUS_APPEAL_TERMS or []))
        if ambiguous_hit:
            appeal_note = (
                f"Manual review triggered by an ambiguous, non-qualifying term ('{evidence}'). "
                "If applicable, document a clearly qualifying comorbidity (HTN, T2DM, dyslipidemia, OSA, ASCVD) "
                "or clarify the diagnosis (e.g., specify OSA vs generic sleep apnea) before resubmission."
            )
        else:
            appeal_note = (
                "Manual review required due to borderline or ambiguous criteria. Verify BMI, qualifying comorbidities, "
                "and safety exclusions; document findings clearly before resubmission."
            )

        # Generate Template Letter with Criteria
        appeal_letter = f"""MANUAL REVIEW REQUIRED
Status: FLAGGED
Reason: {reason}

{criteria_list}
"""

    elif verdict == "CDI_REQUIRED":
        final_status = "CDI_REQUIRED"
        base = reasoning_src or "Clinical criteria met, but administrative Coding Documentation Integrity (CDI) required."
        reason = with_bmi_prefix(base)

        # Repurpose appeal_letter to carry the physician query text
        query_text = f.get("physician_query_text") or (
            "Physician Query: Please clarify diagnosis codes. "
            "Patient meets clinical criteria for weight management therapy, "
            "but requires specific ICD-10 anchor code (e.g., E66.9 Obesity) for payer compliance."
        )

        criteria_list = format_criteria_list(
            bmi,
            f.get("found_diagnosis_string"),
            f.get("found_e66_code"),
            f.get("found_z68_code"),
            evidence,
            f.get("missing_anchor_code"),
            None
        )

        appeal_letter = f"""PHYSICIAN QUERY / CDI ALERT
{query_text}

{criteria_list}
"""

    elif verdict == "SAFETY_SIGNAL_NEEDS_REVIEW":
        final_status = "SAFETY_SIGNAL_NEEDS_REVIEW"
        base = reasoning_src or "Safety signal detected requiring manual clinical review."
        reason = with_bmi_prefix(f"SAFETY SIGNAL: {base}")

        # Generate a safety warning note
        context = f.get("safety_context", "UNKNOWN")
        confidence = f.get("safety_confidence", "SIGNAL")
        exclusion_code = f.get("safety_exclusion_code", "GENERIC")

        appeal_note = (
            f"Safety Signal Detected ({context}/{confidence}). Code: {exclusion_code}. "
            f"Evidence: '{evidence}'. "
            "Verify patient history and clinical appropriateness before proceeding. "
            "This is not a hard denial, but a required safety checkpoint."
        )

    elif verdict in ("DENIED_BENEFIT_EXCLUSION", "DENIED_OTHER"):
        final_status = "DENIED"
        base = reasoning_src or f"Denied due to {verdict.replace('_', ' ').lower()}."
        reason = with_bmi_prefix(f"Denied. {base}")

    else:
        final_status = "FLAGGED"
        reason = f"Unknown verdict '{verdict}'. Route to manual utilization review."

    logger.info("Decision: %s (Reason: %s)", final_status, reason)

    # --- AUDIT LOGGING (CRITICAL) ---
    get_audit_logger().log_event(
        event_type="DECISION",
        details={
            "patient_id": p_data.get("patient_id"),
            "verdict": final_status,
            "raw_verdict": verdict,
            "reasoning": reason,
            "model_used": model_used,
            "policy_path": f.get("policy_path", "UNKNOWN"),
            "bmi": bmi,
        },
        patient_id=p_data.get("patient_id")
    )

    return {
        "final_decision": final_status,
        "reasoning": reason,
        "appeal_letter": appeal_letter,
        "appeal_note": appeal_note,
        "audit_model_flavor": model_used,
        "policy_path": f.get("policy_path"),
        "decision_type": f.get("decision_type"),
        "safety_exclusion_code": f.get("safety_exclusion_code"),
        "ambiguity_code": f.get("ambiguity_code"),
    }


# --- GRAPH BUILDER ---
def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("clinical_audit", clinical_audit)
    workflow.add_node("make_decision", make_decision)
    workflow.set_entry_point("retrieve_policy")
    workflow.add_edge("retrieve_policy", "clinical_audit")
    workflow.add_edge("clinical_audit", "make_decision")
    workflow.add_edge("make_decision", END)
    return workflow.compile()


# --- RUN SINGLE PATIENT TEST ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _ensure_data_loaded()
    app = build_agent()

    try:
        if df_meds is None or getattr(df_meds, "empty", True):
            raise RuntimeError("data_medications.csv not loaded (run etl_pipeline.py then chaos_monkey.py).")

        wegovy_rows = df_meds[df_meds["medication_name"].astype(str).str.contains("Wegovy", na=False)]
        if wegovy_rows.empty:
            raise RuntimeError("No Wegovy entries found in data_medications.csv.")

        target = str(wegovy_rows.iloc[0]["patient_id"])
        res = app.invoke({"patient_id": target, "drug_requested": "Wegovy"})
        logger.info("FINAL OUTPUT: %s | %s", res.get("final_decision"), res.get("reasoning"))
        if res.get("appeal_note"):
            logger.info("NOTE: %s", res["appeal_note"])
        if res.get("appeal_letter"):
            logger.info("LETTER/TEMPLATE:\n%s", res["appeal_letter"])
    except Exception as e:
        logger.error("Test run failed: %s", e)

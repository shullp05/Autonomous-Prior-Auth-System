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
import re
import time
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, TypedDict

import pandas as pd
import psutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# IMPORTANT: load .env BEFORE importing config (config may read env at import time)
load_dotenv()

logger = logging.getLogger(__name__)

# ---- Policy snapshot / deterministic engine (single source of truth) ----
from policy_constants import (  # noqa: E402
    AMBIGUOUS_APPEAL_TERMS,
    BMI_OBESE_THRESHOLD,
    BMI_OVERWEIGHT_THRESHOLD,
)
from policy_engine import _parse_bmi, evaluate_eligibility  # noqa: E402
from policy_snapshot import SNAPSHOT_PATH, load_policy_snapshot  # noqa: E402
from schema_validation import validate_policy_snapshot  # noqa: E402

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
    POLICY_ID as ACTIVE_POLICY_ID,
    PA_PROVIDER_NAME,
    PA_PROVIDER_NPI,
    PA_PRACTICE_NAME,
)

# ---- Optional reranker (safe-to-import wrapper) ----
# This module is designed to not blow up imports if BCEmbedding isn't installed.
from bce_reranker import rerank_bce  # noqa: E402

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

ENABLE_RERANK = os.getenv("PA_ENABLE_RERANK", "true").lower() == "true"
PA_ENABLE_RERANK = ENABLE_RERANK  # backward-compat alias

# RAG scoring knobs: config defaults, env override supported
RAG_SCORE_FLOOR = float(os.getenv("PA_RAG_SCORE_FLOOR", str(PA_RAG_SCORE_FLOOR)))
RAG_MIN_DOCS = int(os.getenv("PA_RAG_MIN_DOCS", str(PA_RAG_MIN_DOCS)))

SNAPSHOT = load_policy_snapshot(SNAPSHOT_PATH, ACTIVE_POLICY_ID)
validate_policy_snapshot(SNAPSHOT)

# --- DATA LOADING (lazy) ---
df_patients: Optional[pd.DataFrame] = None
df_meds: Optional[pd.DataFrame] = None
df_conditions: Optional[pd.DataFrame] = None
df_obs: Optional[pd.DataFrame] = None

_DATA_DIR = Path(os.getenv("ETL_OUTPUT_DIR", "output"))
_PAT_PATH = _DATA_DIR / "data_patients.csv"
_MED_PATH = _DATA_DIR / "data_medications.csv"
_COND_PATH = _DATA_DIR / "data_conditions.csv"
_OBS_PATH = _DATA_DIR / "data_observations.csv"


def _coerce_pid(x: Any) -> str:
    return str(x).strip()


def _clean_str_list(values: Any) -> List[str]:
    if values is None:
        return []
    out: List[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        out.append(s)
    return out


def _ensure_data_loaded() -> None:
    """Lazy load data to allow module import without files present."""
    global df_patients, df_meds, df_conditions, df_obs
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

        # CRITICAL: normalize patient_id to string across all frames for consistent joins/filters
        for df in (df_patients, df_meds, df_conditions, df_obs):
            if df is not None and not getattr(df, "empty", True) and "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].astype(str)

        # Normalize observation type casing if present (defensive)
        if df_obs is not None and not getattr(df_obs, "empty", True) and "type" in df_obs.columns:
            df_obs["type"] = df_obs["type"].astype(str)

    except Exception as e:
        logger.error("Data Load Error: %s", e)


def write_model_trace(model_name: str, role: str, params: dict, required_ram_gb: Optional[float] = None) -> None:
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
    options: Optional[dict] = None,
) -> ChatOllama:
    """
    Create ChatOllama instance with optional format and low-level tuning.
    """
    if options is None:
        options = {}
    kwargs = {"model": model, "temperature": temperature, **options}
    if prefer_json:
        kwargs["format"] = "json"
    try:
        return ChatOllama(**kwargs)
    except TypeError:
        # Legacy fallback if older langchain_ollama doesn't accept `format`
        kwargs.pop("format", None)
        return ChatOllama(**kwargs)


# --- BMI CALCULATION ---
def calculate_bmi_if_missing(patient_id: str, df_obs_in: Optional[pd.DataFrame]) -> str:
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
    if p_obs.empty:
        return "MISSING_DATA"

    if "date" in p_obs.columns:
        p_obs["date_parsed"] = pd.to_datetime(p_obs["date"], errors="coerce")
    else:
        p_obs["date_parsed"] = pd.NaT

    # 1) Explicit BMI
    bmi_rows = p_obs[p_obs["type"] == "BMI"].sort_values("date_parsed", ascending=False)
    if not bmi_rows.empty:
        try:
            v = float(bmi_rows.iloc[0]["value"])
            return f"{round(v, 1)} (Source: EMR)"
        except Exception:
            return "MISSING_DATA"

    # 2) Calculate from height/weight
    try:
        wt = p_obs[p_obs["type"] == "Weight"].sort_values("date_parsed", ascending=False)
        ht = p_obs[p_obs["type"] == "Height"].sort_values("date_parsed", ascending=False)

        if wt.empty or ht.empty:
            return "MISSING_DATA"

        weight_kg = float(wt.iloc[0]["value"])
        height_cm = float(ht.iloc[0]["value"])

        height_m = height_cm / 100.0
        if height_m <= 0:
            return "MISSING_DATA"

        calculated_bmi = round(weight_kg / (height_m**2), 1)
        return f"{calculated_bmi} (Calculated)"
    except Exception:
        return "MISSING_DATA"


# --- PATIENT LOOKUP ---
def look_up_patient_data(patient_id: str) -> Optional[dict]:
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

    meds: List[str] = []
    if df_meds is not None and not getattr(df_meds, "empty", True):
        meds_series = df_meds[df_meds["patient_id"].astype(str) == pid]["medication_name"]
        meds = _clean_str_list(meds_series.dropna().astype(str).tolist())

    conds: List[str] = []
    if df_conditions is not None and not getattr(df_conditions, "empty", True):
        cond_series = df_conditions[df_conditions["patient_id"].astype(str) == pid]["condition_name"]
        conds = _clean_str_list(cond_series.dropna().astype(str).tolist())

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


class ProviderContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    provider_credentials: str = ""
    practice_name: str
    npi: str
    phone: str
    fax: str
    address: str = ""


class PARequestLetterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: str
    patient_dob: str
    patient_id: str
    drug_name: str = "Wegovy (semaglutide)"
    strength: str = "2.4 mg"
    route: str = "subcutaneous"
    frequency: str = "weekly"
    indication: str = "Chronic weight management"

    bmi_value: float
    bmi_date: str = ""
    qualifying_pathway: Literal["BMI_30_PLUS", "BMI_27_29_WITH_COMORBIDITY"]
    qualifying_comorbidity: str = ""
    contraindications_checked: List[str] = Field(default_factory=list)
    contraindications_found: List[str] = Field(default_factory=list)

    requested_action: str = "Request prior authorization coverage for Wegovy as prescribed"

    attachments: List[str] = Field(default_factory=list)


class PARequestLetterDraft(BaseModel):
    """What the LLM must return (JSON only)."""
    model_config = ConfigDict(extra="forbid")

    recipient_org: str
    recipient_department: str
    attention_line: str

    subject_line: str

    opening_paragraph: str
    clinical_summary_bullets: List[str]
    criteria_bullets: List[str]
    safety_paragraph: str
    requested_action_paragraph: str
    attachments_bullets: List[str]


def _require_provider_context() -> ProviderContext:
    """
    Hard fail if provider/practice identifiers are missing.
    No placeholders, no guessing.
    """
    ctx = ProviderContext(
        provider_name=os.getenv("PA_PROVIDER_NAME", "").strip(),
        provider_credentials=os.getenv("PA_PROVIDER_CREDENTIALS", "").strip(),
        practice_name=os.getenv("PA_PRACTICE_NAME", "").strip(),
        npi=os.getenv("PA_PROVIDER_NPI", "").strip(),
        phone=os.getenv("PA_PRACTICE_PHONE", "").strip(),
        fax=os.getenv("PA_PRACTICE_FAX", "").strip(),
        address=os.getenv("PA_PRACTICE_ADDRESS", "").strip(),
    )
    missing = [
        k
        for k, v in ctx.model_dump().items()
        if k in ("provider_name", "practice_name", "npi", "phone", "fax") and not v
    ]
    if missing:
        raise RuntimeError(f"Provider context missing required fields: {missing}. Set PA_PROVIDER_* env vars.")
    return ctx


def _guard_letter_text(text: str) -> None:
    """
    Reject dangerous/incorrect language in provider/payer-facing letters.
    """
    bad_phrases = [
        "Dr. AI",
    ]
    lowered = (text or "").lower()
    if "needappeal" in lowered:
        raise ValueError("Letter contains 'needappeal' language; incorrect for PA request.")
    for p in bad_phrases:
        if p.lower() in lowered:
            raise ValueError(f"Letter contains prohibited phrase: {p!r}")


# --- AMBIGUITY CLARIFICATION HELPER ---
def _get_ambiguity_clarification(evidence: str) -> str:
    """Generate specific clarification guidance based on the ambiguous term."""
    ev_lower = (evidence or "").lower()

    if "prediabetes" in ev_lower or "pre-diabetes" in ev_lower or "borderline diabetes" in ev_lower or "impaired fasting" in ev_lower:
        return """CLARIFICATION NEEDED:
The term "prediabetes" (or similar) does NOT qualify as a weight-related comorbidity for Wegovy coverage.

To support approval, document ONE of the following (if present):
- Type 2 Diabetes Mellitus
- Hypertension / High Blood Pressure
- Dyslipidemia / Hyperlipidemia
- Obstructive Sleep Apnea (OSA)
- Cardiovascular Disease (ASCVD)

If the patient has any of these, update the chart to clearly document it."""

    if "sleep apnea" in ev_lower and "obstructive" not in ev_lower:
        return """CLARIFICATION NEEDED:
Generic "sleep apnea" does not qualify—documentation must specify "Obstructive Sleep Apnea (OSA)".

To support approval:
- Confirm diagnosis is obstructive (not central/mixed)
- Update chart to clearly state "Obstructive Sleep Apnea" or "OSA"

Alternatively, document another qualifying comorbidity (HTN, T2DM, dyslipidemia, CVD)."""

    if "thyroid" in ev_lower:
        return """CLARIFICATION NEEDED:
Thyroid terminology requires clarification for safety determination.

- Medullary Thyroid Carcinoma (MTC) is a contraindication for Wegovy
- Other thyroid cancers (papillary/follicular) are not contraindications per policy

Please clarify the specific thyroid diagnosis/history to determine Wegovy safety."""

    if "blood pressure" in ev_lower or "borderline hypertension" in ev_lower:
        return """CLARIFICATION NEEDED:
"Elevated blood pressure" or "borderline hypertension" may not meet criteria for a qualifying comorbidity.

To support approval:
- Confirm documented Hypertension / HTN requiring treatment
- Update the problem list to clearly state "Hypertension" or "HTN"

Alternatively, document another qualifying comorbidity (T2DM, dyslipidemia, OSA, CVD)."""

    return f"""CLARIFICATION NEEDED:
The term "{evidence}" requires clarification before this PA can be processed.

Please provide documentation of a qualifying weight-related comorbidity:
- Hypertension
- Type 2 Diabetes Mellitus
- Dyslipidemia
- Obstructive Sleep Apnea (OSA)
- Cardiovascular Disease"""


# --- APPEAL / CLARIFICATION GENERATOR ---
def generate_appeal_letter(patient_data: dict, denial_reason: str, findings: dict) -> Optional[str]:
    """
    Generate a provider-facing clarification / documentation letter for DENIED/FLAGGED cases.
    Returns None for safety denials or if generation fails (caller may fall back to templates).
    """
    if not isinstance(patient_data, dict):
        return None

    # Don't generate for safety hard-stops
    verdict = str((findings or {}).get("verdict", "")).upper()
    if verdict == "DENIED_SAFETY":
        return None

    patient_name = str(patient_data.get("name", "")).strip()
    patient_dob = str(patient_data.get("dob", "")).strip()
    if not patient_name or not patient_dob:
        return None  # no placeholders for provider-facing drafting

    bmi_value = (findings or {}).get("bmi_numeric")
    evidence = str((findings or {}).get("evidence_quoted", "")).strip()

    is_flagged_case = verdict == "MANUAL_REVIEW" or "ambiguous" in (denial_reason or "").lower() or "flagged" in (denial_reason or "").lower()

    llm = _make_llm(model=APPEAL_MODEL, temperature=0.2, prefer_json=False)

    try:
        if is_flagged_case and evidence:
            clarification_guidance = _get_ambiguity_clarification(evidence)
            prompt = f"""You are a board-certified Clinical Pharmacist drafting a prior authorization clarification request.

PATIENT: {patient_name}, DOB: {patient_dob}
CURRENT BMI: {bmi_value if bmi_value is not None else "REQUIRES VERIFICATION"} kg/m²

ISSUE: This prior authorization was flagged for manual review due to ambiguous terminology.
AMBIGUOUS TERM FOUND: "{evidence}"

{clarification_guidance}

INSTRUCTIONS:
Write a brief, focused letter that:
1. States the medication (Wegovy/semaglutide) and indication (chronic weight management)
2. Notes the patient's BMI (or requests a current BMI if not documented)
3. Explains ONLY why the specific term "{evidence}" requires clarification
4. States exactly what documentation or clarification is needed
5. Does NOT list unrelated medical conditions

Do NOT include markdown. Do NOT include placeholders in brackets.
Use direct language appropriate for a clinical document."""
        else:
            prompt = f"""You are a board-certified Clinical Pharmacist drafting a prior authorization documentation guidance letter.

PATIENT: {patient_name}, DOB: {patient_dob}
BMI: {bmi_value if bmi_value is not None else "REQUIRES VERIFICATION"} kg/m²

CURRENT STATUS:
{denial_reason}

INSTRUCTIONS:
Write a professional letter that:
1. States the medication being requested (Wegovy/semaglutide) and indication
2. Summarizes only clinically relevant information
3. Identifies what specific documentation is needed to meet criteria
4. Closes professionally

Do NOT include markdown. Do NOT include placeholders in brackets."""

        resp = llm.invoke(prompt)

        write_model_trace(
            model_name=APPEAL_MODEL,
            role="appeal_generator",
            params={"temperature": 0.2},
            required_ram_gb=AUDIT_MODEL_RAM,
        )

        text = str(resp.content or "").strip()

        # Handle accidental JSON wrapper
        if text.startswith("{") and "letter" in text[:80].lower():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("letter"), str):
                    text = parsed["letter"].strip()
            except Exception:
                pass

        if "\\n" in text:
            text = text.replace("\\n", "\n")

        if not text or len(text) < 120:
            return None

        # Must not contain "appeal"
        if "appeal" in text.lower():
            return None

        return text

    except Exception as e:
        logger.error("Error generating appeal/clarification letter: %s", e)
        return None


def _generate_fallback_pa_template(patient_data: dict, reason: str, findings: dict) -> str:
    """Generate a clinically-focused PA template when the LLM cannot produce one."""
    name = str((patient_data or {}).get("name", "")).strip() or "Patient"
    dob = str((patient_data or {}).get("dob", "")).strip() or ""
    bmi = (findings or {}).get("bmi_numeric")
    evidence = str((findings or {}).get("evidence_quoted", "")).strip()
    verdict = str((findings or {}).get("verdict", "")).upper()

    is_ambiguity_case = verdict == "MANUAL_REVIEW" or "ambiguous" in (reason or "").lower() or "flagged" in (reason or "").lower()

    bmi_text = "Current BMI: REQUIRES VERIFICATION"
    bmi_analysis = "BMI could not be reliably extracted. Please verify from chart."
    if bmi is not None:
        try:
            bmi_val = float(bmi)
            bmi_text = f"Current BMI: {bmi_val:.1f} kg/m²"
            if bmi_val >= 30:
                bmi_analysis = "Patient meets BMI threshold (≥30) for obesity pathway."
            elif bmi_val >= 27:
                bmi_analysis = "Patient meets BMI threshold (≥27) but requires documented qualifying comorbidity."
            else:
                bmi_analysis = "Patient BMI is below coverage threshold (<27). Coverage unlikely without exceptional circumstances."
        except Exception:
            pass

    if is_ambiguity_case and evidence:
        cond_text = f'FLAGGED TERM REQUIRING CLARIFICATION: "{evidence}"'
    else:
        cond_text = "Review chart for qualifying comorbidity (HTN, T2DM, dyslipidemia, OSA, ASCVD) if BMI is 27–29.9."

    template = f"""PRIOR AUTHORIZATION DOCUMENTATION TEMPLATE — WEGOVY (SEMAGLUTIDE)
Indication: Chronic weight management

Patient: {name}
Date of Birth: {dob}

CLINICAL PROFILE:
{bmi_text}
Assessment: {bmi_analysis}

{cond_text}

CASE STATUS:
{reason}

NEXT STEPS:
1) Ensure BMI is current and documented
2) If BMI 27–29.9, document a clearly qualifying comorbidity (HTN, T2DM, dyslipidemia, OSA, ASCVD)
3) Review for safety exclusions (MTC/MEN2 history, pregnancy/lactation, concurrent GLP-1/GLP-1-GIP)

_____________________________________________
Provider Signature / Date
"""
    return template


# --- APPROVED LETTER GENERATOR ---
def generate_approved_letter(patient_data: dict, approval_reasoning: str, findings: dict) -> Optional[str]:
    """
    Generate a payer-ready Prior Authorization Request / Letter of Medical Necessity
    for cases that meet criteria.

    Output must be suitable for provider review/signature. No AI identity. No "appeal".
    """
    if not isinstance(patient_data, dict):
        return None

    try:
        provider_ctx = _require_provider_context()
    except Exception as e:
        logger.error("Provider context not configured: %s", e)
        return _generate_fallback_approved_letter(patient_data, approval_reasoning, findings)

    patient_name = str(patient_data.get("name", "")).strip()
    patient_dob = str(patient_data.get("dob", "")).strip()
    patient_id = str(patient_data.get("patient_id", "")).strip()

    if not patient_name or not patient_dob or not patient_id:
        return None

    bmi_value = (findings or {}).get("bmi_numeric")
    if bmi_value is None:
        return None

    try:
        bmi_value_f = float(bmi_value)
    except Exception:
        return None

    comorbidity = str((findings or {}).get("comorbidity_category", "NONE") or "NONE").upper()

    if bmi_value_f >= 30.0:
        pathway = "BMI_30_PLUS"
        qualifying_comorbidity = ""
    else:
        pathway = "BMI_27_29_WITH_COMORBIDITY"
        qualifying_comorbidity = (
            "Hypertension" if comorbidity == "HYPERTENSION" else ("Type 2 Diabetes Mellitus" if comorbidity == "DIABETES" else comorbidity.title())
        )

    letter_input = PARequestLetterInput(
        patient_name=patient_name,
        patient_dob=patient_dob,
        patient_id=patient_id,
        bmi_value=bmi_value_f,
        qualifying_pathway=pathway,
        qualifying_comorbidity=qualifying_comorbidity,
        contraindications_checked=[
            "pregnancy/lactation",
            "MTC/MEN2 (personal/family history)",
            "concurrent GLP-1/GLP-1-GIP therapy",
            "pancreatitis history (if documented)",
        ],
        contraindications_found=[],
        attachments=[
            "Most recent vitals or BMI documentation",
            "Problem list reflecting qualifying comorbidity (if applicable)",
            "Medication list",
        ],
    )

    llm = _make_llm(model=APPEAL_MODEL, temperature=0.0, prefer_json=True)

    system = """You draft a payer-ready Prior Authorization Request / Letter of Medical Necessity for a PCP office.
HARD RULES:
- Draft for provider review and signature. Do not mention AI, automation, models, or internal systems.
- Do NOT use the word "appeal". This is an initial PA request.
- Do NOT say "approved" or "we approved". The provider is requesting authorization.
- Do NOT invent facts, labs, diagnoses, prior therapies, dates, or contraindications. Use only the provided JSON.
- Safety language must be non-absolute. Use: "No contraindications were identified in the reviewed record" if none found.
- Output MUST be a single valid JSON object matching the schema. No markdown. No extra text.

STYLE:
- Professional, concise, clinical-administrative tone.
- One-page structure. Bullets where appropriate.

OUTPUT JSON SCHEMA (exact keys):
{
  "recipient_org": "...",
  "recipient_department": "...",
  "attention_line": "...",
  "subject_line": "...",
  "opening_paragraph": "...",
  "clinical_summary_bullets": ["..."],
  "criteria_bullets": ["..."],
  "safety_paragraph": "...",
  "requested_action_paragraph": "...",
  "attachments_bullets": ["..."]
}
"""

    user = {
        "provider_context": provider_ctx.model_dump(),
        "letter_input": letter_input.model_dump(),
        "approval_reasoning": approval_reasoning,
    }

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
            ]
        )

        write_model_trace(
            model_name=APPEAL_MODEL,
            role="approved_letter_generator",
            params={"temperature": 0.0, "format": "json"},
            required_ram_gb=AUDIT_MODEL_RAM,
        )

        raw = str(resp.content or "").strip()
        obj = _extract_json_object(raw)
        draft = PARequestLetterDraft(**obj)

        date_line = time.strftime("%Y-%m-%d")
        provider_line = provider_ctx.provider_name + (f", {provider_ctx.provider_credentials}" if provider_ctx.provider_credentials else "")
        practice_block = "\n".join(
            [provider_line, provider_ctx.practice_name, f"NPI: {provider_ctx.npi}", f"Phone: {provider_ctx.phone}  Fax: {provider_ctx.fax}"]
            + ([provider_ctx.address] if provider_ctx.address else [])
        )

        recipient_block = "\n".join([draft.recipient_department, f"Attn: {draft.attention_line}", draft.recipient_org])

        def bullets(items: List[str]) -> str:
            return "\n".join([f"- {str(i).strip()}" for i in items if str(i).strip()])

        letter_text = f"""{practice_block}

{date_line}

{recipient_block}

Subject: {draft.subject_line}

Re: {letter_input.patient_name} (DOB: {letter_input.patient_dob}) | Patient ID: {letter_input.patient_id}

Dear Medical Director,

{draft.opening_paragraph}

Clinical Summary:
{bullets(draft.clinical_summary_bullets)}

Medical Necessity & Coverage Criteria:
{bullets(draft.criteria_bullets)}

Safety Review:
{draft.safety_paragraph}

Requested Action:
{draft.requested_action_paragraph}

Attachments:
{bullets(draft.attachments_bullets)}

Sincerely,

______________________________
{provider_line}
{provider_ctx.practice_name}
"""

        _guard_letter_text(letter_text)
        return letter_text

    except Exception as e:
        logger.error("Error generating approved letter (structured): %s", e)
        return _generate_fallback_approved_letter(patient_data, approval_reasoning, findings)


def _generate_fallback_approved_letter(patient_data: dict, approval_reasoning: str, findings: dict) -> str:
    """Generate a simple approval-request letter template when LLM fails or provider context missing."""
    name = str((patient_data or {}).get("name", "")).strip() or "Patient"
    dob = str((patient_data or {}).get("dob", "")).strip() or ""
    patient_id = str((patient_data or {}).get("patient_id", "")).strip()
    bmi = (findings or {}).get("bmi_numeric")
    comorbidity = str((findings or {}).get("comorbidity_category", "NONE") or "NONE")

    bmi_text = "[See chart]"
    try:
        if bmi is not None:
            bmi_text = f"{float(bmi):.1f} kg/m²"
    except Exception:
        pass

    template = f"""LETTER OF MEDICAL NECESSITY
Prior Authorization Request — Wegovy (Semaglutide)

To: Medical Director, Utilization Management

RE: {name}
DOB: {dob}
Patient ID: {patient_id}

Dear Medical Director,

I am submitting this request for prior authorization coverage of Wegovy (semaglutide) for chronic weight management for the above-referenced patient.

CLINICAL JUSTIFICATION:
{approval_reasoning}

Current BMI: {bmi_text}
{f"Qualifying Comorbidity: {comorbidity}" if comorbidity and comorbidity.upper() != "NONE" else ""}

I respectfully request authorization coverage consistent with the applicable policy criteria.

Sincerely,

_______________________________
Prescriber Signature / Date
"""
    return template


def _build_policy_summary(snapshot: dict) -> str:
    lines = [
        f"{snapshot['title']} (Policy ID: {snapshot['policy_id']}, Effective {snapshot['effective_date']})",
        f"Scope: {snapshot['scope']} (excluded: {', '.join(snapshot['excluded_scopes'])})",
        "Eligibility:",
    ]
    for pathway in snapshot["eligibility"]["pathways"]:
        bmi_min = pathway["bmi_min"]
        bmi_max = pathway.get("bmi_max")
        bmi_clause = f"BMI ≥ {bmi_min}" if bmi_max is None else f"BMI ≥ {bmi_min} and < {bmi_max}"
        diag_note = "requires documented diagnosis strings"
        line = f"- {pathway['name']}: {bmi_clause}; {diag_note}"
        if pathway.get("required_comorbidity_categories"):
            labels = [
                snapshot["comorbidities"][key]["label"]
                for key in pathway["required_comorbidity_categories"]
                if key in snapshot["comorbidities"]
            ]
            if labels:
                line = f"{line}; comorbidity: {', '.join(labels)}"
        lines.append(line)
    lines.append("Safety exclusions:")
    for exclusion in snapshot["safety_exclusions"]:
        lines.append(f"- {exclusion['category']}")
    lines.append("No concurrent GLP-1 / GLP-1-GIP agents: " + ", ".join(snapshot["drug_conflicts"]["glp1_or_glp1_gip_agents"]))
    return "\n".join(lines)


# --- STATIC POLICY FALLBACK (used when RAG unavailable) ---
POLICY_SUMMARY_TEXT = _build_policy_summary(SNAPSHOT)
_STATIC_POLICY_FALLBACK = POLICY_SUMMARY_TEXT


def _apply_score_floor(
    scored_docs: List[Tuple[Document, float]],
    floor: float,
    min_docs: int,
) -> Tuple[List[Document], List[float]]:
    """
    Filter by BCE score floor while enforcing a minimum number of documents.

    Returns (filtered_docs, filtered_scores) in the same order.
    """
    if not scored_docs:
        return [], []

    docs, scores = zip(*scored_docs)
    docs = list(docs)
    scores = [float(s) for s in scores]

    filtered_docs: List[Document] = []
    filtered_scores: List[float] = []
    for d, s in zip(docs, scores):
        if s >= floor:
            filtered_docs.append(d)
            filtered_scores.append(s)

    if len(filtered_docs) < min_docs:
        filtered_docs = docs[:min_docs]
        filtered_scores = scores[:min_docs]

    return filtered_docs, filtered_scores


def _policy_bucket(section: str, policy_path: Optional[str]) -> int:
    """
    Assign a coarse priority bucket to a policy section.
    Lower number = higher priority for LLM evidence.
    """
    sec = (section or "").lower()
    path = (policy_path or "").upper() if policy_path else ""

    if path == "SAFETY_EXCLUSION":
        if sec.startswith("drug_conflicts:"):
            return 0
        if sec.startswith("safety_exclusions:"):
            return 1
        if sec.startswith("documentation:"):
            return 3
        return 2

    if path == "BMI30_OBESITY":
        if sec.startswith("documentation:"):
            return 0
        if sec == "eligibility:pathway1":
            return 1
        if sec == "diagnosis:obesity_strings":
            return 2
        if sec.startswith("comorbidity:"):
            return 3
        if sec.startswith("safety_exclusions:") or sec.startswith("drug_conflicts:"):
            return 4
        return 5

    if path == "BMI27_COMORBIDITY":
        if sec.startswith("documentation:"):
            return 0
        if sec == "eligibility:pathway2":
            return 1
        if sec.startswith("comorbidity:"):
            return 2
        if sec == "diagnosis:overweight_strings":
            return 3
        if sec.startswith("safety_exclusions:") or sec.startswith("drug_conflicts:"):
            return 4
        return 5

    if sec.startswith("documentation:"):
        return 0
    if sec.startswith("eligibility:"):
        return 1
    if sec.startswith("comorbidity:") or sec.startswith("diagnosis:"):
        return 2
    if sec.startswith("safety_exclusions:") or sec.startswith("drug_conflicts:"):
        return 3
    if sec.startswith("ambiguity:"):
        return 4
    return 5


def _policy_aware_sort_docs(docs: List[Document], scores: List[float], policy_path: Optional[str]) -> List[Document]:
    """
    Reorder documents so the most policy-relevant sections for the current
    deterministic policy path appear first, while respecting BCE scores
    within each bucket.
    """
    if not docs:
        return []

    indexed: List[Tuple[int, float, int, Document]] = []
    for idx, (d, s) in enumerate(zip(docs, scores)):
        section = (d.metadata or {}).get("section") or ""
        bucket = _policy_bucket(str(section), policy_path)
        indexed.append((bucket, -float(s), idx, d))

    indexed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in indexed]


# --- RETRIEVAL NODE HELPERS ---
def _build_policy_query(det_result: Any, patient_data: Optional[dict], drug: str) -> str:
    drug = drug or "Wegovy"
    if det_result:
        bmi = getattr(det_result, "bmi_numeric", None)
        path = getattr(det_result, "policy_path", "UNKNOWN")
        verdict = getattr(det_result, "verdict", "UNKNOWN")
        category = getattr(det_result, "comorbidity_category", "NONE")
        safety = getattr(det_result, "safety_flag", "CLEAR")
        return (
            f"{drug} prior authorization policy evidence for {path} with verdict {verdict}; "
            f"BMI {bmi}; comorbidity {category}; safety {safety}. "
            "Return diagnosis strings, comorbidity rules, safety exclusions, and documentation requirements."
        )
    bmi_hint = (patient_data or {}).get("latest_bmi") if patient_data else "unknown"
    return (
        f"{drug} prior authorization eligibility criteria and safety exclusions. "
        f"Patient BMI hint: {bmi_hint}. Include ambiguity handling and documentation requirements."
    )


def _format_policy_evidence(docs: List[Document]) -> str:
    if not docs:
        return _STATIC_POLICY_FALLBACK
    blocks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        section = (doc.metadata or {}).get("section", "unknown")
        blocks.append(f"=== POLICY EVIDENCE {idx} ===\n[section: {section}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def retrieve_policy(state: AgentState) -> dict:
    """
    Retrieve the active Wegovy policy from ChromaDB vector store if available,
    otherwise fall back to static policy text.
    """
    drug = state.get("drug_requested", "Wegovy")
    logger.info("[RAG] Retrieving policy for %s", drug)

    patient_id = state.get("patient_id")
    patient_data = state.get("patient_data") or (look_up_patient_data(patient_id) if patient_id else None)

    det_result = evaluate_eligibility(patient_data) if patient_data else None

    policy_text: Optional[str] = None
    policy_docs: List[Document] = []

    if os.path.exists("./chroma_db"):
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings

            embedding_fn = OllamaEmbeddings(model=EMBED_MODEL)
            vectordb = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embedding_fn,
                collection_name="priorauth_policies",
            )

            query = _build_policy_query(det_result, patient_data, drug)

            t0 = time.perf_counter()
            vector_docs: List[Document] = vectordb.similarity_search(
                query,
                k=PA_RAG_K_VECTOR,
                filter={"policy_id": str(ACTIVE_POLICY_ID)},
            )
            vector_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("[RAG] Vector search returned %d docs (%.1f ms)", len(vector_docs), vector_ms)

            if vector_docs:
                if ENABLE_RERANK and len(vector_docs) > 1:
                    try:
                        t1 = time.perf_counter()
                        scored = rerank_bce(query, vector_docs)
                        rerank_ms = (time.perf_counter() - t1) * 1000.0
                        logger.info(
                            "[RAG] BCE reranked %d docs (%.1f ms). Top 5 scores: %s",
                            len(scored),
                            rerank_ms,
                            [f"{s:.3f}" for _, s in scored[:5]],
                        )

                        filtered_docs, filtered_scores = _apply_score_floor(
                            scored_docs=scored,
                            floor=RAG_SCORE_FLOOR,
                            min_docs=RAG_MIN_DOCS,
                        )

                        policy_path = getattr(det_result, "policy_path", None) if det_result else None
                        filtered_docs = _policy_aware_sort_docs(filtered_docs, filtered_scores, policy_path=policy_path)

                        policy_docs = filtered_docs[:PA_RAG_TOP_K_DOCS]
                    except Exception as e:
                        logger.warning("[RAG] Rerank failed; falling back to vector order: %s", e)
                        policy_docs = vector_docs[:PA_RAG_TOP_K_DOCS]
                else:
                    policy_docs = vector_docs[:PA_RAG_TOP_K_DOCS]

                policy_text = _format_policy_evidence(policy_docs)
                logger.info("[RAG] Using %d policy atoms for LLM (%d chars)", len(policy_docs), len(policy_text))
            else:
                logger.warning("[RAG] ChromaDB returned no results, using fallback")

        except ImportError as e:
            logger.warning("[RAG] ChromaDB/embeddings not available: %s", e)
        except Exception as e:
            logger.warning("[RAG] Vector retrieval failed: %s", e)

    if not policy_text:
        policy_text = _STATIC_POLICY_FALLBACK
        logger.info("[RAG] Using static policy fallback")

    write_model_trace(
        model_name=EMBED_MODEL,
        role="policy_retrieval_embed",
        params={
            "pa_enable_rerank": ENABLE_RERANK,
            "pa_rerank_model": RERANK_MODEL if ENABLE_RERANK else None,
            "pa_rerank_device": os.getenv("PA_RERANK_DEVICE", RERANK_DEVICE_DEFAULT),
            "k_vector": PA_RAG_K_VECTOR,
            "top_k_docs": PA_RAG_TOP_K_DOCS,
            "score_floor": RAG_SCORE_FLOOR,
            "min_docs": RAG_MIN_DOCS,
        },
        required_ram_gb=4,
    )

    return {
        "policy_text": policy_text,
        "policy_docs": policy_docs,
        "patient_data": patient_data,
        "deterministic_decision": det_result.to_dict() if det_result else None,
    }


# --- SMALL HELPER: ROBUST JSON EXTRACTION FROM LLM OUTPUT ---
def _extract_json_object(text: str) -> dict:
    """Try very hard to extract a single JSON object from a model response."""
    raw = str(text or "").strip()

    if "```json" in raw:
        raw = raw.split("```json", 1)[1]
        if "```" in raw:
            raw = raw.split("```", 1)[0]
        raw = raw.strip()
    elif "```" in raw:
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
    candidate = raw[first_brace : last_brace + 1].strip() if (first_brace != -1 and last_brace != -1 and last_brace > first_brace) else raw

    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Parsed JSON is not an object.", candidate, 0)
    return obj


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
    ] = "MANUAL_REVIEW"
    reasoning: str = ""


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

    llm_audit_dict: Optional[dict] = None
    try:
        det_bmi = _parse_bmi(p_data.get("latest_bmi"))
        det_bmi_str = str(det_bmi) if det_bmi is not None else "null"

        llm = _make_llm(model=AUDIT_MODEL, temperature=0, prefer_json=True, options=AUDIT_MODEL_OPTS)

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
                "verdict": "APPROVED|DENIED_SAFETY|DENIED_CLINICAL|DENIED_MISSING_INFO|MANUAL_REVIEW|DENIED_BENEFIT_EXCLUSION|DENIED_OTHER",
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

    appeal_letter: Optional[str] = None
    appeal_note: Optional[str] = None
    final_status = "DENIED"

    bmi = None
    raw_bmi = f.get("bmi_numeric", None)
    try:
        if raw_bmi is not None:
            bmi = float(raw_bmi)
    except Exception:
        bmi = None

    evidence = str(f.get("evidence_quoted") or "").strip()
    safety_flag = str(f.get("safety_flag", "CLEAR")).upper()
    reasoning_src = str(f.get("reasoning") or "").strip()

    def with_bmi_prefix(text: str) -> str:
        if bmi is not None:
            return f"BMI {bmi:.2f}. {text}".strip()
        return text

    if verdict == "APPROVED":
        final_status = "APPROVED"
        reason = with_bmi_prefix(reasoning_src or "Meets coverage criteria under policy.")

        # Generate a payer-ready PA request letter for approvals
        approval_reasoning = reasoning_src or reason
        approved_letter = generate_approved_letter(p_data, approval_reasoning, f)
        if approved_letter:
            appeal_letter = approved_letter  # keep key name for backward compatibility

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

        appeal_letter = generate_appeal_letter(p_data, reason, f) or _generate_fallback_pa_template(p_data, reason, f)

    elif verdict == "DENIED_CLINICAL":
        final_status = "DENIED"
        base = reasoning_src or "Denied because BMI and/or qualifying comorbidities do not meet policy criteria."
        reason = with_bmi_prefix(f"Denied. {base}")

        # For BMI 27–29.9 with non-qualifying/ambiguous evidence, generate documentation guidance
        if bmi is not None and BMI_OVERWEIGHT_THRESHOLD <= bmi < BMI_OBESE_THRESHOLD:
            appeal_letter = generate_appeal_letter(p_data, reason, f)

    elif verdict == "MANUAL_REVIEW":
        final_status = "FLAGGED"
        base = reasoning_src or "Manual review required due to ambiguity per policy."
        detail = with_bmi_prefix(base)
        if evidence:
            detail += f" Evidence term: '{evidence}'."
        reason = detail.strip()

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

        appeal_letter = generate_appeal_letter(p_data, reason, f) or _generate_fallback_pa_template(p_data, reason, f)

    elif verdict in ("DENIED_BENEFIT_EXCLUSION", "DENIED_OTHER"):
        final_status = "DENIED"
        base = reasoning_src or f"Denied due to {verdict.replace('_', ' ').lower()}."
        reason = with_bmi_prefix(f"Denied. {base}")

    else:
        final_status = "FLAGGED"
        reason = f"Unknown verdict '{verdict}'. Route to manual utilization review."

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

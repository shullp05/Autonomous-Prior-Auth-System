"""
agent_logic.py - LangGraph-based Clinical Prior Authorization Agent

This module implements an autonomous agent for evaluating Wegovy (semaglutide)
prior authorization requests using a combination of:

1. **RAG (Retrieval-Augmented Generation)**: Policy retrieval from ChromaDB
2. **LLM Reasoning**: Clinical audit via local LLM with structured JSON output
3. **Python Guardrails**: Deterministic override layer for safety-critical logic
4. **Pydantic Validation**: Schema enforcement on LLM outputs

Architecture:
    The agent follows a stateful graph workflow:

    retrieve_policy → clinical_audit → make_decision → END

Key Functions:
    - build_agent(): Constructs the LangGraph workflow
    - retrieve_policy(): RAG retrieval with optional reranking
    - clinical_audit(): LLM-based clinical evaluation with guardrails
    - generate_appeal_letter(): Creates appeal letters for denials

Configuration (Environment Variables):
    - PA_AUDIT_MODEL: LLM for clinical reasoning
    - PA_APPEAL_MODEL: LLM for appeal generation (default: same as AUDIT_MODEL)
    - PA_EMBED_MODEL: Embedding model for RAG
    - PA_RERANK_MODEL: Reranker for improved retrieval (default: bce-reranker)
    - PA_RERANK_DEVICE: cpu|cuda
    - PA_ENABLE_RERANK: Enable/disable reranking (default: true)
    - PA_RAG_SCORE_FLOOR: BCE score threshold (default from config)
    - PA_RAG_MIN_DOCS: Minimum docs to keep even if scores are low (default from config)

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Literal, Optional, TypedDict, List, Tuple, Any

import pandas as pd
import psutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

# IMPORTANT: load .env BEFORE importing config (config may read env at import time)
load_dotenv()

logger = logging.getLogger(__name__)

# Centralized policy constants (single source of truth)
from policy_constants import (
    BMI_OBESE_THRESHOLD,
    BMI_OVERWEIGHT_THRESHOLD,
    AMBIGUOUS_DIABETES,
    AMBIGUOUS_BP,
    AMBIGUOUS_OBESITY,
    AMBIGUOUS_SLEEP_APNEA,
    AMBIGUOUS_THYROID,
    AMBIGUOUS_APPEAL_TERMS,
    SAFETY_MTC_MEN2,
    SAFETY_PREGNANCY_LACTATION,
    SAFETY_HYPERSENSITIVITY,
    SAFETY_PANCREATITIS,
    SAFETY_SUICIDALITY,
    SAFETY_GI_MOTILITY,
    PROHIBITED_GLP1,
)

from policy_snapshot import SNAPSHOT_PATH, load_policy_snapshot
from schema_validation import validate_policy_snapshot
from policy_engine import _parse_bmi, evaluate_eligibility

from config import (
    AUDIT_MODEL_NAME,
    AUDIT_MODEL_OPTIONS,
    AUDIT_MODEL_RAM_GB,
    AUDIT_MODEL_FLAVOR,
    APPEAL_MODEL_NAME,
    EMBED_MODEL_NAME,
    PA_RERANK_MODEL,
    PA_RERANK_DEVICE,
    PA_RAG_K_VECTOR,
    PA_RAG_TOP_K_DOCS,
    POLICY_ID as ACTIVE_POLICY_ID,
    PA_RAG_SCORE_FLOOR,
    PA_RAG_MIN_DOCS,
)

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


def _ensure_data_loaded() -> None:
    """Lazy load data to allow module import without files present."""
    global df_patients, df_meds, df_conditions, df_obs
    if df_patients is not None:
        return

    try:
        if not os.path.exists("output/data_patients.csv"):
            logger.warning("Data files not found. chaos_monkey.py output required for execution.")
            return

        df_patients = pd.read_csv("output/data_patients.csv")
        df_meds = pd.read_csv("output/data_medications.csv")
        df_conditions = pd.read_csv("output/data_conditions.csv")
        df_obs = pd.read_csv("output/data_observations.csv")
    except Exception as e:
        logger.error(f"Data Load Error: {e}")


def write_model_trace(model_name: str, role: str, params: dict, required_ram_gb: float = None) -> None:
    trace = {
        "model_name": model_name,
        "role": role,
        "params": params,
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
        "ram_required_gb": required_ram_gb or "unknown",
    }
    try:
        with open(".last_model_trace.json", "w") as f:
            json.dump(trace, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write model trace: {e}")


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

    p_obs = df_obs_in[df_obs_in["patient_id"] == patient_id]

    # 1) Explicit BMI
    bmi_rows = p_obs[p_obs["type"] == "BMI"].sort_values("date", ascending=False)
    if not bmi_rows.empty:
        return f"{bmi_rows.iloc[0]['value']} (Source: EMR)"

    # 2) Calculate from height/weight
    try:
        wt = p_obs[p_obs["type"] == "Weight"].sort_values("date", ascending=False)
        ht = p_obs[p_obs["type"] == "Height"].sort_values("date", ascending=False)

        if wt.empty or ht.empty:
            return "MISSING_DATA"

        weight_kg = float(wt.iloc[0]["value"])
        height_cm = float(ht.iloc[0]["value"])

        height_m = height_cm / 100.0
        if height_m == 0:
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
        logger.error("Patient data not loaded (run chaos_monkey.py first).")
        return None

    pat_rows = df_patients[df_patients["patient_id"] == patient_id].to_dict("records")
    if not pat_rows:
        return None

    pat = pat_rows[0]

    meds: List[str] = []
    if df_meds is not None and not getattr(df_meds, "empty", True):
        meds = df_meds[df_meds["patient_id"] == patient_id]["medication_name"].tolist()

    conds: List[str] = []
    if df_conditions is not None and not getattr(df_conditions, "empty", True):
        conds = df_conditions[df_conditions["patient_id"] == patient_id]["condition_name"].tolist()

    latest_bmi = calculate_bmi_if_missing(patient_id, df_obs)

    return {
        "name": pat.get("name", ""),
        "dob": pat.get("dob", ""),
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


# --- APPEAL GENERATOR ---
def generate_appeal_letter(patient_data, denial_reason, findings):
    """
    Generate a clinically useful Prior Authorization request or appeal letter.

    For FLAGGED/ambiguous cases, generates a professional documentation request.
    For clear denials with no clinical basis, may still generate guidance for provider.
    Only returns None on LLM error or safety-denied cases where no appeal is appropriate.
    """
    llm = _make_llm(model=APPEAL_MODEL, temperature=0.2, prefer_json=False)

    # Extract key clinical data for the prompt
    patient_name = patient_data.get('name', '[Patient Name]') if isinstance(patient_data, dict) else "[Patient Name]"
    patient_dob = patient_data.get('dob', '[DOB]') if isinstance(patient_data, dict) else "[DOB]"
    conditions = patient_data.get('conditions', []) if isinstance(patient_data, dict) else []
    
    # Filter out non-clinical noise from conditions list
    clinical_conditions = [c for c in conditions if not any(noise in c.lower() for noise in [
        'medication review due', 'finding', 'situation', 'received', 'employment', 
        'education', 'social contact', 'housing', 'transport', 'refugee'
    ])][:15]  # Limit to top 15 relevant conditions
    
    bmi_value = findings.get('bmi_numeric') if findings else None
    verdict = findings.get('verdict', '') if findings else ''
    evidence = findings.get('evidence_quoted', '') if findings else ''
    
    # Determine if this is a FLAGGED/ambiguous case vs a hard denial
    is_flagged_case = verdict == 'MANUAL_REVIEW' or 'ambiguous' in denial_reason.lower() or 'flagged' in denial_reason.lower()
    is_safety_denial = verdict == 'DENIED_SAFETY' or 'safety exclusion' in denial_reason.lower()
    
    # Don't generate appeals for safety denials - those are medically contraindicated
    if is_safety_denial:
        return None

    try:
        prompt = f"""You are a board-certified Clinical Pharmacist drafting a Prior Authorization request letter.
Your goal is to create a professional, clinically useful document that helps the provider understand:
1. What documentation is needed to support the PA request
2. What specific clinical criteria must be met
3. What action the provider should take

PATIENT INFORMATION:
- Name: {patient_name}
- Date of Birth: {patient_dob}
- Current BMI: {bmi_value if bmi_value else '[Not documented - REQUIRES VERIFICATION]'} kg/m²
- Relevant Clinical Conditions: {', '.join(clinical_conditions) if clinical_conditions else '[Requires chart review]'}

CURRENT STATUS:
{denial_reason}

CLINICAL EVIDENCE FROM CHART:
{json.dumps(findings, indent=2) if findings else 'Limited clinical data extracted'}

COVERAGE CRITERIA FOR WEGOVY (Semaglutide):
- BMI ≥ 30 kg/m² with documented obesity diagnosis, OR
- BMI ≥ 27 kg/m² with at least ONE qualifying weight-related comorbidity:
  * Hypertension / High Blood Pressure
  * Type 2 Diabetes Mellitus (NOT prediabetes alone)
  * Dyslipidemia / Hyperlipidemia
  * Obstructive Sleep Apnea (OSA) - must specify "obstructive" or "OSA"
  * Cardiovascular Disease (ASCVD)

IMPORTANT CONTEXT:
- "Prediabetes" alone does NOT qualify as a comorbidity for Wegovy coverage
- Generic "sleep apnea" must be clarified as "obstructive sleep apnea (OSA)" to qualify
- Essential hypertension DOES qualify as a comorbidity

INSTRUCTIONS:
Write a professional Prior Authorization request letter that:
1. Opens with a formal salutation to the Medical Director
2. States the medication being requested (Wegovy/semaglutide) and indication (chronic weight management)
3. Summarizes the patient's relevant clinical profile (BMI, qualifying conditions)
4. Identifies what additional documentation may strengthen the request (if applicable)
5. Makes a clear, respectful request for coverage consideration
6. Closes professionally

Use a formal medical letter format. Do NOT include markdown formatting.
Do NOT output "NO_APPEAL_POSSIBLE" - always generate a useful letter.
The letter should be directly usable by a provider with minimal editing.
"""
        response = llm.invoke(prompt)

        write_model_trace(
            model_name=APPEAL_MODEL,
            role="appeal_generator",
            params={"temperature": 0.2},
            required_ram_gb=AUDIT_MODEL_RAM,
        )

        text = str(response.content or "").strip()
        
        # Ensure we got actual content, not just a refusal
        if not text or text == "NO_APPEAL_POSSIBLE" or len(text) < 100:
            logger.warning(f"LLM returned insufficient appeal content: {text[:50]}...")
            return None
            
        return text

    except Exception as e:
        logger.error(f"Error generating appeal: {e}")
        return None


def _generate_fallback_pa_template(patient_data: dict, reason: str, findings: dict) -> str:
    """Generate a clinically-focused PA template when the LLM cannot produce one."""
    name = patient_data.get("name", "[Patient Name]") if patient_data else "[Patient Name]"
    dob = patient_data.get("dob", "[DOB]") if patient_data else "[DOB]"
    bmi = findings.get("bmi_numeric") if findings else None
    conditions = patient_data.get("conditions", []) if patient_data else []
    evidence = findings.get("evidence_quoted", "") if findings else ""

    # Filter out non-clinical noise from conditions
    noise_terms = [
        'medication review due', 'finding', 'situation', 'received', 'employment',
        'education', 'social contact', 'housing', 'transport', 'refugee', 'part-time',
        'full-time', 'stress', 'limited'
    ]
    clinical_conditions = [
        c for c in conditions 
        if not any(noise in c.lower() for noise in noise_terms)
    ][:12]  # Limit to 12 most relevant

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
            bmi_text = "Current BMI: [REQUIRES VERIFICATION]"
            bmi_analysis = "BMI could not be reliably extracted. Please verify from chart."
    else:
        bmi_text = "Current BMI: [REQUIRES VERIFICATION]"
        bmi_analysis = "BMI could not be reliably extracted. Please verify from chart."

    if clinical_conditions:
        cond_text = "Relevant Clinical Conditions:\n" + "\n".join(f"  • {c}" for c in clinical_conditions)
    else:
        cond_text = "Relevant Clinical Conditions: [Chart review required - no qualifying conditions extracted]"

    # Identify what's missing based on the reason
    if "prediabetes" in reason.lower():
        action_needed = """
ACTION REQUIRED - PREDIABETES DOES NOT QUALIFY:
The term "Prediabetes" alone does not meet coverage criteria. To support this request:
  1. Document Type 2 Diabetes Mellitus if present, OR
  2. Document another qualifying comorbidity (HTN, dyslipidemia, OSA, ASCVD)
  3. If patient has Essential Hypertension, this DOES qualify - verify documentation"""
    elif "sleep apnea" in reason.lower():
        action_needed = """
ACTION REQUIRED - SPECIFY OBSTRUCTIVE SLEEP APNEA:
Generic "sleep apnea" must be specified as "Obstructive Sleep Apnea (OSA)" to qualify.
  1. Review sleep study or pulmonology notes
  2. Update diagnosis to specify OSA if appropriate
  3. Consider alternative qualifying comorbidities if OSA cannot be confirmed"""
    else:
        action_needed = """
ACTION REQUIRED:
  1. Verify current BMI measurement is documented
  2. Confirm presence of qualifying comorbidity (HTN, T2DM, dyslipidemia, OSA, ASCVD)
  3. Ensure no safety exclusions are present (MTC, MEN2, pregnancy, concurrent GLP-1)"""

    template = f"""PRIOR AUTHORIZATION REQUEST - WEGOVY (SEMAGLUTIDE)
Indication: Chronic Weight Management

Patient: {name}
Date of Birth: {dob}

CLINICAL PROFILE:
{bmi_text}
Assessment: {bmi_analysis}

{cond_text}

CASE STATUS:
{reason}

COVERAGE CRITERIA SUMMARY:
Wegovy coverage requires ONE of the following:
  • BMI ≥ 30 kg/m² with documented obesity diagnosis
  • BMI ≥ 27 kg/m² with qualifying comorbidity:
    - Hypertension / High Blood Pressure ✓
    - Type 2 Diabetes Mellitus (NOT prediabetes) ✓
    - Dyslipidemia / Hyperlipidemia ✓
    - Obstructive Sleep Apnea (OSA) ✓
    - Cardiovascular Disease (ASCVD) ✓

{action_needed}

If criteria are met after chart review, please document the specific qualifying
condition(s) and resubmit the prior authorization request.

_____________________________________________
Provider Signature / Date
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
    lines.append(
        "No concurrent GLP-1 / GLP-1-GIP agents: "
        + ", ".join(snapshot["drug_conflicts"]["glp1_or_glp1_gip_agents"])
    )
    return "\n".join(lines)


# --- STATIC POLICY FALLBACK (used when RAG unavailable) ---
POLICY_SUMMARY_TEXT = _build_policy_summary(SNAPSHOT)
_STATIC_POLICY_FALLBACK = POLICY_SUMMARY_TEXT


# --- BCE RERANK (lazy) ---
_BCE_MODEL: Optional[Any] = None
_BCE_LOCK = threading.Lock()


def _get_bce_model() -> Any:
    """
    Lazy-load BCE reranker. Default to CPU unless explicitly configured.
    Keeps this import inside the function so the module can still import even
    if BCEmbedding isn't installed (rerank will then be disabled gracefully).
    """
    global _BCE_MODEL
    if _BCE_MODEL is not None:
        return _BCE_MODEL

    with _BCE_LOCK:
        if _BCE_MODEL is not None:
            return _BCE_MODEL

        try:
            from BCEmbedding import RerankerModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"BCEmbedding is not available; cannot rerank. ({e})")

        model_name = os.getenv("PA_RERANK_MODEL", RERANK_MODEL)
        device = os.getenv("PA_RERANK_DEVICE", RERANK_DEVICE_DEFAULT).lower()
        if device not in ("cpu", "cuda", "mps"):
            device = "cpu"

        _BCE_MODEL = RerankerModel(model_name_or_path=model_name, device=device)
        return _BCE_MODEL


def rerank_bce(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Return list of (doc, score) sorted by descending relevance.
    """
    if not docs:
        return []

    model = _get_bce_model()
    passages = [d.page_content for d in docs]
    pairs = [[query, p] for p in passages]
    scores = model.compute_score(pairs)

    scored: List[Tuple[Document, float]] = [(d, float(s)) for d, s in zip(docs, scores)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


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


def _policy_aware_sort_docs(
    docs: List[Document],
    scores: List[float],
    policy_path: Optional[str],
) -> List[Document]:
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
def _build_policy_query(det_result, patient_data: dict, drug: str) -> str:
    drug = drug or "Wegovy"
    if det_result:
        bmi = det_result.bmi_numeric
        path = det_result.policy_path
        verdict = det_result.verdict
        category = det_result.comorbidity_category
        safety = det_result.safety_flag
        return (
            f"{drug} prior authorization policy evidence for {path} with verdict {verdict}; "
            f"BMI {bmi}; comorbidity {category}; safety {safety}. "
            "Return diagnosis strings, comorbidity rules, safety exclusions, and documentation requirements."
        )
    bmi_hint = patient_data.get("latest_bmi") if patient_data else "unknown"
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
        blocks.append(
            f"=== POLICY EVIDENCE {idx} ===\n[section: {section}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def retrieve_policy(state: AgentState):
    """
    Retrieve the active Wegovy policy from ChromaDB vector store if available,
    otherwise fall back to static policy text.

    Uses:
    - PA_RAG_K_VECTOR: initial vector candidate size
    - PA_RAG_TOP_K_DOCS: max docs passed to the LLM
    - PA_RAG_SCORE_FLOOR / PA_RAG_MIN_DOCS: BCE filter rules (env override supported)
    """
    drug = state.get("drug_requested", "Wegovy")
    logger.info(f"[RAG] Retrieving policy for {drug}")

    patient_id = state.get("patient_id")
    patient_data = state.get("patient_data") or look_up_patient_data(patient_id)
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
                filter={"policy_id": ACTIVE_POLICY_ID},
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
                        filtered_docs = _policy_aware_sort_docs(
                            filtered_docs,
                            filtered_scores,
                            policy_path=policy_path,
                        )

                        policy_docs = filtered_docs[:PA_RAG_TOP_K_DOCS]
                    except Exception as e:
                        logger.warning("[RAG] Rerank failed; falling back to vector order: %s", e)
                        policy_docs = vector_docs[:PA_RAG_TOP_K_DOCS]
                else:
                    policy_docs = vector_docs[:PA_RAG_TOP_K_DOCS]

                policy_text = _format_policy_evidence(policy_docs)
                logger.info(
                    "[RAG] Using %d policy atoms for LLM (%d chars)",
                    len(policy_docs),
                    len(policy_text),
                )
            else:
                logger.warning("[RAG] ChromaDB returned no results, using fallback")

        except ImportError as e:
            logger.warning(f"[RAG] ChromaDB/embeddings not available: {e}")
        except Exception as e:
            logger.warning(f"[RAG] Vector retrieval failed: {e}")

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
    """
    Try very hard to extract a single JSON object from a model response.
    """
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
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace : last_brace + 1].strip()
    else:
        candidate = raw

    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Parsed JSON is not an object.", candidate, 0)
    return obj


# --- PYDANTIC SCHEMA FOR LLM OUTPUT VALIDATION ---
class AuditResult(BaseModel):
    """Strict schema for LLM audit output validation."""
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


# --- POLICY GUARDRAILS (Python-side cross-check) ---
def _has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def _is_generic_sleep_apnea(text: str) -> bool:
    """
    True when string indicates sleep apnea non-specifically.
    - "sleep apnea" (generic) => True
    - "obstructive sleep apnea" => False
    - "OSA" => False
    """
    s = (text or "").strip().lower()
    if "sleep apnea" not in s:
        return False
    if "obstructive" in s:
        return False
    if _has_word(s, "osa"):
        return False
    return True


def _apply_policy_guardrails(audit_result: dict, patient_data: dict = None) -> dict:
    """
    Enforce hard policy rules after the LLM's reasoning.
    """
    verdict = audit_result.get("verdict")
    safety_flag = audit_result.get("safety_flag", "CLEAR")
    category = audit_result.get("comorbidity_category", "NONE")

    raw_ev = audit_result.get("evidence_quoted")
    evidence = str(raw_ev if raw_ev is not None else "").strip()
    ev_lower = evidence.lower()

    bmi = audit_result.get("bmi_numeric")
    try:
        if bmi is not None:
            bmi = float(bmi)
    except Exception:
        bmi = None

    # Deterministic override when raw patient data is available
    if patient_data:
        deterministic = evaluate_eligibility(patient_data)
        if deterministic.verdict != "APPROVED":
            return deterministic.to_dict()

    # --- DETERMINISTIC SAFETY OVERRIDE (CRITICAL) ---
    if patient_data:
        raw_conds = [str(c).lower() for c in patient_data.get("conditions", []) if c is not None]
        raw_meds = [str(m).lower() for m in patient_data.get("meds", []) if m is not None]

        # Condition-based safety
        checks = [
            (SAFETY_MTC_MEN2, "MTC/MEN2 exclusion"),
            (SAFETY_PREGNANCY_LACTATION, "Pregnancy/Lactation exclusion"),
            (SAFETY_HYPERSENSITIVITY, "Hypersensitivity exclusion"),
            (SAFETY_PANCREATITIS, "Pancreatitis exclusion"),
            (SAFETY_SUICIDALITY, "Suicidality exclusion"),
            (SAFETY_GI_MOTILITY, "GI Motility exclusion"),
        ]
        for term_list, reason_text in checks:
            for cond in raw_conds:
                if any(ex in cond for ex in term_list):
                    audit_result["verdict"] = "DENIED_SAFETY"
                    audit_result["safety_flag"] = "DETECTED"
                    audit_result["reasoning"] = f"HARD STOP: {reason_text} detected in patient record ('{cond}')."
                    audit_result["evidence_quoted"] = cond
                    return audit_result

        # Medication-based safety (concurrent GLP-1)
        for med in raw_meds:
            if any(ex in med for ex in PROHIBITED_GLP1):
                audit_result["verdict"] = "DENIED_SAFETY"
                audit_result["safety_flag"] = "DETECTED"
                audit_result["reasoning"] = f"HARD STOP: Concurrent GLP-1 usage detected in patient record ('{med}')."
                audit_result["evidence_quoted"] = med
                return audit_result

    # 1) If verdict is DENIED_SAFETY, ensure it's valid
    if verdict == "DENIED_SAFETY":
        is_mtc_men2 = any(t in ev_lower for t in SAFETY_MTC_MEN2)
        is_pregnancy = any(t in ev_lower for t in SAFETY_PREGNANCY_LACTATION)
        is_hypersensitivity = any(t in ev_lower for t in SAFETY_HYPERSENSITIVITY)
        is_pancreatitis = any(t in ev_lower for t in SAFETY_PANCREATITIS)
        is_suicidality = any(t in ev_lower for t in SAFETY_SUICIDALITY)
        is_gi_motility = any(t in ev_lower for t in SAFETY_GI_MOTILITY)
        is_glp1 = any(t in ev_lower for t in PROHIBITED_GLP1)

        if (is_mtc_men2 or is_pregnancy or is_hypersensitivity or is_pancreatitis or
            is_suicidality or is_gi_motility or is_glp1):
            return audit_result

        is_ambig_thyroid = any(t.lower() in ev_lower for t in AMBIGUOUS_THYROID)
        if is_ambig_thyroid and not is_mtc_men2:
            audit_result["verdict"] = "MANUAL_REVIEW"
            audit_result["safety_flag"] = "CLEAR"
            audit_result["comorbidity_category"] = "NONE"
            audit_result["reasoning"] = (
                "Thyroid malignancy is documented but not clearly Medullary Thyroid "
                "Carcinoma or MEN2; Wegovy safety requires manual review rather than "
                "automatic safety denial."
            )
            return audit_result

    # 1.5) Safety override based on evidence_quoted (even if APPROVED)
    is_mtc_men2 = any(t in ev_lower for t in SAFETY_MTC_MEN2)
    is_pregnancy = any(t in ev_lower for t in SAFETY_PREGNANCY_LACTATION)
    is_hypersensitivity = any(t in ev_lower for t in SAFETY_HYPERSENSITIVITY)
    is_pancreatitis = any(t in ev_lower for t in SAFETY_PANCREATITIS)
    is_suicidality = any(t in ev_lower for t in SAFETY_SUICIDALITY)
    is_gi_motility = any(t in ev_lower for t in SAFETY_GI_MOTILITY)
    is_glp1 = any(t in ev_lower for t in PROHIBITED_GLP1)

    if (is_mtc_men2 or is_pregnancy or is_hypersensitivity or is_pancreatitis or
        is_suicidality or is_gi_motility or is_glp1):
        audit_result["verdict"] = "DENIED_SAFETY"
        audit_result["safety_flag"] = "DETECTED"
        audit_result["reasoning"] = (
            f"Wegovy is denied because a safety exclusion was detected in the evidence ('{evidence}'); "
            "safety overrides any approval criteria."
        )
        return audit_result

    # 2) If verdict is APPROVED, enforce BMI and comorbidity rules
    if verdict == "APPROVED":
        if safety_flag == "DETECTED":
            audit_result["verdict"] = "DENIED_SAFETY"
            audit_result["reasoning"] = (
                "Wegovy is denied because a documented safety exclusion is present; "
                "safety overrides BMI/comorbidity criteria."
            )
            return audit_result

        if bmi is None:
            audit_result["verdict"] = "DENIED_MISSING_INFO"
            audit_result["comorbidity_category"] = "NONE"
            audit_result["reasoning"] = (
                "BMI could not be reliably determined from the chart, so clinical "
                "eligibility for Wegovy cannot be confirmed."
            )
            return audit_result

        if bmi < BMI_OVERWEIGHT_THRESHOLD:
            audit_result["verdict"] = "DENIED_CLINICAL"
            audit_result["comorbidity_category"] = "NONE"
            audit_result["reasoning"] = (
                f"Wegovy is denied because BMI is below {BMI_OVERWEIGHT_THRESHOLD} kg/m², which does not meet "
                "the payer's minimum overweight threshold."
            )
            return audit_result

        if BMI_OVERWEIGHT_THRESHOLD <= bmi < BMI_OBESE_THRESHOLD:
            ambiguous_diabetes_hit = any(term.lower() in ev_lower for term in AMBIGUOUS_DIABETES)
            ambiguous_bp_hit = any(term.lower() in ev_lower for term in AMBIGUOUS_BP)
            ambiguous_obesity_hit = any(term.lower() in ev_lower for term in AMBIGUOUS_OBESITY)
            ambiguous_sleep_hit = _is_generic_sleep_apnea(evidence)

            ambiguous_weight_related_hit = (
                ambiguous_diabetes_hit or ambiguous_bp_hit or ambiguous_obesity_hit or ambiguous_sleep_hit
            )

            if ambiguous_weight_related_hit:
                audit_result["verdict"] = "MANUAL_REVIEW"
                audit_result["comorbidity_category"] = "NONE"
                audit_result["reasoning"] = (
                    "BMI is between 27 and 29.9 with an ambiguous weight-related risk "
                    f"term ('{evidence}') documented; Wegovy eligibility requires "
                    "manual clinical review rather than automatic approval."
                )
                return audit_result

            if category == "NONE" or not evidence:
                audit_result["verdict"] = "DENIED_CLINICAL"
                audit_result["comorbidity_category"] = "NONE"
                audit_result["reasoning"] = (
                    "BMI is between 27 and 29.9 but no qualifying weight-related "
                    "comorbidity is documented; Wegovy does not meet policy criteria."
                )
                return audit_result

    return audit_result


# --- CLINICAL AUDIT NODE ---
def clinical_audit(state: AgentState):
    logger.info(f"[Audit] Checking Patient {state.get('patient_id', '')}")

    p_data = state.get("patient_data") or look_up_patient_data(state.get("patient_id"))
    det_decision = state.get("deterministic_decision")
    if p_data and det_decision is None:
        det_decision = evaluate_eligibility(p_data).to_dict()

    if not p_data:
        audit_result = {
            "bmi_numeric": None,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "verdict": "MANUAL_REVIEW",
            "reasoning": "Patient record could not be found; route to manual review.",
        }
        return {"patient_data": None, "audit_findings": audit_result}

    det_bmi = _parse_bmi(p_data.get("latest_bmi"))
    det_bmi_str = str(det_bmi) if det_bmi is not None else "null"
    det_json = json.dumps(det_decision or {}, ensure_ascii=True)

    llm = _make_llm(model=AUDIT_MODEL, temperature=0, prefer_json=True, options=AUDIT_MODEL_OPTS)

    system_prompt = f"""
You are a Senior Utilization Review Medical Director acting as a deterministic logic engine for a prior authorization request for Wegovy (semaglutide).
Follow the algorithm exactly and output ONE strict JSON object.

[INPUT DATA]
- BMI_raw: "{p_data['latest_bmi']}"
- BMI_numeric_verified: {det_bmi_str}
- Condition_List: {p_data['conditions']}
- Med_List: {p_data['meds']}
- Deterministic decision (source of truth): {det_json}

All matching is CASE-INSENSITIVE. Do NOT assume any condition or medication that is not explicitly listed.

[TAXONOMY DEFINITIONS]
Target_Drug: Wegovy (Semaglutide)

# CLINICAL NOTE:
# Wegovy coverage requires:
#   - BMI ≥ 30 kg/m² (obesity), OR
#   - BMI ≥ 27 kg/m² (overweight) with at least ONE weight-related comorbid condition:
#       * Hypertension (high blood pressure)
#       * Type 2 diabetes mellitus
#       * Dyslipidemia / high cholesterol
#       * Obstructive sleep apnea (OSA)
#       * Established cardiovascular disease (e.g., prior MI, stroke, or peripheral arterial disease)
#
# IMPORTANT: Do NOT treat BMI 29.9 or lower as meeting the obesity threshold.
#            Only BMI ≥ 30.0 qualifies on BMI alone without comorbidities.
#            BMI 27.0–29.9 requires at least one qualifying comorbidity per policy.

# SAFETY EXCLUSIONS:
# The ONLY cancer/carcinoma-related safety exclusions are:
#   - Personal or family history of Medullary Thyroid Carcinoma (MTC)
#   - Personal or family history of Multiple Endocrine Neoplasia syndrome type 2 (MEN 2)
# Papillary or Follicular thyroid cancer/carcinoma are NOT safety exclusions per policy.

SAFETY_EXCLUSIONS are triggered by ANY of the following:
1) MTC / MEN2 PERSONAL OR FAMILY HISTORY (Condition-based)
2) PREGNANCY OR CURRENTLY NURSING (Condition-based)
3) CONCURRENT GLP-1 RECEPTOR AGONIST USE (Medication-based)

VALID_COMORBIDITIES by category (for BMI 27–29.9):
- HYPERTENSION
- LIPIDS
- DIABETES (EXCLUDE prediabetes/borderline)
- OSA (EXCLUDE generic sleep apnea)
- CVD (established cardiovascular disease)

[EXECUTION ALGORITHM]
Step 1 – Use BMI_numeric_verified as BMI_VALUE (do not re-parse BMI_raw if verified exists)
Step 2 – Safety screen (highest priority)
Step 3 – BMI presence check
Step 4 – BMI thresholds and comorbidity evaluation
Step 5 – Self-Correction & Normalization (Crucial)

[JSON OUTPUT SCHEMA]
Return ONE JSON object. No Markdown.
{{
  "bmi_numeric": <number or null>,
  "safety_flag": "CLEAR" or "DETECTED",
  "comorbidity_category": "NONE" or "HYPERTENSION" or "LIPIDS" or "DIABETES" or "OSA" or "CVD",
  "evidence_quoted": "<string>",
  "verdict": "APPROVED" or "DENIED_SAFETY" or "DENIED_CLINICAL" or "DENIED_MISSING_INFO" or "MANUAL_REVIEW" or "DENIED_BENEFIT_EXCLUSION" or "DENIED_OTHER",
  "reasoning": "<concise explanation>"
}}

[FORMAT RULES]
- Output MUST be valid JSON: double-quoted keys and string values, no trailing commas, no comments.
- Do NOT wrap the JSON in backticks or markdown fences.
- Do NOT output any text before or after the JSON.
"""

    try:
        response = llm.invoke(system_prompt)
        content = str(response.content or "").strip()
        raw_json = _extract_json_object(content)

        try:
            validated = AuditResult(**raw_json)
            audit_result = validated.model_dump()
        except ValidationError as ve:
            logger.warning(f"Schema validation failed, using defaults: {ve}")
            audit_result = AuditResult(
                verdict="MANUAL_REVIEW",
                reasoning=f"Schema validation error: {str(ve)[:200]}",
            ).model_dump()

            for key in ["bmi_numeric", "evidence_quoted", "reasoning", "verdict"]:
                if key in raw_json and raw_json[key] is not None:
                    try:
                        if key == "bmi_numeric":
                            audit_result[key] = float(raw_json[key]) if raw_json[key] else None
                        else:
                            audit_result[key] = str(raw_json[key])
                    except (ValueError, TypeError):
                        pass

        audit_result = _apply_policy_guardrails(audit_result, p_data)

        # --- DETERMINISTIC CROSS-CHECK ---
        try:
            det_result = evaluate_eligibility(p_data)

            # Case 1: LLM missed BMI but deterministic found it
            if audit_result.get("bmi_numeric") is None and det_result.bmi_numeric is not None:
                logger.warning(
                    f"[Audit] LLM missed BMI ({det_result.bmi_numeric}); overriding with deterministic BMI."
                )
                audit_result["bmi_numeric"] = det_result.bmi_numeric

                if audit_result.get("verdict") == "DENIED_MISSING_INFO":
                    audit_result["verdict"] = det_result.verdict
                    audit_result["reasoning"] = det_result.reasoning
                    audit_result["comorbidity_category"] = det_result.comorbidity_category
                    audit_result["evidence_quoted"] = det_result.evidence_quoted
                    audit_result["safety_flag"] = det_result.safety_flag

            # Case 2: LLM under-called approval
            elif audit_result.get("verdict") in ["DENIED_CLINICAL", "MANUAL_REVIEW"] and det_result.verdict == "APPROVED":
                logger.warning("[Audit] LLM under-called approval; overriding with deterministic approval.")
                audit_result["verdict"] = "APPROVED"
                audit_result["reasoning"] = det_result.reasoning
                audit_result["comorbidity_category"] = det_result.comorbidity_category
                audit_result["evidence_quoted"] = det_result.evidence_quoted

        except Exception as e:
            logger.error(f"[Audit] Deterministic cross-check failed: {e}")

        for key, default in [
            ("policy_path", (det_decision or {}).get("policy_path", "UNKNOWN")),
            ("decision_type", (det_decision or {}).get("decision_type", audit_result.get("verdict", "UNKNOWN"))),
            ("safety_exclusion_code", (det_decision or {}).get("safety_exclusion_code")),
            ("ambiguity_code", (det_decision or {}).get("ambiguity_code")),
        ]:
            audit_result.setdefault(key, default)

        logger.info(
            f"[AI Logic] BMI {audit_result.get('bmi_numeric')} | "
            f"Safety: {audit_result.get('safety_flag')} | "
            f"Verdict: {audit_result.get('verdict')}"
        )

    except Exception as e:
        logger.exception("Error parsing audit JSON; falling back to deterministic engine")
        if p_data:
            det_result = evaluate_eligibility(p_data)
            audit_result = det_result.to_dict()
            audit_result["reasoning"] = (
                f"LLM parsing failed ({e}); using deterministic policy result: {det_result.reasoning}"
            )
        else:
            audit_result = {
                "bmi_numeric": None,
                "safety_flag": "CLEAR",
                "comorbidity_category": "NONE",
                "evidence_quoted": "",
                "verdict": "MANUAL_REVIEW",
                "reasoning": f"System Parsing Error: {str(e)}",
            }

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
def make_decision(state: AgentState):
    """
    Turn the audit findings into:
      - final_decision: APPROVED / DENIED / FLAGGED / PROVIDER_ACTION_REQUIRED
      - reasoning: human-readable summary
      - appeal_letter: only when there is actually something to argue
      - appeal_note: provider-facing note for specified ambiguous MANUAL_REVIEW triggers
    """
    f = state.get("audit_findings", {}) or {}
    p_data = state.get("patient_data", {}) or {}
    verdict = f.get("verdict", "MANUAL_REVIEW")
    model_used = state.get("audit_model_flavor", "unknown")

    appeal_letter = None
    appeal_note = None
    final_status = "DENIED"

    raw_bmi = f.get("bmi_numeric", None)
    bmi = None
    if isinstance(raw_bmi, (int, float)):
        bmi = float(raw_bmi)
    elif isinstance(raw_bmi, str):
        try:
            bmi = float(raw_bmi)
        except Exception:
            bmi = None

    evidence = str(f.get("evidence_quoted") or "").strip()
    safety_flag = f.get("safety_flag", "CLEAR")

    def with_bmi_prefix(text: str) -> str:
        if bmi is not None:
            return f"BMI {bmi:.2f}. {text}".strip()
        return text

    if verdict == "APPROVED":
        final_status = "APPROVED"
        if bmi is not None and bmi >= 30:
            reason = with_bmi_prefix(
                f"Approved. Safety is {safety_flag}. BMI is at or above 30 kg/m², so Wegovy is approved based on BMI alone."
            )
        elif bmi is not None and 27 <= bmi < 30 and evidence:
            reason = with_bmi_prefix(
                f"Approved. Safety is {safety_flag} and BMI is {bmi:.2f} with '{evidence}' present, so Wegovy is approved per policy criteria."
            )
        else:
            reason = with_bmi_prefix("Approved. Wegovy approved per policy criteria.")

    elif verdict == "DENIED_SAFETY":
        final_status = "DENIED"
        base = (
            f.get("reasoning")
            or "Wegovy is denied due to a documented safety exclusion (e.g., MTC, MEN2, pregnancy, or concurrent GLP-1 use)."
        )
        reason = with_bmi_prefix(f"HARD STOP: Safety Exclusion. {base}")

    elif verdict == "DENIED_MISSING_INFO":
        final_status = "PROVIDER_ACTION_REQUIRED"
        base = f.get("reasoning") or ""
        if bmi is None:
            reason = (
                "Provider action required. BMI is not documented in the chart and could not be calculated "
                "from recent height and weight. Please enter a current BMI in the medical record so this "
                "prior authorization request can be processed."
            )
        else:
            reason = with_bmi_prefix(f"Provider action required. {base}".strip())

        appeal_letter = generate_appeal_letter(p_data, reason, f)
        if appeal_letter is None:
            appeal_letter = _generate_fallback_pa_template(p_data, reason, f)

    elif verdict == "DENIED_CLINICAL":
        final_status = "DENIED"
        base = (
            f.get("reasoning")
            or "Wegovy is denied because BMI and documented weight-related comorbidities do not meet policy criteria."
        )
        reason = with_bmi_prefix(f"Denied. {base}")

        if bmi is not None and 27 <= bmi < 30 and evidence:
            logger.info(f"Drafting appeal for borderline DENIED_CLINICAL case (BMI {bmi}, evidence='{evidence}')")
            appeal_letter = generate_appeal_letter(p_data, reason, f)
            if appeal_letter is None:
                reason += " Auto-appeal suppressed: no defensible clinical basis identified."
        else:
            if bmi is not None and bmi < 27:
                reason += " (BMI is below payer threshold for Wegovy; no automatic appeal generated.)"
            elif bmi is not None and 27 <= bmi < 30 and not evidence:
                reason += (
                    " (BMI is 27–29.9 but no qualifying or ambiguous weight-related "
                    "risk factors are documented; auto-appeal suppressed.)"
                )

    elif verdict == "MANUAL_REVIEW":
        final_status = "FLAGGED"
        base = f.get("reasoning") or "Ambiguous or borderline clinical scenario requires human utilization review."
        detail = base
        if bmi is not None:
            detail = f"BMI {bmi:.2f}. {detail}"
        if evidence:
            detail += f" Evidence term: '{evidence}'."
        reason = f"AI Uncertain. {detail}".strip()

        is_parsing_error = "Parsing Error" in base or "Parsing Error" in detail
        ambiguous_appeal_hit = evidence and any(term in evidence.lower() for term in AMBIGUOUS_APPEAL_TERMS)

        if ambiguous_appeal_hit:
            appeal_note = (
                "Manual review triggered by an ambiguous, non-qualifying term "
                f"('{evidence}'). If applicable, document a clearly qualifying comorbidity "
                "(e.g., HTN, T2DM, dyslipidemia, OSA) or clarify the diagnosis (e.g., specify OSA vs generic sleep apnea) "
                "before resubmission or appeal."
            )
        elif is_parsing_error:
            appeal_note = (
                "This case could not be fully evaluated due to a system processing error. "
                "Please review the patient's chart manually to verify BMI, weight-related comorbidities, "
                "and any safety exclusions (MTC, MEN2, pregnancy, concurrent GLP-1 use). "
                "A draft Prior Authorization template has been provided to assist with documentation."
            )
        else:
            appeal_note = (
                "This case requires human utilization review due to borderline or ambiguous clinical criteria. "
                "Please verify the patient's BMI and weight-related comorbidities meet policy requirements. "
                "A draft Prior Authorization template has been provided to assist with the review process."
            )

        logger.info(
            f"Drafting PA template for MANUAL_REVIEW case (BMI={bmi}, evidence='{evidence}', parsing_error={is_parsing_error})"
        )

        if is_parsing_error:
            denial_context = (
                "Case flagged for manual review due to a system processing error. "
                "The automated clinical audit could not fully parse the patient data. "
                f"Available information: {reason}"
            )
        elif ambiguous_appeal_hit:
            denial_context = "Case flagged for manual review due to an ambiguous weight-related risk term. " + reason
        else:
            denial_context = "Case flagged for manual review due to borderline clinical scenario. " + reason

        letter = generate_appeal_letter(p_data, denial_context, f)
        if letter is not None:
            appeal_letter = letter
            reason += " A draft Prior Authorization template has been generated for provider review."
        else:
            appeal_letter = _generate_fallback_pa_template(p_data, reason, f)
            if appeal_letter:
                reason += " A basic Prior Authorization template has been generated for provider review."

    elif verdict in ("DENIED_BENEFIT_EXCLUSION", "DENIED_OTHER"):
        # Keep deterministic-first posture: these are administrative-ish and should not be over-argued.
        final_status = "DENIED"
        base = f.get("reasoning") or f"Denied due to {verdict.replace('_', ' ').lower()}."
        reason = with_bmi_prefix(f"Denied. {base}")

    else:
        final_status = "FLAGGED"
        reason = f"AI produced unknown verdict '{verdict}'. Route to manual utilization review."

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
            raise RuntimeError("data_medications.csv not loaded (run chaos_monkey.py first).")

        wegovy_rows = df_meds[df_meds["medication_name"].str.contains("Wegovy", na=False)]
        if wegovy_rows.empty:
            raise RuntimeError("No Wegovy entries found in data_medications.csv.")

        target = wegovy_rows.iloc[0]["patient_id"]
        res = app.invoke({"patient_id": target, "drug_requested": "Wegovy"})
        logger.info(f"FINAL OUTPUT: {res.get('final_decision')} | {res.get('reasoning')}")
        if res.get("appeal_note"):
            logger.info(f"APPEAL NOTE: {res['appeal_note']}")
        if res.get("appeal_letter"):
            logger.info(f"APPEAL LETTER: {res['appeal_letter']}")
    except Exception as e:
        logger.error("Test run failed: %s", e)

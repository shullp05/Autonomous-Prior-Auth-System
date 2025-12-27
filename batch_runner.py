"""
batch_runner.py - Production Batch Processing for Prior Authorization Audits

This module orchestrates the batch evaluation of prior authorization requests,
supporting both LLM-augmented and deterministic (LLM-free) processing modes.

Features:
    - Batch processing of all Wegovy claims in the dataset
    - Dual execution modes: LLM-augmented or deterministic
    - Full traceability with run metadata (model versions, timestamps)
    - Automatic governance audit execution after batch completion
    - Performance timing per claim for benchmarking
    - Artifacts isolated in /output directory

Execution Modes:
    LLM Mode (default):
        Uses the full LangGraph agent with RAG retrieval, LLM reasoning,
        and appeal letter generation.

    Deterministic Mode (PA_USE_DETERMINISTIC=true):
        Uses the pure Python policy engine for 100% reproducible results.
        ~4000x faster than LLM mode, ideal for compliance audits.

Output:
    Generates output/dashboard_data.json with structure:
    {
        "metadata": { timestamp, mode, models, policy_version, ... },
        "results": [ { patient_id, status, reason, appeal_letter, ... }, ... ]
    }

Usage:
    # LLM mode (default)
    $ python batch_runner.py

    # Deterministic mode
    $ PA_USE_DETERMINISTIC=true python batch_runner.py

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from offline_mode import enforce_offline

# Enforce offline mode immediately if configured
from offline_mode import enforce_offline
from policy_utils import format_criteria_list

from audit_logger import get_audit_logger
from policy_engine import evaluate_eligibility
from policy_snapshot import POLICY_ID, SNAPSHOT_PATH, load_policy_snapshot
from schema_validation import validate_policy_snapshot

_audit_logger = get_audit_logger()

from config import (
    AUDIT_MODEL_FLAVOR,
    AUDIT_MODEL_NAME,
    AUDIT_MODEL_OPTIONS,
    AUDIT_MODEL_RAM_GB,
    CLAIM_VALUE_USD,
    PA_PRACTICE_NAME,
    PA_PROVIDER_NAME,
    PA_PROVIDER_NPI,
    USE_DETERMINISTIC,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration Constants
# ------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "dashboard_data.json"

DRUG_QUERY: str = "Wegovy"  # Target medication for PA evaluation
SNAPSHOT = load_policy_snapshot(SNAPSHOT_PATH, POLICY_ID)
validate_policy_snapshot(SNAPSHOT)
POLICY_VERSION: str = SNAPSHOT["policy_id"]
DASHBOARD_PUBLIC_DIR = Path("dashboard/public")
TRACE_FILE = Path(".last_model_trace.json")


def _now_iso() -> str:
    """Generate ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mirror_to_dashboard(src_path: Path) -> None:
    """Copy the latest artifact into dashboard/public for the Vite client."""
    if not src_path.exists():
        logger.warning("Skip mirroring '%s' – file not found", src_path)
        return

    try:
        DASHBOARD_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        dest = DASHBOARD_PUBLIC_DIR / src_path.name
        shutil.copy2(src_path, dest)
        logger.info("Mirrored '%s' to '%s'", src_path.name, dest)
    except Exception as exc:
        logger.error("Failed to mirror '%s' to dashboard/public: %s", src_path.name, exc)


def _require_provider_context_if_enforced() -> None:
    """
    Provider-facing system: do NOT allow generating letters without provider/practice context.

    Default behavior:
      - Enforced unless PA_ENFORCE_PROVIDER_CONTEXT=false
      - Auto-skipped under CI/pytest (CI=1 or PYTEST_CURRENT_TEST is set)
    """
    enforce = os.getenv("PA_ENFORCE_PROVIDER_CONTEXT", "true").lower() == "true"
    if not enforce:
        return

    in_test_env = bool(os.getenv("PYTEST_CURRENT_TEST")) or bool(os.getenv("CI"))
    if in_test_env:
        return

    # Use imported config values (which have defaults), not raw os.getenv()
    # since .env files aren't auto-loaded into OS environment
    required = {
        "PA_PROVIDER_NAME": PA_PROVIDER_NAME,
        "PA_PROVIDER_NPI": PA_PROVIDER_NPI,
        "PA_PRACTICE_NAME": PA_PRACTICE_NAME,
    }
    missing = [k for k, v in required.items() if not v.strip()]
    if missing:
        raise RuntimeError(
            "Provider context enforcement is enabled, but required environment variables are missing: "
            + ", ".join(missing)
            + ". Set them or disable with PA_ENFORCE_PROVIDER_CONTEXT=false."
        )


def _normalize_status(raw_verdict: str | None) -> str:
    """
    Normalize engine outputs to a stable set used by UI + governance:
      - APPROVED
      - DENIED
      - FLAGGED
      - PROVIDER_ACTION_REQUIRED
    """
    v = (raw_verdict or "").strip().upper()
    if v == "APPROVED":
        return "APPROVED"
    if v == "MANUAL_REVIEW":
        return "FLAGGED"
    if v == "CDI_REQUIRED":
        return "CDI_REQUIRED"
    if v == "SAFETY_SIGNAL_NEEDS_REVIEW":
        return "SAFETY_SIGNAL_NEEDS_REVIEW"
    if v == "DENIED_MISSING_INFO":
        return "PROVIDER_ACTION_REQUIRED"
    if v.startswith("DENIED"):
        return "DENIED"
    # Anything unexpected should route to manual review posture
    return ""





def _load_patient_data(
    pid: str,
    df_patients: pd.DataFrame,
    df_obs: pd.DataFrame,
    df_conds: pd.DataFrame,
    df_meds: pd.DataFrame,
) -> dict[str, Any]:
    """
    Load and aggregate patient data for the deterministic policy engine.

    Extracts demographics, BMI (from direct observation or height/weight calculation),
    conditions, and medications for a given patient.

    Args:
        pid: Patient identifier.
        df_patients: Patients DataFrame with demographics (name, dob).
        df_obs: Observations DataFrame with BMI/Height/Weight data.
        df_conds: Conditions DataFrame with diagnoses.
        df_meds: Medications DataFrame with active prescriptions.

    Returns:
        Dictionary with keys: name, dob, latest_bmi, conditions, meds
    """
    pid_str = str(pid)

    # Demographics (provider-facing letters need this)
    name = ""
    dob = ""
    try:
        pat_rows = df_patients[df_patients["patient_id"].astype(str) == pid_str]
        if not pat_rows.empty:
            row = pat_rows.iloc[0].to_dict()
            name = str(row.get("name", "") or "")
            dob = str(row.get("dob", "") or "")
    except Exception as e:
        # Keep as empty strings but log warning; provider-context enforcement is handled separately
        logger.warning(f"Error loading patient demographics for {pid}: {e}")
        pass

    # Get latest BMI
    p_obs = df_obs[df_obs["patient_id"].astype(str) == pid_str].copy()
    bmi_obs = p_obs[p_obs["type"] == "BMI"].copy()

    if not bmi_obs.empty:
        if "date" in bmi_obs.columns:
            bmi_obs["date_parsed"] = pd.to_datetime(bmi_obs["date"], errors="coerce")
            bmi_obs = bmi_obs.sort_values("date_parsed", ascending=False)
        latest_bmi = str(bmi_obs.iloc[0]["value"])
    else:
        # Try to calculate from height/weight
        ht_obs = p_obs[p_obs["type"] == "Height"].copy()
        wt_obs = p_obs[p_obs["type"] == "Weight"].copy()

        # Sort by date if present
        if "date" in ht_obs.columns:
            ht_obs["date_parsed"] = pd.to_datetime(ht_obs["date"], errors="coerce")
            ht_obs = ht_obs.sort_values("date_parsed", ascending=False)
        if "date" in wt_obs.columns:
            wt_obs["date_parsed"] = pd.to_datetime(wt_obs["date"], errors="coerce")
            wt_obs = wt_obs.sort_values("date_parsed", ascending=False)

        if not ht_obs.empty and not wt_obs.empty:
            try:
                ht = float(ht_obs.iloc[0]["value"]) / 100.0  # cm to m
                wt = float(wt_obs.iloc[0]["value"])
                if ht > 0:
                    calculated_bmi = wt / (ht**2)
                    latest_bmi = f"{calculated_bmi:.1f} (Calculated)"
                else:
                    latest_bmi = "MISSING_DATA"
            except (ValueError, TypeError):
                latest_bmi = "MISSING_DATA"
        else:
            latest_bmi = "MISSING_DATA"

    # Get conditions
    # Get conditions (full details for coding integrity)
    cond_rows = df_conds[df_conds["patient_id"].astype(str) == pid_str]
    # Ensure columns exist (in case of legacy CSV)
    cols = ["condition_name"]
    if "icd10_dx" in cond_rows.columns:
        cols.append("icd10_dx")
    if "icd10_bmi" in cond_rows.columns:
        cols.append("icd10_bmi")

    conditions = cond_rows[cols].to_dict(orient="records")

    # Get medications
    meds = (
        df_meds[df_meds["patient_id"].astype(str) == pid_str]["medication_name"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return {
        "patient_id": pid_str,
        "name": name,
        "dob": dob,
        "latest_bmi": latest_bmi,
        "conditions": conditions,
        "meds": meds,
    }


def run_batch() -> None:
    """
    Execute batch evaluation of all Wegovy prior authorization requests.

    Processes all patients with Wegovy prescriptions, evaluates eligibility
    using either LLM or deterministic mode, and generates output for the
    dashboard and governance audit.

    Side Effects:
        - Writes output/dashboard_data.json with results and metadata
        - Triggers governance_audit.run_governance_audit()
        - Logs progress to configured logger

    Raises:
        FileNotFoundError: If required CSV data files are missing.
    """
    # Provider-facing enforcement (skip in CI/pytest unless explicitly enabled)
    _require_provider_context_if_enforced()

    # Required data files
    required_files = [
        "output/data_patients.csv",
        "output/data_medications.csv",
        "output/data_observations.csv",
        "output/data_conditions.csv",
    ]
    missing_files = [p for p in required_files if not os.path.exists(p)]
    if missing_files:
        logger.error("Missing required data files: %s", ", ".join(missing_files))
        return

    df_patients = pd.read_csv("output/data_patients.csv")
    df_meds = pd.read_csv("output/data_medications.csv")
    df_obs = pd.read_csv("output/data_observations.csv")
    df_conds = pd.read_csv("output/data_conditions.csv")

    target_meds = df_meds[df_meds["medication_name"].str.contains(DRUG_QUERY, case=False, na=False)]
    target_ids = target_meds["patient_id"].dropna().astype(str).unique().tolist()
    
    # Argparse here or assume global? Better to parse here.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", help="Run only for specific patient UUID")
    args, _ = parser.parse_known_args()
    if args.patient_id:
        if args.patient_id in target_ids:
            target_ids = [args.patient_id]
        else:
            logger.warning(f"Patient {args.patient_id} not found in Wegovy claims.")
            target_ids = []
            
    total_claims = len(target_ids)
    total_claims = len(target_ids)

    mode_str = "DETERMINISTIC (no LLM)" if USE_DETERMINISTIC else "LLM-AUGMENTED"
    logger.info("Batch starting [%s]: Found %d %s claims -> %s", mode_str, total_claims, DRUG_QUERY, OUTPUT_PATH)

    # Init Agent if needed
    agent = None
    if not USE_DETERMINISTIC:
        from agent_logic import _ensure_data_loaded, build_agent

        # Ensure agent module data is loaded before we start building or invoking
        _ensure_data_loaded()
        agent = build_agent()

    results: list[dict[str, Any]] = []
    interrupted = False

    def save_results(partial: bool = False) -> None:
        """Save results to JSON file. Called on completion or interrupt."""
        status_suffix = " (PARTIAL - interrupted)" if partial else ""

        # Phase 9.5: Add aggregate metrics to metadata
        # Calculate CDI Count and Revenue at Risk
        cdi_count = sum(1 for r in results if r.get("status") == "CDI_REQUIRED")
        revenue_at_risk = cdi_count * float(CLAIM_VALUE_USD)

        output: dict[str, Any] = {
            "metadata": {
                "timestamp": _now_iso(),
                "mode": "deterministic" if USE_DETERMINISTIC else "llm",
                "policy_version": POLICY_VERSION,
                "policy_source_hash": SNAPSHOT.get("source_hash"),
                "policy_effective_date": SNAPSHOT.get("effective_date"),
                "model_name": "DETERMINISTIC_ENGINE" if USE_DETERMINISTIC else AUDIT_MODEL_NAME,
                "model_flavor": "N/A" if USE_DETERMINISTIC else AUDIT_MODEL_FLAVOR,
                "total_claims": total_claims,
                "drug_queried": DRUG_QUERY,
                "cdi_required_count": cdi_count,
                "revenue_at_risk_usd": revenue_at_risk,
            },
            "results": results,
        }

        durations = [
            r.get("duration_ms")
            for r in results
            if isinstance(r.get("duration_ms"), (int, float))
        ]
        if durations:
            durations_ms = sorted(float(d) for d in durations)

            def percentile(values: list[float], pct: float) -> float | None:
                if not values:
                    return None
                if len(values) == 1:
                    return values[0]
                k = (len(values) - 1) * pct
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return values[int(k)]
                return values[f] * (c - k) + values[c] * (k - f)

            avg_ms = sum(durations_ms) / len(durations_ms)
            p50 = percentile(durations_ms, 0.5)
            p95 = percentile(durations_ms, 0.95)
            output["metadata"]["avg_duration_ms"] = round(avg_ms, 2)
            output["metadata"]["p50_duration_ms"] = round(p50, 2) if p50 is not None else None
            output["metadata"]["p95_duration_ms"] = round(p95, 2) if p95 is not None else None

        if not USE_DETERMINISTIC:
            output["metadata"]["ram_required_gb"] = AUDIT_MODEL_RAM_GB
            if AUDIT_MODEL_OPTIONS:
                output["metadata"]["model_params"] = AUDIT_MODEL_OPTIONS

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        # Audit Log the Batch Summary
        _audit_logger.log_event(
            event_type="BATCH_COMPLETE",
            actor="batch_runner",
            details={
                "total_claims": total_claims,
                "approved": sum(1 for r in results if r['status'] == 'APPROVED'),
                "denied": sum(1 for r in results if r['status'] == 'DENIED'),
                "flagged": sum(1 for r in results if r['status'] == 'FLAGGED'),
                "cdi_required": cdi_count,
                "revenue_at_risk_usd": revenue_at_risk,
                "duration_sec": round(sum(r.get("duration_sec", 0) for r in results), 2)
            }
        )

        logger.info(
            "Results saved to '%s'%s. Processed %d/%d claims.",
            OUTPUT_PATH,
            status_suffix,
            len(results),
            total_claims,
        )

        if not partial:
            _mirror_to_dashboard(OUTPUT_PATH)

    def process_patient(pid: str) -> dict[str, Any]:
        start_time = time.time()

        # Defaults to keep return object well-formed
        status = "FLAGGED"
        reason = "Unknown error"
        appeal = None
        appeal_note = None
        policy_path = None
        decision_type = None
        safety_exclusion_code = None
        ambiguity_code = None

        # Phase 9.5 and 9.3 fields
        clinical_eligible = False
        admin_ready = False
        missing_anchor_code = None
        safety_context = None
        safety_confidence = None

        raw_verdict = None
        det_result_obj = None
        llm_response_obj: dict[str, Any] | None = None

        try:
            if USE_DETERMINISTIC:
                patient_data = _load_patient_data(pid, df_patients, df_obs, df_conds, df_meds)
                det_result_obj = evaluate_eligibility(patient_data)

                raw_verdict = det_result_obj.verdict
                status = _normalize_status(raw_verdict)
                reason = det_result_obj.reasoning
                policy_path = det_result_obj.policy_path
                decision_type = det_result_obj.decision_type
                safety_exclusion_code = det_result_obj.safety_exclusion_code
                ambiguity_code = det_result_obj.ambiguity_code

                # Capture new fields
                clinical_eligible = det_result_obj.clinical_eligible
                admin_ready = det_result_obj.admin_ready
                missing_anchor_code = det_result_obj.missing_anchor_code
                safety_context = det_result_obj.safety_context
                safety_confidence = det_result_obj.safety_confidence

                # AUDIT LOGGING (Deterministic Mode)
                # LLM mode handles this in agent_logic.py, but deterministic mode must log here.
                _audit_logger.log_event(
                    event_type="DECISION",
                    details={
                        "patient_id": pid,
                        "verdict": status,
                        "raw_verdict": raw_verdict,
                        "reasoning": reason,
                        "model_used": "DETERMINISTIC_ENGINE",
                        "policy_path": policy_path,
                        "bmi": det_result_obj.bmi_numeric,
                    },
                    patient_id=pid
                )

                # Letters: deterministic mode should remain LLM-free when possible.
                # If agent_logic isn't importable (langchain deps missing), use safe deterministic templates.
                if status == "APPROVED":
                    try:
                        from agent_logic import generate_approved_letter

                        findings_dict = {
                            "bmi_numeric": det_result_obj.bmi_numeric,
                            "comorbidity_category": det_result_obj.comorbidity_category,
                            "policy_path": det_result_obj.policy_path,
                        }
                        patient_data["patient_id"] = pid
                        appeal = generate_approved_letter(patient_data, reason, findings_dict)
                        if appeal:
                            logger.info("Generated Letter of Medical Necessity for APPROVED patient %s", pid)
                    except ImportError as e:
                        logger.warning("Could not generate approved letter (missing deps): %s", e)
                        bmi_val = det_result_obj.bmi_numeric
                        comorbidity = det_result_obj.comorbidity_category or "NONE"
                        bmi_str = f"{bmi_val:.1f} kg/m²" if bmi_val is not None else "[See chart]"
                        comorbidity_line = (
                            f"Qualifying Comorbidity: {comorbidity}" if comorbidity != "NONE" else ""
                        )
                        appeal = f"""LETTER OF MEDICAL NECESSITY
Prior Authorization Request - Wegovy (Semaglutide 2.4mg)

To: Medical Director, Utilization Management

RE: Patient {pid}

Dear Medical Director,

I am writing to request prior authorization for Wegovy (semaglutide 2.4mg) for chronic weight management.

CLINICAL JUSTIFICATION:
{reason}

Patient's current BMI: {bmi_str}
{comorbidity_line}

Based on the clinical criteria outlined above, this patient meets coverage requirements for Wegovy.

I respectfully request approval of this prior authorization.

Sincerely,
_______________________________
Prescriber Signature / Date
"""
                elif status == "CDI_REQUIRED":
                     # Generate Physician Query Note (Deterministic)
                     criteria_list = format_criteria_list(
                         det_result_obj.bmi_numeric,
                         det_result_obj.found_diagnosis_string,
                         det_result_obj.found_e66_code,
                         det_result_obj.found_z68_code,
                         det_result_obj.evidence_quoted, # Comorbidity evidence
                         det_result_obj.missing_anchor_code,
                         None
                     )

                     appeal = f"""PHYSICIAN QUERY / CDI ALERT
{det_result_obj.physician_query_text}

{criteria_list}
"""
                     appeal_note = "Physician Query Generated"

                elif status == "DENIED":
                    bmi_val = det_result_obj.bmi_numeric
                    bmi_str = f"{bmi_val:.1f} kg/m²" if bmi_val is not None else "[Not documented]"
                    safety_info = ""
                    if det_result_obj.safety_exclusion_code:
                        safety_info = f"\n\nSAFETY EXCLUSION DETECTED:\n{det_result_obj.safety_exclusion_code}"

                    appeal = f"""PRIOR AUTHORIZATION DENIAL NOTIFICATION
Drug: Wegovy (Semaglutide 2.4mg) - Chronic Weight Management

To: Prescriber / Medical Records

RE: Patient {pid}

Dear Provider,

This letter serves as formal notification that the prior authorization request for Wegovy (semaglutide 2.4mg) has been DENIED.

DENIAL REASON:
{reason}

                DENIAL REASON:
{reason}

CLINICAL DATA REVIEWED:
{format_criteria_list(
    det_result_obj.bmi_numeric,
    det_result_obj.found_diagnosis_string,
    det_result_obj.found_e66_code,
    det_result_obj.found_z68_code,
    det_result_obj.evidence_quoted,
    det_result_obj.missing_anchor_code,
    det_result_obj.ambiguity_code
)}
{safety_info}

POLICY REFERENCE:
This determination was made in accordance with the applicable drug coverage policy criteria for GLP-1 receptor agonists indicated for chronic weight management.

APPEAL RIGHTS:
You may appeal this determination by submitting additional clinical documentation that addresses the criteria not met. Please include:
- Updated BMI measurement
- Documentation of qualifying comorbidities (if applicable)
- Attestation that no safety contraindications exist

For questions regarding this determination, please contact the Pharmacy Benefits Manager.

Sincerely,

Clinical Pharmacy Review Team
"""

            else:
                if agent is None:
                    raise RuntimeError("LLM mode requested but agent was not initialized.")

                llm_response_obj = agent.invoke({"patient_id": pid, "drug_requested": DRUG_QUERY})  # type: ignore[union-attr]

                # NOTE: In LLM mode, agent returns final_decision already normalized (APPROVED/DENIED/FLAGGED/PROVIDER_ACTION_REQUIRED)
                status = str(llm_response_obj.get("final_decision", "FLAGGED") or "FLAGGED")
                reason = str(llm_response_obj.get("reasoning", "") or "")
                appeal = llm_response_obj.get("appeal_letter", None)
                appeal_note = llm_response_obj.get("appeal_note", None)
                policy_path = llm_response_obj.get("policy_path")
                decision_type = llm_response_obj.get("decision_type")
                safety_exclusion_code = llm_response_obj.get("safety_exclusion_code")
                ambiguity_code = llm_response_obj.get("ambiguity_code")

                # Extract new fields from audit_findings
                audit_findings = llm_response_obj.get("audit_findings") or {}
                raw_verdict = audit_findings.get("verdict")
                clinical_eligible = audit_findings.get("clinical_eligible", False)
                admin_ready = audit_findings.get("admin_ready", False)
                missing_anchor_code = audit_findings.get("missing_anchor_code")
                safety_context = audit_findings.get("safety_context")
                safety_confidence = audit_findings.get("safety_confidence")

                # If APPROVED but no letter present, generate one (provider-facing)
                if status == "APPROVED" and not appeal:
                    try:
                        from agent_logic import generate_approved_letter

                        findings_dict = {
                            "bmi_numeric": audit_findings.get("bmi_numeric") if audit_findings else None,
                            "comorbidity_category": audit_findings.get("comorbidity_category", "NONE") if audit_findings else "NONE",
                            "policy_path": policy_path,
                            "found_diagnosis_string": audit_findings.get("found_diagnosis_string"),
                            "found_e66_code": audit_findings.get("found_e66_code"),
                            "found_z68_code": audit_findings.get("found_z68_code"),
                            "found_comorbidity_evidence": audit_findings.get("evidence_quoted"),
                        }

                        # Prefer agent-provided patient_data to avoid dataframe dependency drift
                        patient_data = llm_response_obj.get("patient_data") or {}
                        if not patient_data:
                            patient_data = _load_patient_data(pid, df_patients, df_obs, df_conds, df_meds)
                        patient_data["patient_id"] = pid

                        appeal = generate_approved_letter(patient_data, reason, findings_dict)
                        if appeal:
                            logger.info("Generated Letter of Medical Necessity for APPROVED patient %s", pid)
                    except Exception as e:
                        logger.warning("Could not generate approved letter for %s: %s", pid, e)

                        bmi_val = audit_findings.get("bmi_numeric") if audit_findings else None
                        comorbidity = audit_findings.get("comorbidity_category", "NONE") if audit_findings else "NONE"
                        bmi_str = f"{float(bmi_val):.1f} kg/m²" if isinstance(bmi_val, (int, float)) else "[See chart]"
                        comorbidity_line = (
                            f"Qualifying Comorbidity: {comorbidity}" if comorbidity and comorbidity != "NONE" else ""
                        )
                        appeal = f"""LETTER OF MEDICAL NECESSITY
Prior Authorization Request - Wegovy (Semaglutide 2.4mg)

To: Medical Director, Utilization Management

RE: Patient {pid}

Dear Medical Director,

I am writing to request prior authorization for Wegovy (semaglutide 2.4mg) for chronic weight management.

CLINICAL JUSTIFICATION:
{reason}

Patient's current BMI: {bmi_str}
{comorbidity_line}

Based on the clinical criteria outlined above, this patient meets coverage requirements for Wegovy.

I respectfully request approval of this prior authorization.

Sincerely,
_______________________________
Prescriber Signature / Date
"""

                elif status == "CDI_REQUIRED":
                     criteria_list = format_criteria_list(
                            audit_findings.get("bmi_numeric") if audit_findings else None,
                            audit_findings.get("found_diagnosis_string"),
                            audit_findings.get("found_e66_code"),
                            audit_findings.get("found_z68_code"),
                            audit_findings.get("evidence_quoted"),
                            audit_findings.get("missing_anchor_code"),
                            audit_findings.get("ambiguity_code")
                     )

                     # Construct query text if not present
                     query_body = reason
                     if not query_body:
                         query_body = "Missing administrative documentation required for approval."

                     appeal = f"""PHYSICIAN QUERY / CDI ALERT
{query_body}

{criteria_list}
"""
                     appeal_note = "Physician Query Generated"

                elif status in ["FLAGGED", "MANUAL_REVIEW", "PROVIDER_ACTION_REQUIRED"]:
                     criteria_list = format_criteria_list(
                            audit_findings.get("bmi_numeric") if audit_findings else None,
                            audit_findings.get("found_diagnosis_string"),
                            audit_findings.get("found_e66_code"),
                            audit_findings.get("found_z68_code"),
                            audit_findings.get("evidence_quoted"),
                            audit_findings.get("missing_anchor_code"),
                            audit_findings.get("ambiguity_code")
                     )

                     appeal = f"""MANUAL REVIEW REQUIRED
Status: {status}
Reason: {reason}

{criteria_list}
"""
                     appeal_note = "Manual Review Flagged"

                # If DENIED and no letter present, generate a denial notification
                elif status == "DENIED" and not appeal:
                    audit_findings = llm_response_obj.get("audit_findings") or {}
                    bmi_val = audit_findings.get("bmi_numeric") if audit_findings else None
                    bmi_str = f"{float(bmi_val):.1f} kg/m²" if isinstance(bmi_val, (int, float)) else "[Not documented]"
                    safety_info = ""
                    if safety_exclusion_code:
                        safety_info = f"\n\nSAFETY EXCLUSION DETECTED:\n{safety_exclusion_code}"

                    appeal = f"""PRIOR AUTHORIZATION DENIAL NOTIFICATION
Drug: Wegovy (Semaglutide 2.4mg) - Chronic Weight Management

To: Prescriber / Medical Records

RE: Patient {pid}

Dear Provider,

This letter serves as formal notification that the prior authorization request for Wegovy (semaglutide 2.4mg) has been DENIED.

                DENIAL REASON:
{reason}

CLINICAL DATA REVIEWED:
{format_criteria_list(
    audit_findings.get("bmi_numeric") if audit_findings else None,
    audit_findings.get("found_diagnosis_string"),
    audit_findings.get("found_e66_code"),
    audit_findings.get("found_z68_code"),
    audit_findings.get("evidence_quoted"),
    audit_findings.get("missing_anchor_code"),
    audit_findings.get("ambiguity_code")
)}
{safety_info}

POLICY REFERENCE:
This determination was made in accordance with the applicable drug coverage policy criteria for GLP-1 receptor agonists indicated for chronic weight management.

APPEAL RIGHTS:
You may appeal this determination by submitting additional clinical documentation that addresses the criteria not met. Please include:
- Updated BMI measurement
- Documentation of qualifying comorbidities (if applicable)
- Attestation that no safety contraindications exist

For questions regarding this determination, please contact the Pharmacy Benefits Manager.

Sincerely,

Clinical Pharmacy Review Team
"""

        except Exception as e:
            status = "FLAGGED"
            reason = str(e)
            logger.error("Error processing patient %s: %s", pid, e)

        duration_sec = time.time() - start_time
        duration_ms = int(duration_sec * 1000)

        # Safe extraction of BMI based on which mode ran
        final_bmi_val = None
        if USE_DETERMINISTIC and det_result_obj is not None:
            final_bmi_val = det_result_obj.bmi_numeric
        elif (not USE_DETERMINISTIC) and llm_response_obj is not None:
            final_bmi_val = (llm_response_obj.get("audit_findings") or {}).get("bmi_numeric")

        return {
            "patient_id": pid,
            "status": status,              # normalized: APPROVED / DENIED / FLAGGED / PROVIDER_ACTION_REQUIRED / CDI_REQUIRED
            "raw_verdict": raw_verdict,    # preserves engine output: DENIED_CLINICAL, MANUAL_REVIEW, etc
            "reason": reason,
            "appeal_letter": appeal,
            "appeal_note": appeal_note,
            "value": float(CLAIM_VALUE_USD),
            "duration_ms": duration_ms,    # Standardized unit
            "policy_path": policy_path,
            "decision_type": decision_type,
            "safety_exclusion_code": safety_exclusion_code,
            "ambiguity_code": ambiguity_code,
            "bmi_value": final_bmi_val,
            "clinical_eligible": clinical_eligible,
            "admin_ready": admin_ready,
            "missing_anchor_code": missing_anchor_code,
            "safety_context": safety_context,
            "safety_confidence": safety_confidence,
        }

    # Main processing loop with graceful shutdown handling
    try:
        for i, pid in enumerate(target_ids):
            res = process_patient(pid)
            results.append(res)
            logger.info("[%d/%d] Patient %s -> %s (%dms)", i + 1, total_claims, pid, res["status"], res["duration_ms"])
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("\n\nInterrupted by user. Saving %d partial results...", len(results))
        save_results(partial=True)
        logger.info("Partial results saved. You can resume or re-run later.")
        return

    # Save complete results
    save_results(partial=False)

    # Automatically run governance audit (only on complete runs)
    try:
        import governance_audit

        governance_audit.run_governance_audit()
        _mirror_to_dashboard(OUTPUT_DIR / "governance_report.json")
    except Exception as e:
        logger.error("Governance audit error: %s", e)
    finally:
        if TRACE_FILE.exists():
            _mirror_to_dashboard(TRACE_FILE)

    if interrupted:
        logger.warning("Run completed with interruption flag (unexpected state).")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_batch()

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
from typing import Any, Optional

import pandas as pd

from policy_engine import evaluate_eligibility
from policy_snapshot import POLICY_ID, SNAPSHOT_PATH, load_policy_snapshot
from schema_validation import validate_policy_snapshot

from config import (
    USE_DETERMINISTIC,
    AUDIT_MODEL_NAME,
    AUDIT_MODEL_FLAVOR,
    CLAIM_VALUE_USD,
    AUDIT_MODEL_RAM_GB,
    AUDIT_MODEL_OPTIONS,
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
# Enforce separation: generated files go to /output, not root
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
        logger.warning(f"Skip mirroring '{src_path}' – file not found")
        return

    try:
        DASHBOARD_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        dest = DASHBOARD_PUBLIC_DIR / src_path.name
        shutil.copy2(src_path, dest)
        logger.info(f"Mirrored '{src_path.name}' to '{dest}'")
    except Exception as exc:
        logger.error(f"Failed to mirror '{src_path.name}' to dashboard/public: {exc}")


def _load_patient_data(
    pid: str,
    df_obs: pd.DataFrame,
    df_conds: pd.DataFrame,
    df_meds: pd.DataFrame,
) -> dict[str, Any]:
    """
    Load and aggregate patient data for the deterministic policy engine.
    
    Extracts BMI (from direct observation or height/weight calculation),
    conditions, and medications for a given patient.
    
    Args:
        pid: Patient identifier.
        df_obs: Observations DataFrame with BMI/Height/Weight data.
        df_conds: Conditions DataFrame with diagnoses.
        df_meds: Medications DataFrame with active prescriptions.
    
    Returns:
        Dictionary with keys: latest_bmi, conditions, meds
    """
    # Get latest BMI
    p_obs = df_obs[df_obs["patient_id"] == pid].copy()
    bmi_obs = p_obs[p_obs["type"] == "BMI"].copy()
    
    if not bmi_obs.empty:
        if "date" in bmi_obs.columns:
            bmi_obs["date_parsed"] = pd.to_datetime(bmi_obs["date"], errors="coerce")
            bmi_obs = bmi_obs.sort_values("date_parsed", ascending=False)
        latest_bmi = str(bmi_obs.iloc[0]["value"])
    else:
        # Try to calculate from height/weight
        ht_obs = p_obs[p_obs["type"] == "Height"]
        wt_obs = p_obs[p_obs["type"] == "Weight"]
        if not ht_obs.empty and not wt_obs.empty:
            try:
                ht = float(ht_obs.iloc[0]["value"]) / 100.0  # cm to m
                wt = float(wt_obs.iloc[0]["value"])
                if ht > 0:
                    calculated_bmi = wt / (ht ** 2)
                    latest_bmi = f"{calculated_bmi:.1f} (Calculated)"
                else:
                    latest_bmi = "MISSING_DATA"
            except (ValueError, TypeError):
                latest_bmi = "MISSING_DATA"
        else:
            latest_bmi = "MISSING_DATA"
    
    # Get conditions
    conditions = (
        df_conds[df_conds["patient_id"] == pid]["condition_name"]
        .dropna()
        .astype(str)
        .tolist()
    )
    
    # Get medications
    meds = (
        df_meds[df_meds["patient_id"] == pid]["medication_name"]
        .dropna()
        .astype(str)
        .tolist()
    )
    
    return {
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
    # Load raw data (assumed to be in CWD as per typical setup)
    if not os.path.exists("output/data_medications.csv"):
        logger.error("Missing output/data_medications.csv. Cannot run batch.")
        return

    df_meds = pd.read_csv("output/data_medications.csv")
    df_obs = pd.read_csv("output/data_observations.csv") if USE_DETERMINISTIC else None
    df_conds = pd.read_csv("output/data_conditions.csv") if USE_DETERMINISTIC else None

    target_meds = df_meds[df_meds["medication_name"].str.contains(DRUG_QUERY, case=False, na=False)]
    target_ids = target_meds["patient_id"].dropna().astype(str).unique().tolist()
    total_claims = len(target_ids)

    mode_str = "DETERMINISTIC (no LLM)" if USE_DETERMINISTIC else "LLM-AUGMENTED"
    logger.info(f"Batch starting [{mode_str}]: Found {total_claims} {DRUG_QUERY} claims -> {OUTPUT_PATH}")

    # Init Agent if needed
    if not USE_DETERMINISTIC:
        from agent_logic import build_agent, _ensure_data_loaded
        # Ensure agent module data is loaded before we start building or invoking
        _ensure_data_loaded()
        agent = build_agent()
    else:
        agent = None
        
    results = []
    interrupted = False

    def save_results(partial: bool = False) -> None:
        """Save results to JSON file. Called on completion or interrupt."""
        status_suffix = " (PARTIAL - interrupted)" if partial else ""
        output = {
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
            },
            "results": results,
        }

        durations = [r.get("duration_sec") for r in results if isinstance(r.get("duration_sec"), (int, float))]
        if durations:
            durations_ms = sorted(d * 1000 for d in durations)

            def percentile(values, pct):
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

        logger.info(f"Results saved to '{OUTPUT_PATH}'{status_suffix}. Processed {len(results)}/{total_claims} claims.")

        if not partial:
            _mirror_to_dashboard(OUTPUT_PATH)

    def process_patient(pid: str):
        start_time = time.time()

        # Initialize variables explicitly to prevent UnboundLocalError
        decision = "ERROR"
        reason = "Unknown error"
        appeal = None
        appeal_note = None
        policy_path = None
        decision_type = None
        safety_exclusion_code = None
        ambiguity_code = None
        
        # Containers for object access later
        det_result_obj = None 
        llm_response_obj = None

        try:
            if USE_DETERMINISTIC:
                # Pure Python deterministic evaluation - no LLM
                patient_data = _load_patient_data(pid, df_obs, df_conds, df_meds)
                det_result_obj = evaluate_eligibility(patient_data)
                
                decision = det_result_obj.verdict
                reason = det_result_obj.reasoning
                policy_path = det_result_obj.policy_path
                decision_type = det_result_obj.decision_type
                safety_exclusion_code = det_result_obj.safety_exclusion_code
                ambiguity_code = det_result_obj.ambiguity_code
            else:
                # LLM-augmented evaluation
                llm_response_obj = agent.invoke({"patient_id": pid, "drug_requested": DRUG_QUERY})
                
                decision = llm_response_obj.get("final_decision", "ERROR")
                reason = llm_response_obj.get("reasoning", "") or ""
                appeal = llm_response_obj.get("appeal_letter", None)
                appeal_note = llm_response_obj.get("appeal_note", None)
                policy_path = llm_response_obj.get("policy_path")
                decision_type = llm_response_obj.get("decision_type")
                safety_exclusion_code = llm_response_obj.get("safety_exclusion_code")
                ambiguity_code = llm_response_obj.get("ambiguity_code")
        except Exception as e:
            decision = "ERROR"
            reason = str(e)
            logger.error(f"Error processing patient {pid}: {e}")

        duration = time.time() - start_time

        # Safe extraction of BMI based on which mode ran
        final_bmi_val = None
        if USE_DETERMINISTIC and det_result_obj:
            final_bmi_val = det_result_obj.bmi_numeric
        elif not USE_DETERMINISTIC and llm_response_obj:
            final_bmi_val = llm_response_obj.get("audit_findings", {}).get("bmi_numeric")

        return {
            "patient_id": pid,
            "status": decision,         # used by frontend + governance
            "reason": reason,           # used by frontend
            "appeal_letter": appeal,    # used by "Review Draft" button
            "appeal_note": appeal_note, # provider guidance for FLAGGED cases
            "value": float(CLAIM_VALUE_USD),
            "duration_sec": round(duration, 2),
            "policy_path": policy_path,
            "decision_type": decision_type,
            "safety_exclusion_code": safety_exclusion_code,
            "ambiguity_code": ambiguity_code,
            "bmi_value": final_bmi_val,
        }

    # Main processing loop with graceful shutdown handling
    try:
        for i, pid in enumerate(target_ids):
            res = process_patient(pid)
            results.append(res)
            logger.info(f"[{i+1}/{total_claims}] Patient {pid} -> {res['status']} ({res['duration_sec']}s)")
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(f"\n\nInterrupted by user. Saving {len(results)} partial results...")
        save_results(partial=True)
        logger.info("Partial results saved. You can resume or re-run later.")
        return  # Exit without running governance audit on partial data

    # Save complete results
    save_results(partial=False)

    # Automatically run governance audit (only on complete runs)
    # The audit now looks in output/dashboard_data.json and writes to output/governance_report.json
    try:
        import governance_audit
        governance_audit.run_governance_audit()
        _mirror_to_dashboard(OUTPUT_DIR / "governance_report.json")
    except Exception as e:
        logger.error(f"Governance audit error: {e}")
    finally:
        if TRACE_FILE.exists():
            _mirror_to_dashboard(TRACE_FILE)


if __name__ == "__main__":
    # Configure logging for batch runs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_batch()
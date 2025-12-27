# governance_audit.py

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Centralized policy constants (single source of truth)
from policy_constants import (
    AMBIGUOUS_BP,
    AMBIGUOUS_DIABETES,
    AMBIGUOUS_OBESITY,
    AMBIGUOUS_SLEEP_APNEA,
    AMBIGUOUS_THYROID,
    BMI_MAX_REASONABLE,
    BMI_MIN_REASONABLE,
    BMI_OBESE_THRESHOLD,
    BMI_OVERWEIGHT_THRESHOLD,
    PROHIBITED_GLP1,
    QUALIFYING_CVD_ABBREVS,
    QUALIFYING_CVD_PHRASES,
    QUALIFYING_HYPERTENSION,
    QUALIFYING_LIPIDS,
    QUALIFYING_OSA,
    QUALIFYING_T2DM,
    SAFETY_GI_MOTILITY,
    SAFETY_HYPERSENSITIVITY,
    SAFETY_MTC_MEN2,
    SAFETY_PANCREATITIS,
    SAFETY_PREGNANCY_LACTATION,
    SAFETY_SUICIDALITY,
)
from policy_utils import expand_safety_variants, has_word_boundary, is_snf_phrase, matches_term, normalize
from schema_validation import validate_governance_report

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Default locations align with your ETL + batch_runner + chaos_monkey outputs
OUTPUT_DIR = Path(os.getenv("PA_OUTPUT_DIR", "output"))
INPUT_FILE = OUTPUT_DIR / "dashboard_data.json"
REPORT_FILE = OUTPUT_DIR / "governance_report.json"

PATIENTS_CSV = OUTPUT_DIR / "data_patients.csv"
OBS_CSV = OUTPUT_DIR / "data_observations.csv"
CONDS_CSV = OUTPUT_DIR / "data_conditions.csv"
MEDS_CSV = OUTPUT_DIR / "data_medications.csv"

# -----------------------------
# Policy thresholds dict (for backward compatibility)
# Values now sourced from policy_constants.py
# -----------------------------
POLICY = {
    "bmi_obesity_threshold": float(BMI_OBESE_THRESHOLD),
    "bmi_overweight_threshold": float(BMI_OVERWEIGHT_THRESHOLD),
    "max_reasonable_bmi": float(BMI_MAX_REASONABLE),
    "min_reasonable_bmi": float(BMI_MIN_REASONABLE),
}


def _any_match_in_conditions(conds: list[str], phrases: list[str]) -> str | None:
    """Return the first original condition string that matches any phrase."""
    if not conds or not phrases:
        return None
    for c in conds:
        c_norm = normalize(c)
        for p in phrases:
            if matches_term(c_norm, p):
                return c
    return None


def _has_osa_specific(conds: list[str]) -> str | None:
    """
    Only Obstructive Sleep Apnea qualifies: must include 'obstructive' or explicit 'OSA'.
    Generic 'sleep apnea' alone is ambiguous and should NOT qualify.
    """
    for c in conds:
        c_norm = normalize(c)
        if "obstructive" in c_norm or has_word_boundary(c_norm, "osa"):
            for term in QUALIFYING_OSA:
                if matches_term(c_norm, term):
                    return c
    return None


def _is_generic_sleep_apnea_only(conds: list[str]) -> str | None:
    """Detect generic 'sleep apnea' as an ambiguous term for audit/debug."""
    for c in conds:
        c_norm = normalize(c)
        if "sleep apnea" in c_norm and ("obstruct" not in c_norm) and (not has_word_boundary(c_norm, "osa")):
            return c
    return None


def _is_t2_diabetes(conds: list[str]) -> str | None:
    """T2DM qualifies; prediabetes terms explicitly do not."""
    for c in conds:
        if _any_match_in_conditions([c], AMBIGUOUS_DIABETES):
            continue
        if _any_match_in_conditions([c], QUALIFYING_T2DM):
            return c
    return None


def _is_hypertension(conds: list[str]) -> str | None:
    for c in conds:
        if _any_match_in_conditions([c], AMBIGUOUS_BP):
            continue
        if _any_match_in_conditions([c], QUALIFYING_HYPERTENSION):
            return c
    return None


def _is_lipids(conds: list[str]) -> str | None:
    return _any_match_in_conditions(conds, QUALIFYING_LIPIDS)


def _is_cvd(conds: list[str]) -> str | None:
    for c in conds:
        c_norm = normalize(c)
        for phrase in QUALIFYING_CVD_PHRASES:
            if matches_term(c_norm, phrase):
                return c
        for abbrev in QUALIFYING_CVD_ABBREVS:
            if has_word_boundary(c_norm, normalize(abbrev)):
                return c
    return None


def _detect_safety_exclusion(conds: list[str], meds: list[str]) -> tuple[bool, str]:
    """
    Policy-aligned safety exclusion detection.
    Returns: (is_excluded, evidence_string)
    """
    # Condition-based exclusions
    exclusion_map = [
        (SAFETY_MTC_MEN2, "MTC/MEN2"),
        (SAFETY_PREGNANCY_LACTATION, "Pregnancy"),
        (SAFETY_HYPERSENSITIVITY, "Hypersensitivity"),
        (SAFETY_PANCREATITIS, "Pancreatitis"),
        (SAFETY_SUICIDALITY, "Suicidality"),
        (SAFETY_GI_MOTILITY, "GI Motility"),
    ]

    for term_list, label in exclusion_map:
        for c in conds:
            c_norm = normalize(c)
            for term_raw in term_list:
                for term in expand_safety_variants(term_raw):
                    if label == "Pregnancy":
                        if ("pregnan" in term) or ("breast" in term) or ("nursing" in term) or ("lactat" in term):
                            if is_snf_phrase(c):
                                continue
                    if matches_term(c_norm, term):
                        return True, c

    # Duplicate therapy (med-based)
    meds_norm = [normalize(m) for m in meds]
    has_wegovy = any("wegovy" in m for m in meds_norm)

    for m in meds:
        m_low = normalize(m)
        if "wegovy" in m_low:
            continue
        for bad in PROHIBITED_GLP1:
            for variant in expand_safety_variants(bad):
                if matches_term(m_low, variant):
                    # protect against generic semaglutide strings if Wegovy is also present
                    if variant.startswith("semaglutide") and has_wegovy:
                        continue
                    return True, m

    return False, ""


def _get_latest_numeric_obs(df_obs: pd.DataFrame, pid: str, obs_type: str) -> float | None:
    if df_obs.empty:
        return None
    if "patient_id" not in df_obs.columns or "type" not in df_obs.columns:
        return None

    p_obs = df_obs[df_obs["patient_id"] == pid].copy()
    p_obs = p_obs[p_obs["type"] == obs_type].copy()
    if p_obs.empty:
        return None

    if "date" in p_obs.columns:
        p_obs["date_parsed"] = pd.to_datetime(p_obs["date"], errors="coerce")
        p_obs = p_obs.sort_values(["date_parsed"], ascending=False)

    try:
        return float(p_obs.iloc[0]["value"])
    except Exception:
        return None


def _calculate_bmi_ground_truth(df_obs: pd.DataFrame, pid: str) -> float | None:
    bmi = _get_latest_numeric_obs(df_obs, pid, "BMI")
    if bmi is not None:
        if POLICY["min_reasonable_bmi"] <= bmi <= POLICY["max_reasonable_bmi"]:
            return round(float(bmi), 1)
        return None

    wt = _get_latest_numeric_obs(df_obs, pid, "Weight")
    ht = _get_latest_numeric_obs(df_obs, pid, "Height")
    if wt is None or ht is None:
        return None

    try:
        height_m = float(ht) / 100.0
        if height_m <= 0:
            return None
        bmi_calc = float(wt) / (height_m**2)
        if POLICY["min_reasonable_bmi"] <= bmi_calc <= POLICY["max_reasonable_bmi"]:
            return round(float(bmi_calc), 1)
        return None
    except Exception:
        return None


def calculate_ground_truth_eligibility(
    pid: str,
    df_obs: pd.DataFrame,
    df_conds: pd.DataFrame,
    df_meds: pd.DataFrame,
) -> dict[str, Any]:
    """
    Returns a policy-aligned deterministic truth object.
    - eligible: True / False / None (None means unknown due to missing BMI)
    - evidence: key string that drove decision (comorbidity or safety exclusion), when applicable

    IMPORTANT: This is intended to match the CURRENT deterministic engine behavior:
      - Safety exclusions deny
      - BMI missing => unknown
      - BMI >= obese threshold => requires adult obesity diagnosis string (per current policy_engine.py)
      - BMI < overweight threshold => deny
      - BMI 27–29.9 => requires non-ambiguous qualifying comorbidity
      - Ambiguous terms => not eligible (audit/debug classification)
    """
    if df_conds.empty or "condition_name" not in df_conds.columns:
        conds = []
    else:
        conds = (
            df_conds[df_conds["patient_id"] == pid]["condition_name"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if df_meds.empty or "medication_name" not in df_meds.columns:
        meds = []
    else:
        meds = (
            df_meds[df_meds["patient_id"] == pid]["medication_name"]
            .dropna()
            .astype(str)
            .tolist()
        )

    safety, safety_ev = _detect_safety_exclusion(conds, meds)
    if safety:
        return {"eligible": False, "reason": "SAFETY_EXCLUSION", "evidence": safety_ev}

    bmi = _calculate_bmi_ground_truth(df_obs, pid)
    if bmi is None:
        return {"eligible": None, "reason": "MISSING_BMI", "evidence": ""}

    # BMI >= obese threshold: align with current policy_engine.py requirement
    # BMI >= obese threshold: align with current policy_engine.py requirement
    # Engine Logic: BMI >= 30 is sufficient for clinical eligibility (verdict=APPROVED or CDI_REQUIRED).
    # We mark them as eligible=True regardless of the text string presence, because
    # the lack of a string is an administrative (CDI) issue, not a clinical eligibility failure.
    if bmi >= POLICY["bmi_obesity_threshold"]:
        return {"eligible": True, "reason": "BMI30_OBESITY", "evidence": f"BMI {bmi}"}

    if bmi < POLICY["bmi_overweight_threshold"]:
        return {"eligible": False, "reason": "BMI_BELOW_THRESHOLD", "evidence": f"BMI {bmi}"}

    # BMI 27–29.9 requires qualifying comorbidity (not ambiguous)
    cvd_ev = _is_cvd(conds)
    if cvd_ev:
        return {"eligible": True, "reason": "CVD", "evidence": cvd_ev}

    htn_ev = _is_hypertension(conds)
    if htn_ev:
        return {"eligible": True, "reason": "HYPERTENSION", "evidence": htn_ev}

    lip_ev = _is_lipids(conds)
    if lip_ev:
        return {"eligible": True, "reason": "LIPIDS", "evidence": lip_ev}

    t2_ev = _is_t2_diabetes(conds)
    if t2_ev:
        return {"eligible": True, "reason": "DIABETES_T2", "evidence": t2_ev}

    osa_ev = _has_osa_specific(conds)
    if osa_ev:
        return {"eligible": True, "reason": "OSA", "evidence": osa_ev}

    # Non-qualifying but useful for audit/debug alignment
    amb_sleep = _is_generic_sleep_apnea_only(conds)
    if amb_sleep:
        return {"eligible": False, "reason": "AMBIGUOUS_SLEEP_APNEA", "evidence": amb_sleep}

    amb_thy = _any_match_in_conditions(conds, AMBIGUOUS_THYROID)
    if amb_thy:
        return {"eligible": False, "reason": "AMBIGUOUS_THYROID", "evidence": amb_thy}

    # Other ambiguous terms (still not eligible)
    for amb_list in (AMBIGUOUS_BP, AMBIGUOUS_DIABETES, AMBIGUOUS_OBESITY, AMBIGUOUS_SLEEP_APNEA):
        amb_hit = _any_match_in_conditions(conds, amb_list)
        if amb_hit:
            return {"eligible": False, "reason": "AMBIGUOUS_TERM", "evidence": amb_hit}

    return {"eligible": False, "reason": "NO_QUALIFYING_COMORBIDITY", "evidence": ""}


# -----------------------------
# Stats helpers (robust FNR parity)
# -----------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (round(lo, 4), round(hi, 4))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z_pvalue(k1: int, n1: int, k2: int, n2: int) -> float | None:
    if n1 <= 0 or n2 <= 0:
        return None
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = ((k1 / n1) - (k2 / n2)) / se
    return float(2 * (1 - norm_cdf(abs(z))))


# -----------------------------
# Governance audit (Equality of Opportunity via FNR parity)
# -----------------------------
def run_governance_audit(
    ai_results_path: str = str(INPUT_FILE),
    out_path: str = str(REPORT_FILE),
    min_eligible_sample_size: int = 30,
    disparity_threshold: float = 0.10,
    alpha: float = 0.05,
) -> None:
    logger.info("Running clinical fairness audit (equal opportunity / FNR parity) on %s", ai_results_path)

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load AI results
    try:
        if not Path(ai_results_path).exists():
            logger.error("Input file not found: %s", ai_results_path)
            return

        with open(ai_results_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "results" in data:
            ai_results = data["results"]
        elif isinstance(data, list):
            ai_results = data
        else:
            logger.error("Invalid JSON structure in %s", ai_results_path)
            return

        df_ai = pd.DataFrame(ai_results)

        # Load raw data from OUTPUT_DIR (not repo root)
        required = [PATIENTS_CSV, OBS_CSV, CONDS_CSV, MEDS_CSV]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            logger.error("Missing required data files: %s", missing)
            return

        df_pat = pd.read_csv(PATIENTS_CSV)
        df_obs = pd.read_csv(OBS_CSV)
        df_conds = pd.read_csv(CONDS_CSV)
        df_meds = pd.read_csv(MEDS_CSV)

    except Exception as e:
        logger.error("Data Load Error: %s", e)
        return

    if df_ai.empty:
        logger.warning("No AI results to audit.")
        return

    if "patient_id" not in df_ai.columns:
        raise ValueError("AI results must include 'patient_id' per record.")

    status_col = "status" if "status" in df_ai.columns else ("final_decision" if "final_decision" in df_ai.columns else None)
    if status_col is None:
        raise ValueError("AI results must include 'status' or 'final_decision' per record.")

    # Normalize IDs to string across all frames
    for d in (df_ai, df_pat, df_obs, df_conds, df_meds):
        if not d.empty and "patient_id" in d.columns:
            d["patient_id"] = d["patient_id"].astype(str)

    # Ground truth (tri-state)
    logger.info("Calculating Ground Truth (policy-aligned)...")
    truth_map: dict[str, bool | None] = {}
    truth_meta: dict[str, dict[str, str]] = {}

    for pid in df_ai["patient_id"].dropna().unique():
        pid = str(pid)
        t = calculate_ground_truth_eligibility(pid, df_obs, df_conds, df_meds)
        truth_map[pid] = t["eligible"]  # True / False / None
        truth_meta[pid] = {"reason": str(t.get("reason", "")), "evidence": str(t.get("evidence", ""))}

    df_ai["ground_truth_eligible"] = df_ai["patient_id"].map(truth_map)
    df_ai["ground_truth_reason"] = df_ai["patient_id"].map(lambda x: truth_meta.get(str(x), {}).get("reason", ""))
    df_ai["ground_truth_evidence"] = df_ai["patient_id"].map(lambda x: truth_meta.get(str(x), {}).get("evidence", ""))

    # Merge demographics
    df_merged = df_ai.merge(df_pat, on="patient_id", how="left")

    # Normalize decision labels
    df_merged["decision_norm"] = df_merged[status_col].astype(str).str.upper().str.strip()
    df_merged["is_approved"] = df_merged["decision_norm"] == "APPROVED"
    df_merged["is_denied"] = df_merged["decision_norm"].str.startswith("DENIED")

    # False negative definitions
    df_merged["fn_access"] = (df_merged["ground_truth_eligible"] == True) & (~df_merged["is_approved"])
    df_merged["fn_denied_only"] = (df_merged["ground_truth_eligible"] == True) & (df_merged["is_denied"])

    # Exclude unknown ground truth from parity denominators
    df_eval = df_merged[df_merged["ground_truth_eligible"].isin([True, False])].copy()

    def compute_group_metrics(df_group: pd.DataFrame) -> dict:
        total_n = int(len(df_group))
        eligible_mask = df_group["ground_truth_eligible"] == True
        eligible_n = int(eligible_mask.sum())
        unknown_truth_n = int(df_group["ground_truth_eligible"].isna().sum())

        if eligible_n <= 0:
            return {
                "total_n": total_n,
                "eligible_n": 0,
                "unknown_truth_n": unknown_truth_n,
                "fn_access_n": 0,
                "fn_denied_only_n": 0,
                "fnr_access": None,
                "fnr_access_ci95": None,
                "fnr_denied_only": None,
                "fnr_denied_only_ci95": None,
                "insufficient_data": True,
            }

        fn_access_n = int(df_group.loc[eligible_mask, "fn_access"].sum())
        fn_denied_n = int(df_group.loc[eligible_mask, "fn_denied_only"].sum())

        fnr_access = fn_access_n / eligible_n
        fnr_denied = fn_denied_n / eligible_n

        return {
            "total_n": total_n,
            "eligible_n": eligible_n,
            "unknown_truth_n": unknown_truth_n,
            "fn_access_n": fn_access_n,
            "fn_denied_only_n": fn_denied_n,
            "fnr_access": round(fnr_access, 4),
            "fnr_access_ci95": wilson_ci(fn_access_n, eligible_n),
            "fnr_denied_only": round(fnr_denied, 4),
            "fnr_denied_only_ci95": wilson_ci(fn_denied_n, eligible_n),
            "insufficient_data": eligible_n < min_eligible_sample_size,
        }

    def audit_attribute(attr: str) -> dict:
        if attr not in df_merged.columns:
            return {
                "attribute": attr,
                "error": f"Missing column '{attr}' in {PATIENTS_CSV.name}; cannot audit this attribute.",
                "group_metrics": {},
                "bias_detected": False,
                "stop_ship_groups": [],
            }

        metrics: dict[str, dict] = {}
        groups = [g for g in df_eval[attr].dropna().unique()]

        for g in groups:
            df_g_all = df_merged[df_merged[attr] == g]
            metrics[str(g)] = compute_group_metrics(df_g_all)

        valid_groups = [
            k for k, v in metrics.items()
            if (not v.get("insufficient_data")) and (v.get("fnr_access") is not None)
        ]
        m_tests = max(1, len(valid_groups))
        alpha_bonf = alpha / m_tests

        stop_ship_groups: list[str] = []

        for g in valid_groups:
            df_g = df_eval[df_eval[attr] == g]
            df_rest = df_eval[df_eval[attr] != g]

            g_eligible_mask = df_g["ground_truth_eligible"] == True
            r_eligible_mask = df_rest["ground_truth_eligible"] == True

            n1 = int(g_eligible_mask.sum())
            n2 = int(r_eligible_mask.sum())
            k1 = int(df_g.loc[g_eligible_mask, "fn_access"].sum())
            k2 = int(df_rest.loc[r_eligible_mask, "fn_access"].sum())

            pval = two_proportion_z_pvalue(k1, n1, k2, n2)
            rest_rate = (k2 / n2) if n2 > 0 else None
            diff = (metrics[g]["fnr_access"] - rest_rate) if rest_rate is not None else None

            metrics[g]["p_value_vs_rest_access"] = None if pval is None else round(pval, 6)
            metrics[g]["rest_fnr_access"] = None if rest_rate is None else round(rest_rate, 4)
            metrics[g]["diff_vs_rest_access"] = None if diff is None else round(diff, 4)

            significant = (
                (pval is not None)
                and (pval < alpha_bonf)
                and (diff is not None)
                and (abs(diff) >= disparity_threshold)
            )
            metrics[g]["significant_vs_rest_access"] = significant

            if significant:
                stop_ship_groups.append(g)

        valid_rates = {
            k: v["fnr_access"]
            for k, v in metrics.items()
            if isinstance(v.get("fnr_access"), float) and not v.get("insufficient_data")
        }
        if valid_rates:
            worst_group = max(valid_rates, key=lambda k: valid_rates[k])
            best_group = min(valid_rates, key=lambda k: valid_rates[k])
            max_diff = float(valid_rates[worst_group] - valid_rates[best_group])
        else:
            worst_group = None
            best_group = None
            max_diff = None

        bias_detected = len(stop_ship_groups) > 0
        warning = (
            f"Warning: significant disparity detected. Stop-ship groups: {stop_ship_groups}. "
            f"(threshold={disparity_threshold:.2f}, alpha_bonf={alpha_bonf:.4f})"
            if bias_detected
            else "No statistically-supported parity breach detected under current thresholds."
        )

        return {
            "attribute": attr,
            "definition": {
                "ground_truth": "Deterministic Wegovy policy logic (BMI + qualifying comorbidity + safety exclusions).",
                "false_negative_access": "Among truly-eligible patients: system outcome != APPROVED (includes DENIED/FLAGGED/etc.).",
                "false_negative_denied_only": "Among truly-eligible patients: system outcome startswith DENIED.",
                "min_eligible_sample_size": min_eligible_sample_size,
                "disparity_threshold": disparity_threshold,
                "alpha": alpha,
                "bonferroni_alpha": round(alpha_bonf, 6),
            },
            "group_metrics": metrics,
            "summary": {
                "max_diff_fnr_access": None if max_diff is None else round(max_diff, 4),
                "worst_group": worst_group,
                "best_group": best_group,
                "stop_ship_groups": stop_ship_groups,
            },
            "bias_detected": bias_detected,
            "bias_warning": warning,
        }

    logger.info("Analyzing Disparities (Minimum Eligible N=%d)...", min_eligible_sample_size)

    audits = [audit_attribute(attr) for attr in ["race", "gender"]]

    eligible_known = df_eval[df_eval["ground_truth_eligible"] == True]
    overall_eligible_n = int(len(eligible_known))
    overall_fn_access_n = int(eligible_known["fn_access"].sum()) if overall_eligible_n > 0 else 0
    overall_fnr_access = (overall_fn_access_n / overall_eligible_n) if overall_eligible_n > 0 else None

    report = {
        "metric_name": f"False Negative Rate Parity (Equality of Opportunity) | Min Eligible N={min_eligible_sample_size}",
        "inputs": {
            "ai_results_path": str(ai_results_path),
            "patients_csv": str(PATIENTS_CSV),
            "observations_csv": str(OBS_CSV),
            "conditions_csv": str(CONDS_CSV),
            "medications_csv": str(MEDS_CSV),
        },
        "overall": {
            "eligible_n": overall_eligible_n,
            "fn_access_n": overall_fn_access_n,
            "fnr_access": None if overall_fnr_access is None else round(overall_fnr_access, 4),
            "fnr_access_ci95": None if overall_eligible_n <= 0 else wilson_ci(overall_fn_access_n, overall_eligible_n),
            "notes": "Ground truth unknown cases (e.g., missing BMI) are excluded from eligible denominators.",
        },
        "attribute_audits": audits,
    }

    # Optional schema validation (do not block report writing if schema evolves)
    try:
        validate_governance_report(report)  # if this expects a dict, great; if not, it will be caught
    except Exception as e:
        logger.warning("Governance report schema validation skipped/failed: %s", e)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Governance report saved to %s", out_path)
    logger.debug(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_governance_audit()

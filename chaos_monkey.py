"""
chaos_monkey.py - Adversarial Scenario Injector for Prior Authorization Testing

This module:
1) Runs ETL (Synthea FHIR -> canonical CSV-shaped DataFrames)
2) Selects a cohort of patients for Wegovy prior-auth "claims"
3) Injects controlled, adversarial clinical scenarios:
   - True BMI removal (forces BMI calculation from height/weight, or missing-info flags)
   - Clear approvals (BMI >= 30 OR BMI 27-29.9 + qualifying comorbidity)
   - Clinical denials (BMI < 27 OR BMI 27-29.9 with no qualifying comorbidity)
   - Safety denials (MTC/MEN2, pregnancy/nursing, concurrent GLP-1)
   - Manual-review triggers (ambiguous terms that must NOT auto-qualify)
4) Writes CSVs to ./output and writes a per-claim scenario manifest for ground truth.

Key fixes vs earlier version:
- Implements *actual* BMI deletion for relevant scenarios (not just appending).
- Adds scenarios for missing height OR missing weight (tests DENIED_MISSING_INFO).
- Adds claim_count (takes precedence over claim_rate) so experiments match your story.
- Forces injected records to be "latest" by date (based on per-patient max date).
- Sanitizes confounders so expected outcomes are stable and interpretable.
- Writes output/scenario_manifest.json (+ optional CSV) with scenario + expected verdict.

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CLAIM_RATE
from etl_pipeline import run_etl

# Reuse policy constants to avoid "magic strings" diverging across modules.
# These imports should exist in your repo (agent_logic.py already uses them).
from policy_constants import (
    PROHIBITED_GLP1,
    SAFETY_GI_MOTILITY,
    SAFETY_HYPERSENSITIVITY,
    SAFETY_MTC_MEN2,
    SAFETY_PANCREATITIS,
    SAFETY_PREGNANCY_LACTATION,
    SAFETY_SUICIDALITY,
)

logger = logging.getLogger(__name__)


# -----------------------------
# Scenario / Ground Truth Model
# -----------------------------
@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    weight: float
    expected_verdict: str  # policy_engine-style (APPROVED / DENIED_SAFETY / DENIED_CLINICAL / DENIED_MISSING_INFO / MANUAL_REVIEW)
    description: str


DEFAULT_SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(
        name="SAFETY_MTC_MEN2",
        weight=0.10,
        expected_verdict="DENIED_SAFETY",
        description="Inject MTC/MEN2 term; should hard-stop safety denial.",
    ),
    ScenarioSpec(
        name="SAFETY_DUPLICATE_GLP1",
        weight=0.10,
        expected_verdict="DENIED_SAFETY",
        description="Inject concurrent GLP-1 med (e.g., Ozempic); should hard-stop safety denial.",
    ),
    ScenarioSpec(
        name="SAFETY_PREGNANCY_NURSING",
        weight=0.06,
        expected_verdict="DENIED_SAFETY",
        description="Inject pregnancy or nursing status (female-only); should hard-stop safety denial.",
    ),
    ScenarioSpec(
        name="DATA_GAP_HEIGHT_WEIGHT_ONLY",
        weight=0.16,
        expected_verdict="APPROVED",
        description="Delete BMI; provide height+weight so BMI is calculated (obese range).",
    ),
    ScenarioSpec(
        name="DATA_GAP_MISSING_HEIGHT",
        weight=0.05,
        expected_verdict="DENIED_MISSING_INFO",
        description="Delete BMI; provide weight only -> BMI cannot be calculated.",
    ),
    ScenarioSpec(
        name="DATA_GAP_MISSING_WEIGHT",
        weight=0.05,
        expected_verdict="DENIED_MISSING_INFO",
        description="Delete BMI; provide height only -> BMI cannot be calculated.",
    ),
    ScenarioSpec(
        name="OBESE_BMI30_PLUS",
        weight=0.12,
        expected_verdict="APPROVED",
        description="BMI >= 30 with no comorbidity requirement; should approve if safety clear.",
    ),
    ScenarioSpec(
        name="OVERWEIGHT_VALID_COMORBIDITY",
        weight=0.20,
        expected_verdict="APPROVED",
        description="BMI 27-29.9 + qualifying comorbidity; should approve if safety clear.",
    ),
    ScenarioSpec(
        name="OVERWEIGHT_AMBIGUOUS_TERM",
        weight=0.12,
        expected_verdict="MANUAL_REVIEW",
        description="BMI 27-29.9 + ambiguous term (prediabetes, generic sleep apnea, etc.); should flag manual review.",
    ),
    ScenarioSpec(
        name="OVERWEIGHT_NO_COMORBIDITY",
        weight=0.08,
        expected_verdict="DENIED_CLINICAL",
        description="BMI 27-29.9 with no qualifying comorbidity; should deny clinical.",
    ),
    ScenarioSpec(
        name="BMI_UNDER_27",
        weight=0.04,
        expected_verdict="DENIED_CLINICAL",
        description="BMI < 27; should deny clinical.",
    ),
]


# -----------------------------
# Text match / sanitization lists
# -----------------------------
QUALIFYING_CONDITION_SUBSTRINGS: List[str] = [
    # HTN
    "hypertension",
    "essential hypertension",
    "htn",
    # Lipids
    "hyperlipidemia",
    "dyslipidemia",
    "high cholesterol",
    "cholesterol",
    # Diabetes (qualifying)
    "type 2 diabetes",
    "type ii diabetes",
    "t2dm",
    "diabetes mellitus type 2",
    # OSA
    "obstructive sleep apnea",
    "osa",
    # CVD / ASCVD
    "coronary artery disease",
    "cad",
    "myocardial infarction",
    "ischemic stroke",
    "stroke",
    "peripheral arterial disease",
    "ascvd",
    "cardiovascular disease",
    "heart attack",
]

AMBIGUOUS_TERMS: List[str] = [
    "Prediabetes",
    "Impaired fasting glucose",
    "Elevated BP",
    "Borderline hypertension",
    "Obesity",
    "Body mass index 30+",
    "Sleep apnea",  # intentionally generic (not "obstructive")
    "Malignant tumor of thyroid",
    "Thyroid malignancy",
]

QUALIFYING_CONDITION_CANONICAL: List[str] = [
    "Essential hypertension (disorder)",
    "Hypertension",
    "Hyperlipidemia",
    "Dyslipidemia",
    "Type 2 diabetes mellitus",
    "T2DM",
    "Obstructive Sleep Apnea",
    "OSA",
    "Coronary Artery Disease",
    "Myocardial Infarction",
    "Ischemic Stroke",
    "Peripheral Arterial Disease",
]

SAFETY_CONDITION_CANONICAL: List[str] = [
    "Medullary Thyroid Carcinoma (MTC)",
    "MTC",
    "Multiple Endocrine Neoplasia type 2 (MEN2)",
    "MEN2",
    "Pregnant",
    "Currently pregnant",
    "Breastfeeding",
    "Nursing",
]


# -----------------------------
# Helpers
# -----------------------------
def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _norm_str(x: Any) -> str:
    return str(x or "").strip()


def _contains_any(haystack: str, needles: Sequence[str]) -> bool:
    h = (haystack or "").lower()
    return any(n.lower() in h for n in needles)


def _safe_to_datetime_series(s: pd.Series) -> pd.Series:
    # Parse YYYY-MM-DD best-effort; invalid -> NaT
    return pd.to_datetime(s, errors="coerce", utc=False)


def _build_patient_attr_maps(df_p: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return pid->gender map and pid->race map (best-effort)."""
    gender_map: Dict[str, str] = {}
    race_map: Dict[str, str] = {}
    if df_p is None or df_p.empty:
        return gender_map, race_map

    cols = set(df_p.columns)
    for _, row in df_p.iterrows():
        pid = _norm_str(row.get("patient_id"))
        if not pid:
            continue
        if "gender" in cols:
            gender_map[pid] = _norm_str(row.get("gender")).lower()
        if "race" in cols:
            race_map[pid] = _norm_str(row.get("race"))
    return gender_map, race_map


def _max_date_by_patient(df: pd.DataFrame, pid_col: str, date_col: str) -> Dict[str, datetime]:
    """
    Compute per-patient max date from a DataFrame column containing YYYY-MM-DD.
    Returns a dict pid -> datetime (naive, local) for max valid date.
    """
    out: Dict[str, datetime] = {}
    if df is None or df.empty or pid_col not in df.columns or date_col not in df.columns:
        return out

    tmp = df[[pid_col, date_col]].copy()
    tmp[pid_col] = tmp[pid_col].astype(str)
    tmp["_dt"] = _safe_to_datetime_series(tmp[date_col].astype(str))
    tmp = tmp.dropna(subset=["_dt"])
    if tmp.empty:
        return out

    g = tmp.groupby(pid_col)["_dt"].max()
    for pid, dtv in g.items():
        try:
            out[str(pid)] = dtv.to_pydatetime().replace(tzinfo=None)
        except Exception:
            continue
    return out


def _next_injection_date(
    pid: str,
    base_map: Dict[str, datetime],
    *,
    fallback: Optional[datetime] = None,
    days_ahead: int = 1,
) -> str:
    """
    Make sure injected events are "latest" for a patient by choosing max_date+days_ahead.
    """
    base = base_map.get(str(pid))
    if base is None:
        base = fallback or datetime(2025, 1, 1)
    inj = base + timedelta(days=days_ahead)
    return inj.strftime("%Y-%m-%d")


def _drop_observations(df_o: pd.DataFrame, pid: str, types: Sequence[str]) -> pd.DataFrame:
    if df_o is None or df_o.empty:
        return df_o
    if "patient_id" not in df_o.columns or "type" not in df_o.columns:
        return df_o
    mask = ~((df_o["patient_id"].astype(str) == str(pid)) & (df_o["type"].astype(str).isin(list(types))))
    return df_o.loc[mask].reset_index(drop=True)


def _drop_conditions_matching(df_c: pd.DataFrame, pid: str, substrings: Sequence[str]) -> pd.DataFrame:
    if df_c is None or df_c.empty:
        return df_c
    if "patient_id" not in df_c.columns or "condition_name" not in df_c.columns:
        return df_c
    pid_mask = df_c["patient_id"].astype(str) == str(pid)
    if not pid_mask.any():
        return df_c
    conds = df_c.loc[pid_mask, "condition_name"].astype(str).str.lower()
    hit = conds.apply(lambda s: any(sub.lower() in s for sub in substrings))
    drop_mask = pid_mask.copy()
    drop_mask.loc[pid_mask] = hit.values
    return df_c.loc[~drop_mask].reset_index(drop=True)


def _drop_meds_matching(df_m: pd.DataFrame, pid: str, substrings: Sequence[str]) -> pd.DataFrame:
    if df_m is None or df_m.empty:
        return df_m
    if "patient_id" not in df_m.columns or "medication_name" not in df_m.columns:
        return df_m
    pid_mask = df_m["patient_id"].astype(str) == str(pid)
    if not pid_mask.any():
        return df_m
    meds = df_m.loc[pid_mask, "medication_name"].astype(str).str.lower()
    hit = meds.apply(lambda s: any(sub.lower() in s for sub in substrings))
    drop_mask = pid_mask.copy()
    drop_mask.loc[pid_mask] = hit.values
    return df_m.loc[~drop_mask].reset_index(drop=True)


def _safety_substrings_from_constants() -> List[str]:
    """
    Build a consolidated list of safety substrings used elsewhere in the agent.
    This helps sanitize non-safety scenarios so expected outcomes hold.
    """
    # These lists are typically substrings already; we normalize them as lower-case checks.
    buckets = [
        SAFETY_MTC_MEN2,
        SAFETY_PREGNANCY_LACTATION,
        SAFETY_HYPERSENSITIVITY,
        SAFETY_PANCREATITIS,
        SAFETY_SUICIDALITY,
        SAFETY_GI_MOTILITY,
    ]
    out: List[str] = []
    for b in buckets:
        for term in (b or []):
            t = _norm_str(term)
            if t:
                out.append(t)
    return out


SAFETY_SUBSTRINGS: List[str] = _safety_substrings_from_constants()


# -----------------------------
# Core Injection API
# -----------------------------
def inject_complex_scenarios(
    df_p: pd.DataFrame,
    df_c: pd.DataFrame,
    df_m: pd.DataFrame,
    df_o: pd.DataFrame,
    *,
    claim_rate: float = DEFAULT_CLAIM_RATE,
    claim_count: Optional[int] = None,
    seed: int = 42,
    sanitize_for_expected_outcome: bool = True,
    force_latest_dates: bool = True,
    scenarios: Optional[List[ScenarioSpec]] = None,
    return_manifest: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Inject synthetic Wegovy prior-auth scenarios aligned to your agent/policy engine.

    Args:
        df_p/df_c/df_m/df_o: canonical ETL outputs.
        claim_rate: fraction of patient cohort to receive Wegovy (ignored if claim_count set).
        claim_count: absolute number of Wegovy claims to generate (takes precedence).
        seed: RNG seed (deterministic).
        sanitize_for_expected_outcome: removes confounders (safety terms, prohibited meds,
            qualifying comorbidities) in scenarios where they would break ground truth.
        force_latest_dates: injects dates newer than existing per-patient max dates.
        scenarios: override default scenario mix.
        return_manifest: if True, returns an additional DataFrame manifest.

    Returns:
        Either (df_p, df_c, df_m, df_o) or (df_p, df_c, df_m, df_o, df_manifest).
    """
    df_p = df_p.copy()
    df_c = df_c.copy()
    df_m = df_m.copy()
    df_o = df_o.copy()

    if df_p is None or df_p.empty:
        logger.warning("No patients provided to chaos monkey.")
        if return_manifest:
            return df_p, df_c, df_m, df_o, pd.DataFrame()
        return df_p, df_c, df_m, df_o

    rng = np.random.default_rng(seed)
    scenario_specs = scenarios or DEFAULT_SCENARIOS

    # Normalize and validate weights
    weights = np.array([max(0.0, float(s.weight)) for s in scenario_specs], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("Scenario weights must sum to > 0.")
    weights = weights / weights.sum()

    # Patient universe
    df_p["patient_id"] = df_p["patient_id"].astype(str)
    all_ids = df_p["patient_id"].dropna().astype(str).unique().tolist()
    n_pat = len(all_ids)
    logger.info("Injecting scenarios into cohort of %d patients", n_pat)

    if claim_count is not None:
        sample_size = int(claim_count)
    else:
        sample_size = int(round(n_pat * float(claim_rate)))

    sample_size = max(1, min(n_pat, sample_size))
    cohort_ids = rng.choice(np.array(all_ids, dtype=object), size=sample_size, replace=False).tolist()
    cohort_ids = [str(x) for x in cohort_ids]

    logger.info("Generating %d Wegovy claims (seed=%s)", len(cohort_ids), seed)

    # Maps for faster attribute lookup
    gender_map, race_map = _build_patient_attr_maps(df_p)

    # Date maps to force "latest"
    obs_max = _max_date_by_patient(df_o, "patient_id", "date") if force_latest_dates else {}
    med_max = _max_date_by_patient(df_m, "patient_id", "date") if force_latest_dates else {}
    cond_max = _max_date_by_patient(df_c, "patient_id", "onset_date") if force_latest_dates else {}

    # For quick "already has wegovy" checks
    has_wegovy = set()
    if df_m is not None and not df_m.empty and "medication_name" in df_m.columns and "patient_id" in df_m.columns:
        m = df_m.copy()
        m["patient_id"] = m["patient_id"].astype(str)
        mname = m["medication_name"].astype(str)
        hit = mname.str.contains("wegovy", case=False, na=False)
        has_wegovy = set(m.loc[hit, "patient_id"].astype(str).tolist())

    # Collect new rows
    new_meds: List[Dict[str, Any]] = []
    new_obs: List[Dict[str, Any]] = []
    new_conds: List[Dict[str, Any]] = []

    manifest_rows: List[Dict[str, Any]] = []

    # Med injections
    def add_wegovy(pid: str, date_str: str) -> None:
        # If it already exists, we still add a row (it’s synthetic) unless you want to avoid duplicates.
        # Keeping it explicit helps in audit logs.
        new_meds.append(
            {
                "patient_id": pid,
                "medication_name": "Wegovy 2.4 MG Injection",
                "rx_code": "",
                "date": date_str,
                "status": "active",
            }
        )

    def add_concurrent_glp1(pid: str, date_str: str) -> None:
        new_meds.append(
            {
                "patient_id": pid,
                "medication_name": "Ozempic (semaglutide) injection",
                "rx_code": "",
                "date": date_str,
                "status": "active",
            }
        )

    # Observation injections
    def add_bmi(pid: str, bmi: float, date_str: str) -> None:
        new_obs.append(
            {
                "patient_id": pid,
                "type": "BMI",
                "value": round(float(bmi), 1),
                "unit": "kg/m2",
                "date": date_str,
            }
        )

    def add_height_weight(pid: str, height_cm: float, weight_kg: float, date_str: str) -> None:
        new_obs.append(
            {"patient_id": pid, "type": "Height", "value": round(float(height_cm), 1), "unit": "cm", "date": date_str}
        )
        new_obs.append(
            {"patient_id": pid, "type": "Weight", "value": round(float(weight_kg), 1), "unit": "kg", "date": date_str}
        )

    def add_height_only(pid: str, height_cm: float, date_str: str) -> None:
        new_obs.append(
            {"patient_id": pid, "type": "Height", "value": round(float(height_cm), 1), "unit": "cm", "date": date_str}
        )

    def add_weight_only(pid: str, weight_kg: float, date_str: str) -> None:
        new_obs.append(
            {"patient_id": pid, "type": "Weight", "value": round(float(weight_kg), 1), "unit": "kg", "date": date_str}
        )

    # Condition injections
    def add_condition(pid: str, name: str, onset_str: str) -> None:
        new_conds.append(
            {
                "patient_id": pid,
                "condition_name": name,
                "code": "",
                "onset_date": onset_str,
            }
        )

    # Scenario draw helper
    scenario_names = [s.name for s in scenario_specs]

    def draw_scenario_for_pid(pid: str) -> ScenarioSpec:
        # Pregnancy scenario should only apply to females; resample otherwise.
        # We do NOT silently mutate into a different scenario under the same label.
        gender = gender_map.get(pid, "")
        for _ in range(20):
            idx = int(rng.choice(len(scenario_specs), p=weights))
            spec = scenario_specs[idx]
            if spec.name == "SAFETY_PREGNANCY_NURSING" and "female" not in (gender or ""):
                continue
            return spec
        # Fallback: pick a non-pregnancy scenario
        non_preg = [s for s in scenario_specs if s.name != "SAFETY_PREGNANCY_NURSING"]
        idx = int(rng.integers(0, len(non_preg)))
        return non_preg[idx]

    # Main injection loop
    for pid in cohort_ids:
        pid = str(pid)

        # Choose dates (force "latest" per patient)
        obs_date = _next_injection_date(pid, obs_max, fallback=datetime(2025, 1, 1), days_ahead=1) if force_latest_dates else "2025-01-02"
        med_date = _next_injection_date(pid, med_max, fallback=datetime(2025, 1, 1), days_ahead=2) if force_latest_dates else "2025-01-03"
        onset_date = _next_injection_date(pid, cond_max, fallback=datetime(2024, 1, 1), days_ahead=1) if force_latest_dates else "2024-01-02"

        spec = draw_scenario_for_pid(pid)

        # Optional sanitization to stabilize expected outcomes
        if sanitize_for_expected_outcome:
            # For non-safety scenarios, remove safety-confounding conditions and prohibited meds
            if not spec.name.startswith("SAFETY_"):
                df_c = _drop_conditions_matching(df_c, pid, SAFETY_SUBSTRINGS)
                df_m = _drop_meds_matching(df_m, pid, PROHIBITED_GLP1)

            # For scenarios where comorbidity must be absent (or ambiguous only), remove qualifying conditions
            if spec.name in ("OVERWEIGHT_AMBIGUOUS_TERM", "OVERWEIGHT_NO_COMORBIDITY", "BMI_UNDER_27", "OBESE_BMI30_PLUS"):
                df_c = _drop_conditions_matching(df_c, pid, QUALIFYING_CONDITION_SUBSTRINGS)

        # Always add Wegovy claim (even if already present; this is a test harness)
        add_wegovy(pid, med_date)

        injected: Dict[str, Any] = {
            "injected_bmi": None,
            "injected_height_cm": None,
            "injected_weight_kg": None,
            "injected_condition": None,
            "injected_med": None,
            "bmi_deleted": False,
            "height_deleted": False,
            "weight_deleted": False,
        }

        # Scenario behaviors
        if spec.name == "SAFETY_MTC_MEN2":
            bmi = float(rng.uniform(32.0, 40.0))
            add_bmi(pid, bmi, obs_date)
            cond = str(rng.choice(np.array(["Medullary Thyroid Carcinoma (MTC)", "MTC", "Multiple Endocrine Neoplasia type 2 (MEN2)", "MEN2"], dtype=object)))
            add_condition(pid, cond, onset_date)
            injected["injected_bmi"] = round(bmi, 1)
            injected["injected_condition"] = cond

        elif spec.name == "SAFETY_DUPLICATE_GLP1":
            bmi = float(rng.uniform(30.0, 38.0))
            add_bmi(pid, bmi, obs_date)
            add_concurrent_glp1(pid, med_date)
            injected["injected_bmi"] = round(bmi, 1)
            injected["injected_med"] = "Ozempic (semaglutide) injection"

        elif spec.name == "SAFETY_PREGNANCY_NURSING":
            bmi = float(rng.uniform(27.0, 36.0))
            add_bmi(pid, bmi, obs_date)
            cond = str(rng.choice(np.array(["Pregnant", "Currently pregnant", "Breastfeeding", "Nursing"], dtype=object)))
            add_condition(pid, cond, onset_date)
            injected["injected_bmi"] = round(bmi, 1)
            injected["injected_condition"] = cond

        elif spec.name == "DATA_GAP_HEIGHT_WEIGHT_ONLY":
            # Delete BMI so engine must compute from height/weight (and make values obese-range)
            df_o = _drop_observations(df_o, pid, ["BMI"])
            injected["bmi_deleted"] = True

            # Create obese BMI via height/weight: e.g., 175cm, 110kg => ~35.9
            height_cm = float(rng.uniform(165.0, 185.0))
            # pick weight to roughly ensure BMI >= 30
            target_bmi = float(rng.uniform(31.0, 38.0))
            height_m = height_cm / 100.0
            weight_kg = target_bmi * (height_m**2)

            add_height_weight(pid, height_cm, weight_kg, obs_date)
            injected["injected_height_cm"] = round(height_cm, 1)
            injected["injected_weight_kg"] = round(weight_kg, 1)

        elif spec.name == "DATA_GAP_MISSING_HEIGHT":
            # Delete BMI; provide weight only (remove existing height too if we want a true gap)
            df_o = _drop_observations(df_o, pid, ["BMI"])
            injected["bmi_deleted"] = True

            # Make sure height is missing: remove Height observations for this pid
            df_o = _drop_observations(df_o, pid, ["Height"])
            injected["height_deleted"] = True

            weight_kg = float(rng.uniform(75.0, 140.0))
            add_weight_only(pid, weight_kg, obs_date)
            injected["injected_weight_kg"] = round(weight_kg, 1)

        elif spec.name == "DATA_GAP_MISSING_WEIGHT":
            # Delete BMI; provide height only (remove existing weight too)
            df_o = _drop_observations(df_o, pid, ["BMI"])
            injected["bmi_deleted"] = True

            df_o = _drop_observations(df_o, pid, ["Weight"])
            injected["weight_deleted"] = True

            height_cm = float(rng.uniform(150.0, 195.0))
            add_height_only(pid, height_cm, obs_date)
            injected["injected_height_cm"] = round(height_cm, 1)

        elif spec.name == "OBESE_BMI30_PLUS":
            bmi = float(rng.uniform(30.0, 45.0))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

        elif spec.name == "OVERWEIGHT_VALID_COMORBIDITY":
            bmi = float(rng.uniform(27.0, 29.9))
            add_bmi(pid, bmi, obs_date)

            cond = str(rng.choice(np.array(QUALIFYING_CONDITION_CANONICAL, dtype=object)))
            add_condition(pid, cond, onset_date)

            injected["injected_bmi"] = round(bmi, 1)
            injected["injected_condition"] = cond

        elif spec.name == "OVERWEIGHT_AMBIGUOUS_TERM":
            bmi = float(rng.uniform(27.0, 29.9))
            add_bmi(pid, bmi, obs_date)

            cond = str(rng.choice(np.array(AMBIGUOUS_TERMS, dtype=object)))
            add_condition(pid, cond, onset_date)

            injected["injected_bmi"] = round(bmi, 1)
            injected["injected_condition"] = cond

        elif spec.name == "OVERWEIGHT_NO_COMORBIDITY":
            bmi = float(rng.uniform(27.0, 29.9))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

        elif spec.name == "BMI_UNDER_27":
            bmi = float(rng.uniform(22.0, 26.9))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

        else:
            # Defensive: unknown scenario should be visible in manifest
            bmi = float(rng.uniform(27.0, 29.9))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

        # Manifest row for ground truth
        manifest_rows.append(
            {
                "patient_id": pid,
                "scenario": spec.name,
                "expected_verdict": spec.expected_verdict,
                "scenario_description": spec.description,
                "seed": seed,
                "wegovy_injected": True,
                "obs_date": obs_date,
                "med_date": med_date,
                "onset_date": onset_date,
                "gender": gender_map.get(pid, ""),
                "race": race_map.get(pid, ""),
                **injected,
            }
        )

    # Append new rows
    if new_meds:
        df_m = pd.concat([df_m, pd.DataFrame(new_meds)], ignore_index=True)
    if new_obs:
        df_o = pd.concat([df_o, pd.DataFrame(new_obs)], ignore_index=True)
    if new_conds:
        df_c = pd.concat([df_c, pd.DataFrame(new_conds)], ignore_index=True)

    # Normalize patient_id types
    for df in (df_p, df_c, df_m, df_o):
        if df is not None and not df.empty and "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)

    manifest_df = pd.DataFrame(manifest_rows)

    # Basic logging summary
    try:
        dist = manifest_df["scenario"].value_counts().to_dict()
        logger.info("Scenario distribution: %s", dist)
        exp = manifest_df["expected_verdict"].value_counts().to_dict()
        logger.info("Expected verdict distribution: %s", exp)
    except Exception:
        pass

    if return_manifest:
        return df_p, df_c, df_m, df_o, manifest_df
    return df_p, df_c, df_m, df_o


# -----------------------------
# Script entrypoint (writes artifacts)
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_dir = Path("output")
    _ensure_output_dir(out_dir)

    # Allow overriding from env so your experiment narratives match reality.
    # Examples:
    #   PA_CLAIM_COUNT=85 python chaos_monkey.py
    #   PA_CLAIM_RATE=0.02125 python chaos_monkey.py
    env_claim_count = os.getenv("PA_CLAIM_COUNT")
    env_claim_rate = os.getenv("PA_CLAIM_RATE")
    env_seed = os.getenv("PA_SEED")

    claim_count: Optional[int] = int(env_claim_count) if env_claim_count else None
    claim_rate: float = float(env_claim_rate) if env_claim_rate else float(DEFAULT_CLAIM_RATE)
    seed: int = int(env_seed) if env_seed else 42

    df_p, df_c, df_m, df_o = run_etl()
    if df_p is None or df_p.empty:
        raise SystemExit("No patients found. Check your FHIR output directory and rerun.")

    res = inject_complex_scenarios(
        df_p,
        df_c,
        df_m,
        df_o,
        claim_rate=claim_rate,
        claim_count=claim_count,
        seed=seed,
        sanitize_for_expected_outcome=True,
        force_latest_dates=True,
        return_manifest=True,
    )
    df_p2, df_c2, df_m2, df_o2, manifest = res

    # Write canonical CSVs (what your batch_runner expects)
    df_p2.to_csv(out_dir / "data_patients.csv", index=False)
    df_c2.to_csv(out_dir / "data_conditions.csv", index=False)
    df_m2.to_csv(out_dir / "data_medications.csv", index=False)
    df_o2.to_csv(out_dir / "data_observations.csv", index=False)

    # Write ground truth manifest
    manifest_path = out_dir / "scenario_manifest.json"
    manifest_csv_path = out_dir / "scenario_manifest.csv"

    meta = {
        "timestamp_utc": _now_iso_utc(),
        "seed": seed,
        "claim_count": int(claim_count) if claim_count is not None else None,
        "claim_rate": float(claim_rate) if claim_count is None else None,
        "scenario_mix": [asdict(s) for s in DEFAULT_SCENARIOS],
        "notes": [
            "expected_verdict uses policy_engine-style outcomes.",
            "sanitize_for_expected_outcome=True removes confounders so scenarios are stable.",
            "force_latest_dates=True ensures injected items are newest per patient.",
        ],
    }

    payload = {"metadata": meta, "claims": manifest.to_dict(orient="records")}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Optional: CSV version for quick inspection / pivoting
    try:
        manifest.to_csv(manifest_csv_path, index=False)
    except Exception as e:
        logger.warning("Could not write manifest CSV: %s", e)

    logger.info("Data generation complete.")
    logger.info("Wrote: %s", (out_dir / "data_patients.csv"))
    logger.info("Wrote: %s", (out_dir / "data_conditions.csv"))
    logger.info("Wrote: %s", (out_dir / "data_medications.csv"))
    logger.info("Wrote: %s", (out_dir / "data_observations.csv"))
    logger.info("Wrote: %s", manifest_path)
    logger.info("Wrote: %s", manifest_csv_path)

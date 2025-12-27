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
- Writes output/scenario_manifest.json (+ optional CSV) enriched with ICD-10/Z68 codes.

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_CLAIM_RATE
from etl_pipeline import run_etl

# Reuse policy constants to avoid "magic strings" diverging across modules.
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
    expected_verdict: str  # policy_engine-style outcome
    description: str


DEFAULT_SCENARIOS: list[ScenarioSpec] = [
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
        weight=0.10,
        expected_verdict="CDI_REQUIRED",
        description="Delete BMI; provide height+weight so BMI is calculated (obese range). FAILS Admin check (no codes).",
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
        name="OBESE_NO_CODES",
        weight=0.10,
        expected_verdict="CDI_REQUIRED",
        description="BMI >= 30 and 'Obesity' text, but missing E66/Z68 codes. Should trigger CDI.",
    ),
    ScenarioSpec(
        name="OVERWEIGHT_VALID_COMORBIDITY",
        weight=0.15,
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
QUALIFYING_CONDITION_SUBSTRINGS: list[str] = [
    "hypertension", "essential hypertension", "htn",
    "hyperlipidemia", "dyslipidemia", "high cholesterol", "cholesterol",
    "type 2 diabetes", "type ii diabetes", "t2dm", "diabetes mellitus type 2",
    "obstructive sleep apnea", "osa",
    "coronary artery disease", "cad", "myocardial infarction",
    "ischemic stroke", "stroke", "peripheral arterial disease", "ascvd",
    "cardiovascular disease", "heart attack",
]

AMBIGUOUS_TERMS: list[str] = [
    "Prediabetes",
    "Impaired fasting glucose",
    "Elevated BP",
    "Borderline hypertension",
    "Sleep apnea",
    "Malignant tumor of thyroid",
    "Thyroid malignancy",
]

QUALIFYING_CONDITION_CANONICAL: list[str] = [
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


# -----------------------------
# Helpers
# -----------------------------
def _now_iso_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _norm_str(x: Any) -> str:
    return str(x or "").strip()


def _safe_to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _get_z68_code(bmi: float | None) -> str | None:
    """Calculate Z68 code for a given BMI value."""
    if bmi is None or pd.isna(bmi):
        return None

    if bmi < 20.0:
        return 'Z68.1'
    elif 20.0 <= bmi < 30.0:
        digit = int(bmi) - 20
        return f"Z68.2{digit}"
    elif 30.0 <= bmi < 40.0:
        digit = int(bmi) - 30
        return f"Z68.3{digit}"
    elif 40.0 <= bmi < 45.0:
        return 'Z68.41'
    elif 45.0 <= bmi < 50.0:
        return 'Z68.42'
    elif 50.0 <= bmi < 60.0:
        return 'Z68.43'
    elif 60.0 <= bmi < 70.0:
        return 'Z68.44'
    elif bmi >= 70.0:
        return 'Z68.45'
    return None


def _get_icd10_code(bmi: float | None) -> str | None:
    """Calculate ICD-10 diagnosis code for a given BMI value."""
    if bmi is None or pd.isna(bmi):
        return None

    if bmi >= 40.0:
        return 'E66.01'  # Morbid Obesity
    elif 30.0 <= bmi < 40.0:
        return 'E66.9'   # Obesity, Unspecified
    elif 27.0 <= bmi < 30.0:
        return 'E66.3'   # Overweight

    # BMI < 27 typically doesn't trigger an E66 code in this context
    return None


def _build_patient_attr_maps(df_p: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    gender_map: dict[str, str] = {}
    race_map: dict[str, str] = {}
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


def _max_date_by_patient(df: pd.DataFrame, pid_col: str, date_col: str) -> dict[str, datetime]:
    out: dict[str, datetime] = {}
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
    base_map: dict[str, datetime],
    *,
    fallback: datetime | None = None,
    days_ahead: int = 1,
) -> str:
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


def _safety_substrings_from_constants() -> list[str]:
    buckets = [
        SAFETY_MTC_MEN2,
        SAFETY_PREGNANCY_LACTATION,
        SAFETY_HYPERSENSITIVITY,
        SAFETY_PANCREATITIS,
        SAFETY_SUICIDALITY,
        SAFETY_GI_MOTILITY,
    ]
    out: list[str] = []
    for b in buckets:
        for term in (b or []):
            t = _norm_str(term)
            if t:
                out.append(t)
    return out


SAFETY_SUBSTRINGS: list[str] = _safety_substrings_from_constants()


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
    claim_count: int | None = None,
    seed: int = 42,
    sanitize_for_expected_outcome: bool = True,
    force_latest_dates: bool = True,
    scenarios: list[ScenarioSpec] | None = None,
    return_manifest: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
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

    weights = np.array([max(0.0, float(s.weight)) for s in scenario_specs], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("Scenario weights must sum to > 0.")
    weights = weights / weights.sum()

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

    gender_map, race_map = _build_patient_attr_maps(df_p)

    obs_max = _max_date_by_patient(df_o, "patient_id", "date") if force_latest_dates else {}
    med_max = _max_date_by_patient(df_m, "patient_id", "date") if force_latest_dates else {}
    cond_max = _max_date_by_patient(df_c, "patient_id", "onset_date") if force_latest_dates else {}

    new_meds: list[dict[str, Any]] = []
    new_obs: list[dict[str, Any]] = []
    new_conds: list[dict[str, Any]] = []

    manifest_rows: list[dict[str, Any]] = []

    # --- Injection Helpers ---
    def add_wegovy(pid: str, date_str: str) -> None:
        new_meds.append({
            "patient_id": pid,
            "medication_name": "Wegovy 2.4 MG Injection",
            "rx_code": "",
            "date": date_str,
            "status": "active",
        })

    def add_concurrent_glp1(pid: str, date_str: str) -> None:
        new_meds.append({
            "patient_id": pid,
            "medication_name": "Ozempic (semaglutide) injection",
            "rx_code": "",
            "date": date_str,
            "status": "active",
        })

    def add_bmi(pid: str, bmi: float, date_str: str) -> None:
        # Calculate enrichments
        icd10 = _get_icd10_code(bmi)
        z68 = _get_z68_code(bmi)

        new_obs.append({
            "patient_id": pid,
            "type": "BMI",
            "value": round(float(bmi), 1),
            "unit": "kg/m2",
            "date": date_str,
            "icd10_dx": icd10,   # Populate enriched columns
            "icd10_bmi": z68
        })

    def add_height_weight(pid: str, height_cm: float, weight_kg: float, date_str: str) -> None:
        new_obs.append({"patient_id": pid, "type": "Height", "value": round(float(height_cm), 1), "unit": "cm", "date": date_str, "icd10_dx": None, "icd10_bmi": None})
        new_obs.append({"patient_id": pid, "type": "Weight", "value": round(float(weight_kg), 1), "unit": "kg", "date": date_str, "icd10_dx": None, "icd10_bmi": None})

    def add_height_only(pid: str, height_cm: float, date_str: str) -> None:
        new_obs.append({"patient_id": pid, "type": "Height", "value": round(float(height_cm), 1), "unit": "cm", "date": date_str, "icd10_dx": None, "icd10_bmi": None})

    def add_weight_only(pid: str, weight_kg: float, date_str: str) -> None:
        new_obs.append({"patient_id": pid, "type": "Weight", "value": round(float(weight_kg), 1), "unit": "kg", "date": date_str, "icd10_dx": None, "icd10_bmi": None})

    def add_condition(pid: str, name: str, onset_str: str, icd10_dx: str | None = None, icd10_bmi: str | None = None) -> None:
        new_conds.append({
            "patient_id": pid,
            "condition_name": name,
            "code": "",
            "onset_date": onset_str,
            "icd10_dx": icd10_dx,
            "icd10_bmi": icd10_bmi,
        })

    def draw_scenario_for_pid(pid: str) -> ScenarioSpec:
        gender = gender_map.get(pid, "")
        for _ in range(20):
            idx = int(rng.choice(len(scenario_specs), p=weights))
            spec = scenario_specs[idx]
            if spec.name == "SAFETY_PREGNANCY_NURSING" and "female" not in (gender or ""):
                continue
            return spec
        non_preg = [s for s in scenario_specs if s.name != "SAFETY_PREGNANCY_NURSING"]
        idx = int(rng.integers(0, len(non_preg)))
        return non_preg[idx]

    # --- Main Injection Loop ---
    for pid in cohort_ids:
        pid = str(pid)

        obs_date = _next_injection_date(pid, obs_max, fallback=datetime(2025, 1, 1), days_ahead=1) if force_latest_dates else "2025-01-02"
        med_date = _next_injection_date(pid, med_max, fallback=datetime(2025, 1, 1), days_ahead=2) if force_latest_dates else "2025-01-03"
        onset_date = _next_injection_date(pid, cond_max, fallback=datetime(2024, 1, 1), days_ahead=1) if force_latest_dates else "2024-01-02"

        spec = draw_scenario_for_pid(pid)

        if sanitize_for_expected_outcome:
            if not spec.name.startswith("SAFETY_"):
                df_c = _drop_conditions_matching(df_c, pid, SAFETY_SUBSTRINGS)
                df_m = _drop_meds_matching(df_m, pid, PROHIBITED_GLP1)
            if spec.name in ("OVERWEIGHT_AMBIGUOUS_TERM", "OVERWEIGHT_NO_COMORBIDITY", "BMI_UNDER_27", "OBESE_BMI30_PLUS"):
                df_c = _drop_conditions_matching(df_c, pid, QUALIFYING_CONDITION_SUBSTRINGS)

        add_wegovy(pid, med_date)

        injected: dict[str, Any] = {
            "injected_bmi": None,
            "injected_height_cm": None,
            "injected_weight_kg": None,
            "injected_condition": None,
            "injected_med": None,
            "bmi_deleted": False,
            "height_deleted": False,
            "weight_deleted": False,
        }

        # Scenario Implementation
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
            df_o = _drop_observations(df_o, pid, ["BMI"])
            injected["bmi_deleted"] = True
            height_cm = float(rng.uniform(165.0, 185.0))
            target_bmi = float(rng.uniform(31.0, 38.0))
            height_m = height_cm / 100.0
            weight_kg = target_bmi * (height_m**2)
            add_height_weight(pid, height_cm, weight_kg, obs_date)
            injected["injected_height_cm"] = round(height_cm, 1)
            injected["injected_weight_kg"] = round(weight_kg, 1)

        elif spec.name == "DATA_GAP_MISSING_HEIGHT":
            df_o = _drop_observations(df_o, pid, ["BMI", "Height"])
            injected["bmi_deleted"] = True
            injected["height_deleted"] = True
            weight_kg = float(rng.uniform(75.0, 140.0))
            add_weight_only(pid, weight_kg, obs_date)
            injected["injected_weight_kg"] = round(weight_kg, 1)

        elif spec.name == "DATA_GAP_MISSING_WEIGHT":
            df_o = _drop_observations(df_o, pid, ["BMI", "Weight"])
            injected["bmi_deleted"] = True
            injected["weight_deleted"] = True
            height_cm = float(rng.uniform(150.0, 195.0))
            add_height_only(pid, height_cm, obs_date)
            injected["injected_height_cm"] = round(height_cm, 1)

        elif spec.name == "OBESE_BMI30_PLUS":
            bmi = float(rng.uniform(30.0, 45.0))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

            # Strict Verification: Must have Obesity diagnosis with codes
            # Derived from the injected observation logic
            icd10_dx = _get_icd10_code(bmi) or "E66.9"
            icd10_bmi = _get_z68_code(bmi)
            add_condition(pid, "Obesity", onset_date, icd10_dx=icd10_dx, icd10_bmi=icd10_bmi)

        elif spec.name == "OBESE_NO_CODES":
            # New Scenario: High BMI and "Obesity" text, but NO underlying codes.
            bmi = float(rng.uniform(30.0, 45.0))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

            # Add "Obesity" text ONLY. No codes passed.
            add_condition(pid, "Obesity", onset_date, icd10_dx=None, icd10_bmi=None)

        elif spec.name == "OVERWEIGHT_VALID_COMORBIDITY":
            bmi = float(rng.uniform(27.0, 29.9))
            add_bmi(pid, bmi, obs_date)

            # Strict Verification: Must have Overweight diagnosis
            icd10_dx = _get_icd10_code(bmi) or "E66.3"
            icd10_bmi = _get_z68_code(bmi)
            add_condition(pid, "Overweight", onset_date, icd10_dx=icd10_dx, icd10_bmi=icd10_bmi)

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
            bmi = float(rng.uniform(27.0, 29.9))
            add_bmi(pid, bmi, obs_date)
            injected["injected_bmi"] = round(bmi, 1)

        # Enrichment for Manifest
        inj_bmi_val = injected.get("injected_bmi")
        injected["injected_icd10"] = _get_icd10_code(inj_bmi_val)
        injected["injected_z68"] = _get_z68_code(inj_bmi_val)

        manifest_rows.append({
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
        })

    # Concatenate new data
    if new_meds:
        df_m = pd.concat([df_m, pd.DataFrame(new_meds)], ignore_index=True)
    if new_obs:
        df_o = pd.concat([df_o, pd.DataFrame(new_obs)], ignore_index=True)
    if new_conds:
        df_c = pd.concat([df_c, pd.DataFrame(new_conds)], ignore_index=True)

    # Normalize
    for df in (df_p, df_c, df_m, df_o):
        if df is not None and not df.empty and "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)

    manifest_df = pd.DataFrame(manifest_rows)

    try:
        dist = manifest_df["scenario"].value_counts().to_dict()
        logger.info("Scenario distribution: %s", dist)
    except Exception:
        pass

    if return_manifest:
        return df_p, df_c, df_m, df_o, manifest_df
    return df_p, df_c, df_m, df_o


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_dir = Path("output")
    _ensure_output_dir(out_dir)

    env_claim_count = os.getenv("PA_CLAIM_COUNT")
    env_claim_rate = os.getenv("PA_CLAIM_RATE")
    env_seed = os.getenv("PA_SEED")

    claim_count: int | None = int(env_claim_count) if env_claim_count else None
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

    # Write canonical CSVs
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
            "Manifest enriched with injected_icd10 and injected_z68 codes.",
        ],
    }

    # Sanitize manifest for JSON compliance (NaN -> None)
    manifest_clean = manifest.astype(object).where(pd.notnull(manifest), None)
    payload = {"metadata": meta, "claims": manifest_clean.to_dict(orient="records")}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    try:
        manifest.to_csv(manifest_csv_path, index=False)
    except Exception as e:
        logger.warning("Could not write manifest CSV: %s", e)

    logger.info("Data generation complete.")
    logger.info("Wrote enriched manifest to: %s", manifest_path)

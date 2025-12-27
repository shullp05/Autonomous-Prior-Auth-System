"""
etl_pipeline.py - FHIR -> Flat Files ETL

Reads Synthea-style FHIR Bundles (JSON) and produces the normalized CSV artifacts
used by the PA pipeline:

- output/data_patients.csv
- output/data_conditions.csv
- output/data_medications.csv
- output/data_observations.csv (Enriched with ICD-10 and Z68 codes)

Compatibility notes:
- Guarantees canonical columns + patient_id as string across all frames
- Observation "type" is one of: BMI, Height, Weight (exact casing)
- Auto-calculates ICD-10 E66.* and Z68.* codes for BMI observations
- Resolves urn:uuid / fullUrl references for stable joins

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Updated imports to include ICD-10 constants
from config import (
    ICD10_OBESITY,
    ICD10_OVERWEIGHT,
    LOINC_BMI,
    LOINC_HEIGHT,
    LOINC_WEIGHT,
)

logger = logging.getLogger(__name__)

# Default to recursive scan; override with ETL_FHIR_GLOB if needed
DEFAULT_FHIR_GLOB = "./output/fhir/**/*.json"
FHIR_GLOB = os.getenv("ETL_FHIR_GLOB", DEFAULT_FHIR_GLOB)

# Output paths (aligns with batch_runner expectations)
OUTPUT_DIR = Path(os.getenv("ETL_OUTPUT_DIR", "output"))
OUT_PATIENTS = OUTPUT_DIR / "data_patients.csv"
OUT_CONDITIONS = OUTPUT_DIR / "data_conditions.csv"
OUT_MEDS = OUTPUT_DIR / "data_medications.csv"
OUT_OBS = OUTPUT_DIR / "data_observations.csv"


def _safe_get(d: dict[str, Any], path: list[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_diagnosis_map() -> dict[str, str]:
    """
    Load SNOMED -> ICD-10 DX mapping from diagnosisMap.csv.
    Returns: Dict[snomed_code, icd10_dx]
    """
    map_path = Path("diagnosisMap.csv")
    if not map_path.exists():
        logger.warning("diagnosisMap.csv not found at %s", map_path)
        return {}

    mapping = {}
    try:
        df_map = pd.read_csv(map_path)
        # We want Correlating_SNOMED_Code -> icd10_dx
        # Since map has multiple rows per SNOMED (for BMI ranges), we pick the first/default for the condition check.
        # Ideally, we'd pick based on BMI, but here we just map the Code itself.
        # E66.9 is safe for 162864005 (Obesity) if unspecified.
        # E66.3 is safe for 162863004 (Overweight).
        for _, row in df_map.iterrows():
            snomed = str(row.get("Correlating_SNOMED_Code", "")).strip()
            dx = str(row.get("icd10_dx", "")).strip()
            if snomed and dx:
                # Prefer E66.9/E66.3 over others if duplicates, or just take first found?
                # The CSV sorts Overweight -> E66.3, Obesity -> E66.9, then later E66.01
                # If we encounter a new code for existing key, maybe keep existing?
                if snomed not in mapping:
                    mapping[snomed] = dx
                elif dx == "E66.9" and mapping[snomed] != "E66.9":
                     # Allow override? No, simple map for now.
                     pass
    except Exception as e:
        logger.error("Failed to load diagnosisMap.csv: %s", e)

    return mapping


def _first_nonempty_str(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _codeableconcept_display(cc: dict[str, Any]) -> str:
    """Return best-effort display/text for a CodeableConcept."""
    if not isinstance(cc, dict):
        return ""
    codings = cc.get("coding") or []
    if isinstance(codings, list):
        for c in codings:
            if not isinstance(c, dict):
                continue
            d = c.get("display")
            if isinstance(d, str) and d.strip():
                return d.strip()
    t = cc.get("text")
    return t.strip() if isinstance(t, str) else ""


def _any_coding_code(resource: dict[str, Any]) -> list[str]:
    """Return all codes found in resource.code.coding[].code (as strings)."""
    codes: list[str] = []
    cc = resource.get("code") or {}
    codings = (cc.get("coding") or []) if isinstance(cc, dict) else []
    if isinstance(codings, list):
        for c in codings:
            if isinstance(c, dict) and isinstance(c.get("code"), str):
                codes.append(c["code"].strip())
    return [c for c in codes if c]


def _normalize_ref_string(ref: str) -> str:
    return (ref or "").strip()


def _extract_ref_id(ref: str, resource_type: str) -> str | None:
    """
    Extract <id> from common FHIR reference forms:
      - ResourceType/<id>
      - https://.../ResourceType/<id>
      - urn:uuid:<id>
    """
    ref = _normalize_ref_string(ref)
    if not ref:
        return None

    # urn:uuid:<id>
    if ref.lower().startswith("urn:uuid:"):
        val = ref.split(":", 2)[-1].strip()
        return val or None

    needle = f"{resource_type}/"
    if needle in ref:
        try:
            return ref.split(needle, 1)[1].split("/", 1)[0].strip() or None
        except Exception:
            return None

    return None


def _extract_race_us_core(patient_resource: dict[str, Any]) -> str:
    """
    Extract race from US Core Race extension if present.
    """
    race = "Unknown"
    for ext in patient_resource.get("extension", []) or []:
        url = (ext.get("url") or "").lower()
        if "us-core-race" not in url:
            continue

        for sub_ext in ext.get("extension", []) or []:
            if sub_ext.get("url") == "text" and isinstance(sub_ext.get("valueString"), str):
                val = sub_ext["valueString"].strip()
                if val:
                    return val

        for sub_ext in ext.get("extension", []) or []:
            if sub_ext.get("url") == "ombCategory":
                display = _safe_get(sub_ext, ["valueCoding", "display"])
                if isinstance(display, str) and display.strip():
                    return display.strip()

    return race


def _calculate_bmi_codes(row: pd.Series) -> tuple[str | None, str | None]:
    """
    Apply logic to map numeric BMI to ICD-10 Diagnosis (E66.*) and Z-Code (Z68.*).
    This runs during ETL enrichment before Chaos Monkey sees the data.
    
    Returns tuple: (icd10_dx, icd10_bmi)
    """
    # Only process BMI rows
    if row.get("type") != "BMI":
        return None, None

    try:
        bmi = float(row.get("value"))
    except (ValueError, TypeError):
        return None, None

    if pd.isna(bmi):
        return None, None

    # Calculate Z-Code (Z68.xx)
    # Z68.1: <19.9
    # Z68.20-29: 20-29.9 (Each 1.0)
    # Z68.30-39: 30-39.9 (Each 1.0)
    # Z68.41: 40-44.9
    # Z68.42: 45-49.9
    # Z68.43: 50-59.9
    # Z68.44: 60-69.9
    # Z68.45: 70+

    icd10_bmi = None
    if bmi < 20:
         icd10_bmi = "Z68.1"
    elif 20 <= bmi < 40:
        # Easy mapping: Z68. + integral part
        # e.g. 29.9 -> 29 -> "Z68.29"
        # e.g. 30.1 -> 30 -> "Z68.30"
        val = int(bmi)
        icd10_bmi = f"Z68.{val}"
    elif 40 <= bmi < 45:
        icd10_bmi = "Z68.41"
    elif 45 <= bmi < 50:
        icd10_bmi = "Z68.42"
    elif 50 <= bmi < 60:
        icd10_bmi = "Z68.43"
    elif 60 <= bmi < 70:
        icd10_bmi = "Z68.44"
    elif bmi >= 70:
        icd10_bmi = "Z68.45"

    # Calculate ICD-10 DX (E66.x)
    icd10_dx = None
    if bmi >= 30:
        # Obesity, unspecified
        icd10_dx = ICD10_OBESITY if 'ICD10_OBESITY' in globals() else "E66.9"
    elif bmi >= 25:
        # Overweight
        icd10_dx = ICD10_OVERWEIGHT if 'ICD10_OVERWEIGHT' in globals() else "E66.3"

    return icd10_dx, icd10_bmi


def parse_fhir_bundle(
    file_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse a single FHIR Bundle JSON file into normalized rows.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    demographics: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []
    medications: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    bundle_patient_id: str | None = None

    entries = data.get("entry", []) or []
    if not isinstance(entries, list):
        entries = []

    # Reference resolvers
    patient_ref_map: dict[str, str] = {}
    medication_catalog: dict[str, str] = {}

    # ---------- First pass: capture Patient + Medication catalogs ----------
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue

        r_type = resource.get("resourceType")
        full_url = _normalize_ref_string(entry.get("fullUrl") or "")

        if r_type == "Patient":
            pid = _first_nonempty_str(resource.get("id"))
            if pid:
                bundle_patient_id = pid or bundle_patient_id
                patient_ref_map[pid] = pid
                patient_ref_map[f"Patient/{pid}"] = pid
                if full_url:
                    patient_ref_map[full_url] = pid
                    if full_url.lower().startswith("urn:uuid:"):
                        patient_ref_map[full_url.split(":", 2)[-1].strip()] = pid

        elif r_type == "Medication":
            mid = _first_nonempty_str(resource.get("id"))
            name = _codeableconcept_display(resource.get("code") or {}) or _first_nonempty_str(
                resource.get("text", {}).get("div") if isinstance(resource.get("text"), dict) else "",
            )
            if not name:
                name = _first_nonempty_str(resource.get("status"))

            if mid and name:
                medication_catalog[mid] = name
                medication_catalog[f"Medication/{mid}"] = name
            if full_url and name:
                medication_catalog[full_url] = name
                if full_url.lower().startswith("urn:uuid:"):
                    medication_catalog[full_url.split(":", 2)[-1].strip()] = name

    def resolve_patient_id_from_resource(resource: dict[str, Any]) -> str | None:
        for ref_path in (["subject", "reference"], ["patient", "reference"], ["beneficiary", "reference"]):
            ref = _safe_get(resource, ref_path)
            if not isinstance(ref, str) or not ref.strip():
                continue
            ref_s = _normalize_ref_string(ref)
            if ref_s in patient_ref_map:
                return patient_ref_map[ref_s]
            extracted = _extract_ref_id(ref_s, "Patient")
            if extracted:
                if extracted in patient_ref_map:
                    return patient_ref_map[extracted]
                return extracted
        return bundle_patient_id

    def resolve_med_name_from_request(resource: dict[str, Any]) -> tuple[str, str]:
        med_name = ""
        rx_code = ""
        mcc = resource.get("medicationCodeableConcept")
        if isinstance(mcc, dict):
            codings = mcc.get("coding") or []
            if isinstance(codings, list) and codings:
                c0 = codings[0] if isinstance(codings[0], dict) else {}
                rx_code = _first_nonempty_str(c0.get("code"))
                med_name = _first_nonempty_str(c0.get("display"), mcc.get("text"))
            else:
                med_name = _first_nonempty_str(mcc.get("text"))

        if med_name:
            return med_name, rx_code

        mref = _safe_get(resource, ["medicationReference", "reference"])
        if isinstance(mref, str) and mref.strip():
            ref = _normalize_ref_string(mref)
            if ref in medication_catalog:
                return medication_catalog[ref], rx_code
            extracted = _extract_ref_id(ref, "Medication")
            if extracted:
                if extracted in medication_catalog:
                    return medication_catalog[extracted], rx_code
                if f"Medication/{extracted}" in medication_catalog:
                    return medication_catalog[f"Medication/{extracted}"], rx_code
        return "", rx_code

    # ---------- Second pass: extract normalized rows ----------
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue

        r_type = resource.get("resourceType")
        if not r_type:
            continue

        if r_type == "Patient":
            bundle_patient_id = _first_nonempty_str(resource.get("id")) or bundle_patient_id
            given = ""
            family = ""
            try:
                nm0 = (resource.get("name") or [])[0] if (resource.get("name") or []) else {}
                given_list = nm0.get("given") or []
                given = given_list[0] if given_list else ""
                family = nm0.get("family") or ""
            except Exception:
                pass

            race = _extract_race_us_core(resource)
            demographics = {
                "patient_id": bundle_patient_id,
                "name": (f"{given} {family}").strip() or "Unknown",
                "gender": resource.get("gender") or "unknown",
                "race": race or "Unknown",
                "dob": resource.get("birthDate") or "",
            }

        elif r_type == "Condition":
            pid = resolve_patient_id_from_resource(resource)
            code0 = ((resource.get("code") or {}).get("coding") or [{}])[0] or {}
            display = _first_nonempty_str(code0.get("display"), (resource.get("code") or {}).get("text"))
            onset = resource.get("onsetDateTime") or resource.get("recordedDate") or ""
            conditions.append({
                "patient_id": pid,
                "condition_name": display,
                "code": code0.get("code") or "",
                "onset_date": (onset[:10] if isinstance(onset, str) else ""),
            })

        elif r_type == "MedicationRequest":
            pid = resolve_patient_id_from_resource(resource)
            authored = resource.get("authoredOn") or ""
            med_name, rx_code = resolve_med_name_from_request(resource)
            medications.append({
                "patient_id": pid,
                "medication_name": med_name or "",
                "rx_code": rx_code or "",
                "date": (authored[:10] if isinstance(authored, str) else ""),
                "status": resource.get("status") or "",
            })

        elif r_type == "Observation":
            pid = resolve_patient_id_from_resource(resource)
            codes = set(_any_coding_code(resource))
            target_codes = {LOINC_BMI: "BMI", LOINC_WEIGHT: "Weight", LOINC_HEIGHT: "Height"}

            matched: str | None = None
            for loinc, label in target_codes.items():
                if loinc in codes:
                    matched = label
                    break
            if not matched:
                continue

            vq = resource.get("valueQuantity") or {}
            if not isinstance(vq, dict) or "value" not in vq:
                continue

            eff = resource.get("effectiveDateTime") or resource.get("issued") or ""
            if not eff and isinstance(resource.get("effectivePeriod"), dict):
                eff = resource["effectivePeriod"].get("start") or resource["effectivePeriod"].get("end") or ""

            try:
                val = float(vq["value"])
            except Exception:
                continue

            observations.append({
                "patient_id": pid,
                "type": matched,
                "value": round(val, 2),
                "unit": vq.get("unit") or "",
                "date": (eff[:10] if isinstance(eff, str) else ""),
            })

    return demographics, conditions, medications, observations


def run_etl(
    write_csv: bool = True,
    dedupe: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run ETL across all FHIR JSON files and optionally write the normalized CSVs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning for FHIR files with glob: %s", FHIR_GLOB)
    files = glob.glob(FHIR_GLOB, recursive=True)

    df_pat_cols = ["patient_id", "name", "gender", "race", "dob"]
    # Update to include enriched columns
    df_cond_cols = ["patient_id", "condition_name", "code", "onset_date", "icd10_dx", "icd10_bmi"]
    df_med_cols = ["patient_id", "medication_name", "rx_code", "date", "status"]
    # Update cols to include new enriched fields
    df_obs_cols = ["patient_id", "type", "value", "unit", "date", "icd10_dx", "icd10_bmi"]

    if not files:
        logger.error("No FHIR files found for glob: %s", FHIR_GLOB)
        empty = (
            pd.DataFrame(columns=df_pat_cols),
            pd.DataFrame(columns=df_cond_cols),
            pd.DataFrame(columns=df_med_cols),
            pd.DataFrame(columns=df_obs_cols),
        )
        if write_csv:
            empty[0].to_csv(OUT_PATIENTS, index=False)
            empty[1].to_csv(OUT_CONDITIONS, index=False)
            empty[2].to_csv(OUT_MEDS, index=False)
            empty[3].to_csv(OUT_OBS, index=False)
            logger.info("Wrote empty CSVs to %s", OUTPUT_DIR)
        return empty

    all_demographics: list[dict[str, Any]] = []
    all_conditions: list[dict[str, Any]] = []
    all_medications: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []

    parsed_files = 0
    skipped_files = 0

    for fpath in files:
        try:
            demo, cond, meds, obs = parse_fhir_bundle(fpath)
        except Exception as e:
            skipped_files += 1
            logger.warning("Skipping unreadable bundle '%s': %s", fpath, e)
            continue

        demo_pid = demo.get("patient_id") if isinstance(demo, dict) else None
        if not demo_pid:
            skipped_files += 1
            continue

        # Ensure pid filled even if subject refs were missing
        for c in cond:
            c["patient_id"] = c.get("patient_id") or demo_pid
        for m in meds:
            m["patient_id"] = m.get("patient_id") or demo_pid
        for o in obs:
            o["patient_id"] = o.get("patient_id") or demo_pid

        all_demographics.append(demo)
        all_conditions.extend(cond)
        all_medications.extend(meds)
        all_observations.extend(obs)
        parsed_files += 1

    df_pat = pd.DataFrame(all_demographics, columns=df_pat_cols)
    # Initialize df_con with extra cols to avoid warnings
    for c in all_conditions:
        c["icd10_dx"] = None
        c["icd10_bmi"] = None
    df_con = pd.DataFrame(all_conditions, columns=df_cond_cols)
    df_med = pd.DataFrame(all_medications, columns=df_med_cols)
    # Note: df_obs columns will be set during creation or enrichment
    df_obs = pd.DataFrame(all_observations, columns=["patient_id", "type", "value", "unit", "date"])

    # Normalize patient_id to str across frames
    for df in (df_pat, df_con, df_med, df_obs):
        if not df.empty and "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)

    # --- ENRICHMENT STEP: ADD ICD-10 and Z68 CODES ---
    if not df_obs.empty:
        logger.info("Enriching observations with ICD-10 and Z68 codes...")

        # We use apply with result_type='expand' to map the tuple return to two columns
        enrichment = df_obs.apply(_calculate_bmi_codes, axis=1, result_type='expand')
        enrichment.columns = ["icd10_dx", "icd10_bmi"]

        # Concatenate results back
        df_obs = pd.concat([df_obs, enrichment], axis=1)
    else:
        # Create empty columns if no data
        df_obs["icd10_dx"] = None
        df_obs["icd10_bmi"] = None

    # --- ENRICHMENT STEP 2: STAMP CONDITIONS WITH ICD-10 DX (Map) AND BMI (Obs) ---
    logger.info("Enriching conditions with diagnosis map and patient BMI context...")

    # 1. Load map
    snomed_map = load_diagnosis_map()

    # 2. Build Patient BMI Context (Latest valid Z-code)
    # We want the most recent Z-code for each patient to stamp onto their Obesity condition
    pat_z_map = {}
    if not df_obs.empty and "icd10_bmi" in df_obs.columns:
        # Filter for rows with Z-codes
        z_rows = df_obs.dropna(subset=["icd10_bmi"])
        # Sort by date descending (assuming date col exists and is sortable YYYY-MM-DD or similar)
        # df_obs already has 'date' string.
        z_rows = z_rows.sort_values(by="date", ascending=True) # Ascending, then take last = latest

        g = z_rows.groupby("patient_id")["icd10_bmi"].last()
        pat_z_map = g.to_dict()

    # 3. Apply to Conditions
    def enrich_condition(row):
        # ICD-10 DX from SNOMED Logic
        code = str(row.get("code", "")).strip()
        if code in snomed_map:
            row["icd10_dx"] = snomed_map[code]

        # ICD-10 BMI (Z-code) from Patient Context
        # Only enrich if condition relates to Obesity/Overweight?
        # Workflow implies check implies "Obesity" name match too?
        # But we can stamp Z-code if available for the patient, relevant for checking.
        # Better to stamp it.
        pid = str(row.get("patient_id"))
        if pid in pat_z_map:
            row["icd10_bmi"] = pat_z_map[pid]

        return row

    if not df_con.empty:
        df_con = df_con.apply(enrich_condition, axis=1)

    # Optional dedupe (exact row duplicates)
    if dedupe:
        if not df_pat.empty:
            df_pat = df_pat.drop_duplicates(subset=["patient_id"], keep="first")
        if not df_con.empty:
            df_con = df_con.drop_duplicates()
        if not df_med.empty:
            df_med = df_med.drop_duplicates()
        if not df_obs.empty:
            df_obs = df_obs.drop_duplicates()

    logger.info(
        "ETL Complete: parsed=%d skipped=%d patients=%d conditions=%d meds=%d obs=%d",
        parsed_files,
        skipped_files,
        len(df_pat),
        len(df_con),
        len(df_med),
        len(df_obs),
    )

    if write_csv:
        df_pat.to_csv(OUT_PATIENTS, index=False)
        df_con.to_csv(OUT_CONDITIONS, index=False)
        df_med.to_csv(OUT_MEDS, index=False)
        df_obs.to_csv(OUT_OBS, index=False)
        logger.info("Wrote CSVs to %s", OUTPUT_DIR)

    return df_pat, df_con, df_med, df_obs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df_pat, df_con, df_med, df_obs = run_etl(write_csv=True, dedupe=True)
    print(f"ETL Complete. Processed {len(df_pat)} unique patients.")

    if not df_pat.empty and "race" in df_pat.columns:
        print("\nRace Distribution Found:")
        print(df_pat["race"].fillna("Unknown").value_counts())

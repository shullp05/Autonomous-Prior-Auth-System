"""
etl_pipeline.py - FHIR -> Flat Files ETL

Reads Synthea-style FHIR Bundles (JSON) and produces the normalized CSV artifacts
used by the PA pipeline:

- output/data_patients.csv
- output/data_conditions.csv
- output/data_medications.csv
- output/data_observations.csv

Compatibility notes (for updated chaos_monkey.py):
- Guarantees canonical columns + patient_id as string across all frames
- Observation "type" is one of: BMI, Height, Weight (exact casing)
- Resolves urn:uuid / fullUrl references so Condition/Observation/MedicationRequest
  patient_id matches the Patient.patient_id row (critical for stable joins)
- Resolves MedicationRequest.medicationReference via Medication fullUrl/id catalog
- Writes CSV outputs by default (downstream batch_runner can run immediately)

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import LOINC_BMI, LOINC_HEIGHT, LOINC_WEIGHT

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


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_nonempty_str(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _codeableconcept_display(cc: Dict[str, Any]) -> str:
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


def _any_coding_code(resource: Dict[str, Any]) -> List[str]:
    """Return all codes found in resource.code.coding[].code (as strings)."""
    codes: List[str] = []
    cc = resource.get("code") or {}
    codings = (cc.get("coding") or []) if isinstance(cc, dict) else []
    if isinstance(codings, list):
        for c in codings:
            if isinstance(c, dict) and isinstance(c.get("code"), str):
                codes.append(c["code"].strip())
    return [c for c in codes if c]


def _normalize_ref_string(ref: str) -> str:
    return (ref or "").strip()


def _extract_ref_id(ref: str, resource_type: str) -> Optional[str]:
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


def _extract_race_us_core(patient_resource: Dict[str, Any]) -> str:
    """
    Extract race from US Core Race extension if present.
    Prefers:
      - extension[].extension[url="text"].valueString
    Fallback:
      - extension[].extension[url="ombCategory"].valueCoding.display
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


def parse_fhir_bundle(
    file_path: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse a single FHIR Bundle JSON file into normalized rows.

    Important: resolves urn:uuid/fullUrl references to canonical Patient.id
    so downstream joins (and chaos_monkey injections) behave predictably.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    demographics: Dict[str, Any] = {}
    conditions: List[Dict[str, Any]] = []
    medications: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []

    bundle_patient_id: Optional[str] = None

    entries = data.get("entry", []) or []
    if not isinstance(entries, list):
        entries = []

    # Reference resolvers:
    # - patient_ref_map maps various references (fullUrl, urn:uuid, Patient/<id>, raw <id>) -> canonical Patient.id
    # - medication_catalog maps Medication reference keys -> best-effort medication display name
    patient_ref_map: Dict[str, str] = {}
    medication_catalog: Dict[str, str] = {}

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

                # map canonical variants -> pid
                patient_ref_map[pid] = pid
                patient_ref_map[f"Patient/{pid}"] = pid

                # map fullUrl / urn:uuid -> pid
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
                # last-ditch fallback: sometimes "status" is present; better than empty
                name = _first_nonempty_str(resource.get("status"))

            # Map by resource.id
            if mid and name:
                medication_catalog[mid] = name
                medication_catalog[f"Medication/{mid}"] = name

            # Map by fullUrl / urn:uuid if present
            if full_url and name:
                medication_catalog[full_url] = name
                if full_url.lower().startswith("urn:uuid:"):
                    medication_catalog[full_url.split(":", 2)[-1].strip()] = name

    def resolve_patient_id_from_resource(resource: Dict[str, Any]) -> Optional[str]:
        """
        Resolve patient id from common fields using patient_ref_map.
        Falls back to extracted ids and bundle_patient_id.
        """
        for ref_path in (["subject", "reference"], ["patient", "reference"], ["beneficiary", "reference"]):
            ref = _safe_get(resource, ref_path)
            if not isinstance(ref, str) or not ref.strip():
                continue

            ref_s = _normalize_ref_string(ref)
            # Direct map hits (fullUrl / urn uuid / Patient/<id>)
            if ref_s in patient_ref_map:
                return patient_ref_map[ref_s]

            # Extracted id from Patient/<id> or urn:uuid:<id>
            extracted = _extract_ref_id(ref_s, "Patient")
            if extracted:
                # extracted might be the canonical Patient.id OR a fullUrl UUID mapped to it
                if extracted in patient_ref_map:
                    return patient_ref_map[extracted]
                return extracted

        return bundle_patient_id

    def resolve_med_name_from_request(resource: Dict[str, Any]) -> Tuple[str, str]:
        """
        Resolve (med_name, rx_code) for MedicationRequest from either:
        - medicationCodeableConcept
        - medicationReference (+ medication_catalog by id/fullUrl/urn uuid)
        """
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

            # Direct catalog hits (fullUrl / urn uuid / Medication/<id>)
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
            conditions.append(
                {
                    "patient_id": pid,
                    "condition_name": display,
                    "code": code0.get("code") or "",
                    "onset_date": (onset[:10] if isinstance(onset, str) else ""),
                }
            )

        elif r_type == "MedicationRequest":
            pid = resolve_patient_id_from_resource(resource)
            authored = resource.get("authoredOn") or ""
            med_name, rx_code = resolve_med_name_from_request(resource)

            medications.append(
                {
                    "patient_id": pid,
                    "medication_name": med_name or "",
                    "rx_code": rx_code or "",
                    "date": (authored[:10] if isinstance(authored, str) else ""),
                    "status": resource.get("status") or "",
                }
            )

        elif r_type == "Observation":
            pid = resolve_patient_id_from_resource(resource)

            # Scan all codings for a relevant LOINC
            codes = set(_any_coding_code(resource))
            target_codes = {LOINC_BMI: "BMI", LOINC_WEIGHT: "Weight", LOINC_HEIGHT: "Height"}

            matched: Optional[str] = None
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

            observations.append(
                {
                    "patient_id": pid,
                    "type": matched,  # must be exactly BMI/Height/Weight for chaos_monkey + batch_runner
                    "value": round(val, 2),
                    "unit": vq.get("unit") or "",
                    "date": (eff[:10] if isinstance(eff, str) else ""),
                }
            )

    return demographics, conditions, medications, observations


def run_etl(
    write_csv: bool = True,
    dedupe: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run ETL across all FHIR JSON files and optionally write the normalized CSVs.

    Args:
        write_csv: If True, writes output/*.csv files expected by downstream pipeline.
        dedupe: If True, drop exact duplicate rows in each output frame.

    Returns:
        (df_patients, df_conditions, df_medications, df_observations)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning for FHIR files with glob: %s", FHIR_GLOB)
    files = glob.glob(FHIR_GLOB, recursive=True)

    df_pat_cols = ["patient_id", "name", "gender", "race", "dob"]
    df_cond_cols = ["patient_id", "condition_name", "code", "onset_date"]
    df_med_cols = ["patient_id", "medication_name", "rx_code", "date", "status"]
    df_obs_cols = ["patient_id", "type", "value", "unit", "date"]

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

    all_demographics: List[Dict[str, Any]] = []
    all_conditions: List[Dict[str, Any]] = []
    all_medications: List[Dict[str, Any]] = []
    all_observations: List[Dict[str, Any]] = []

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
    df_con = pd.DataFrame(all_conditions, columns=df_cond_cols)
    df_med = pd.DataFrame(all_medications, columns=df_med_cols)
    df_obs = pd.DataFrame(all_observations, columns=df_obs_cols)

    # Normalize patient_id to str across frames (critical for chaos_monkey joins)
    for df in (df_pat, df_con, df_med, df_obs):
        if not df.empty and "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)

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

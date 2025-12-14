import json
import glob
from config import LOINC_BMI, LOINC_HEIGHT, LOINC_WEIGHT
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

FHIR_DIR = "./output/fhir/*.json"


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_patient_ref(resource: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of patient id from common FHIR reference fields.
    Synthea bundles often include subject.reference = "Patient/<id>".
    """
    for ref_path in (
        ["subject", "reference"],
        ["patient", "reference"],  # some resources use 'patient'
        ["beneficiary", "reference"],
    ):
        ref = _safe_get(resource, ref_path)
        if isinstance(ref, str) and "Patient/" in ref:
            return ref.split("Patient/", 1)[1].split("/", 1)[0].strip() or None
        if isinstance(ref, str) and ref.startswith("Patient/"):
            return ref.split("/", 1)[1].strip() or None
    return None


def _extract_race_us_core(patient_resource: Dict[str, Any]) -> str:
    """
    Extract race from US Core Race extension if present.
    Handles common synthea-ish shapes:
    - extension.url includes "us-core-race"
      - extension[].url == "text" -> valueString
      - extension[].url == "ombCategory" -> valueCoding.display
    """
    race = "Unknown"

    for ext in patient_resource.get("extension", []) or []:
        url = (ext.get("url") or "").lower()
        if "us-core-race" not in url:
            continue

        # Prefer human-friendly "text"
        for sub_ext in ext.get("extension", []) or []:
            if sub_ext.get("url") == "text" and isinstance(sub_ext.get("valueString"), str):
                val = sub_ext["valueString"].strip()
                if val:
                    return val

        # Fallback to OMB category display
        for sub_ext in ext.get("extension", []) or []:
            if sub_ext.get("url") == "ombCategory":
                display = _safe_get(sub_ext, ["valueCoding", "display"])
                if isinstance(display, str) and display.strip():
                    return display.strip()

    return race


def parse_fhir_bundle(
    file_path: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    demographics: Dict[str, Any] = {}
    conditions: List[Dict[str, Any]] = []
    medications: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []

    bundle_patient_id: Optional[str] = None

    for entry in data.get("entry", []) or []:
        resource = entry.get("resource") or {}
        r_type = resource.get("resourceType")
        if not r_type:
            continue

        if r_type == "Patient":
            bundle_patient_id = resource.get("id") or bundle_patient_id

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
            pid = _extract_patient_ref(resource) or bundle_patient_id
            code0 = ((resource.get("code") or {}).get("coding") or [{}])[0] or {}
            display = code0.get("display") or (resource.get("code") or {}).get("text") or ""
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
            pid = _extract_patient_ref(resource) or bundle_patient_id
            coding0 = ((resource.get("medicationCodeableConcept") or {}).get("coding") or [{}])[0] or {}
            authored = resource.get("authoredOn") or ""
            medications.append(
                {
                    "patient_id": pid,
                    "medication_name": coding0.get("display")
                    or (resource.get("medicationCodeableConcept") or {}).get("text")
                    or "",
                    "rx_code": coding0.get("code") or "",
                    "date": (authored[:10] if isinstance(authored, str) else ""),
                    "status": resource.get("status") or "",
                }
            )

        elif r_type == "Observation":
            pid = _extract_patient_ref(resource) or bundle_patient_id
            coding0 = ((resource.get("code") or {}).get("coding") or [{}])[0] or {}
            code = coding0.get("code")

            target_codes = {LOINC_BMI: "BMI", LOINC_WEIGHT: "Weight", LOINC_HEIGHT: "Height"}
            if code not in target_codes:
                continue

            vq = resource.get("valueQuantity") or {}
            if "value" not in vq:
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
                    "type": target_codes[code],
                    "value": round(val, 2),
                    "unit": vq.get("unit") or "",
                    "date": (eff[:10] if isinstance(eff, str) else ""),
                }
            )

    return demographics, conditions, medications, observations


def run_etl() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Scanning for FHIR files in {FHIR_DIR}...")
    files = glob.glob(FHIR_DIR)

    df_pat_cols = ["patient_id", "name", "gender", "race", "dob"]
    df_cond_cols = ["patient_id", "condition_name", "code", "onset_date"]
    df_med_cols = ["patient_id", "medication_name", "rx_code", "date", "status"]
    df_obs_cols = ["patient_id", "type", "value", "unit", "date"]

    if not files:
        print("ERROR: No FHIR files found.")
        return (
            pd.DataFrame(columns=df_pat_cols),
            pd.DataFrame(columns=df_cond_cols),
            pd.DataFrame(columns=df_med_cols),
            pd.DataFrame(columns=df_obs_cols),
        )

    all_demographics: List[Dict[str, Any]] = []
    all_conditions: List[Dict[str, Any]] = []
    all_medications: List[Dict[str, Any]] = []
    all_observations: List[Dict[str, Any]] = []

    for fpath in files:
        demo, cond, meds, obs = parse_fhir_bundle(fpath)

        demo_pid = demo.get("patient_id") if isinstance(demo, dict) else None
        if not demo_pid:
            continue

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

    df_pat = pd.DataFrame(all_demographics, columns=df_pat_cols)
    df_con = pd.DataFrame(all_conditions, columns=df_cond_cols)
    df_med = pd.DataFrame(all_medications, columns=df_med_cols)
    df_obs = pd.DataFrame(all_observations, columns=df_obs_cols)

    for df in (df_pat, df_con, df_med, df_obs):
        if not df.empty and "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)

    return df_pat, df_con, df_med, df_obs


if __name__ == "__main__":
    df_pat, df_con, df_med, df_obs = run_etl()
    print(f"ETL Complete. Processed {len(df_pat)} patients.")
    if not df_pat.empty and "race" in df_pat.columns:
        print("\nRace Distribution Found:")
        print(df_pat["race"].fillna("Unknown").value_counts())

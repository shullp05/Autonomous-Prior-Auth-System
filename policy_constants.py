# policy_constants.py
"""
Snapshot-derived constants for RX-WEG-2025.

All values are loaded from the canonical policy snapshot (policies/RX-WEG-2025.json)
to prevent drift between deterministic rules, prompts, and governance checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from policy_snapshot import POLICY_ID, SNAPSHOT_PATH, load_policy_snapshot
from schema_validation import validate_policy_snapshot

# Load and validate once at import for deterministic behavior
_SNAPSHOT = load_policy_snapshot(SNAPSHOT_PATH, POLICY_ID)
validate_policy_snapshot(_SNAPSHOT)

# General thresholds
ELIGIBILITY_PATHWAYS: List[Dict] = _SNAPSHOT["eligibility"]["pathways"]
try:
    _OBESITY_PATHWAY = next(p for p in ELIGIBILITY_PATHWAYS if not p.get("required_comorbidity_categories"))
    _OVERWEIGHT_PATHWAY = next(p for p in ELIGIBILITY_PATHWAYS if p.get("required_comorbidity_categories"))
except StopIteration:
    raise ImportError("Policy snapshot invalid: missing required pathways.")

BMI_OBESE_THRESHOLD = float(_OBESITY_PATHWAY["bmi_min"])
BMI_OVERWEIGHT_THRESHOLD = float(_OVERWEIGHT_PATHWAY["bmi_min"])
BMI_MIN_REASONABLE = 5.0
BMI_MAX_REASONABLE = 100.0

# Diagnosis strings
ADULT_OBESITY_DIAGNOSES = _SNAPSHOT["diagnosis_strings"]["adult_obesity"]
ADULT_OVERWEIGHT_DIAGNOSES = _SNAPSHOT["diagnosis_strings"]["adult_excess_weight"]

# Comorbidities (ordered mapping)
COMORBIDITIES: Dict[str, Dict[str, List[str]]] = _SNAPSHOT["comorbidities"]
QUALIFYING_HYPERTENSION = COMORBIDITIES.get("hypertension", {}).get("accepted_strings", [])
QUALIFYING_T2DM = COMORBIDITIES.get("type2_diabetes", {}).get("accepted_strings", [])
QUALIFYING_LIPIDS = COMORBIDITIES.get("dyslipidemia", {}).get("accepted_strings", [])
QUALIFYING_OSA = COMORBIDITIES.get("obstructive_sleep_apnea", {}).get("accepted_strings", [])
QUALIFYING_CVD_PHRASES = COMORBIDITIES.get("cardiovascular_disease", {}).get("accepted_strings", [])
QUALIFYING_CVD_ABBREVS = [phrase for phrase in QUALIFYING_CVD_PHRASES if len(phrase.split()) == 1 or phrase.isupper()]

# Safety exclusions
SAFETY_EXCLUSIONS = _SNAPSHOT["safety_exclusions"]

def _category_terms(name_fragment: str) -> List[str]:
    terms: List[str] = []
    for entry in SAFETY_EXCLUSIONS:
        if name_fragment.lower() in entry["category"].lower():
            terms.extend(entry.get("accepted_strings", []))
    return terms


SAFETY_MTC_MEN2 = _category_terms("Medullary Thyroid Carcinoma") + _category_terms("Multiple Endocrine Neoplasia")
SAFETY_PREGNANCY_LACTATION = _category_terms("Pregnant")
SAFETY_HYPERSENSITIVITY = _category_terms("hypersensitivity")
SAFETY_PANCREATITIS = _category_terms("Pancreatitis")
SAFETY_SUICIDALITY = _category_terms("Suicidality")
SAFETY_GI_MOTILITY = _category_terms("GI motility")

# Drug conflicts
PROHIBITED_GLP1 = _SNAPSHOT["drug_conflicts"]["glp1_or_glp1_gip_agents"]

# Ambiguities (exposed for guardrails/governance)
AMBIGUITIES = _SNAPSHOT.get("ambiguities", [])

def _ambiguity_patterns(keyword: str) -> List[str]:
    return [a["pattern"] for a in AMBIGUITIES if keyword.lower() in a["pattern"].lower()]


AMBIGUOUS_DIABETES = _ambiguity_patterns("diabetes")
AMBIGUOUS_BP = _ambiguity_patterns("blood pressure") + _ambiguity_patterns("hypertension")
AMBIGUOUS_OBESITY = _ambiguity_patterns("obesity")
AMBIGUOUS_SLEEP_APNEA = _ambiguity_patterns("sleep apnea")
AMBIGUOUS_THYROID = _ambiguity_patterns("thyroid")
AMBIGUOUS_APPEAL_TERMS = AMBIGUOUS_BP + AMBIGUOUS_DIABETES + AMBIGUOUS_OBESITY + AMBIGUOUS_SLEEP_APNEA

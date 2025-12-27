# policy_constants.py
"""
Snapshot-derived constants for RX-WEG-2025.

All values are loaded from the canonical policy snapshot (policies/RX-WEG-2025.json)
to prevent drift between deterministic rules, prompts, and governance checks.
"""

from __future__ import annotations

from typing import Any

from policy_snapshot import POLICY_ID, SNAPSHOT_PATH, load_policy_snapshot
from schema_validation import validate_policy_snapshot


def _as_list(val: Any) -> list[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _as_str_list(val: Any) -> list[str]:
    out: list[str] = []
    for x in _as_list(val):
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _require(snapshot: dict, path: list[str]) -> Any:
    cur: Any = snapshot
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise ImportError(f"Policy snapshot invalid: missing key path: {'/'.join(path)}")
        cur = cur[k]
    return cur


# Load and validate once at import for deterministic behavior
_SNAPSHOT = load_policy_snapshot(SNAPSHOT_PATH, POLICY_ID)
validate_policy_snapshot(_SNAPSHOT)

# -----------------------------------------------------------------------------
# General thresholds
# -----------------------------------------------------------------------------
ELIGIBILITY_PATHWAYS: list[dict[str, Any]] = _as_list(_require(_SNAPSHOT, ["eligibility", "pathways"]))

try:
    _OBESITY_PATHWAY = next(
        p for p in ELIGIBILITY_PATHWAYS
        if not _as_list(p.get("required_comorbidity_categories"))
    )
    _OVERWEIGHT_PATHWAY = next(
        p for p in ELIGIBILITY_PATHWAYS
        if _as_list(p.get("required_comorbidity_categories"))
    )
except StopIteration as e:
    raise ImportError("Policy snapshot invalid: missing required eligibility pathways.") from e

BMI_OBESE_THRESHOLD = float(_OBESITY_PATHWAY.get("bmi_min"))
BMI_OVERWEIGHT_THRESHOLD = float(_OVERWEIGHT_PATHWAY.get("bmi_min"))

# Sanity bounds for numeric parsing
BMI_MIN_REASONABLE = 5.0
BMI_MAX_REASONABLE = 100.0

# -----------------------------------------------------------------------------
# Diagnosis strings
# -----------------------------------------------------------------------------
_diag = _SNAPSHOT.get("diagnosis_strings") or {}
ADULT_OBESITY_DIAGNOSES = _as_str_list(_diag.get("adult_obesity"))
ADULT_OVERWEIGHT_DIAGNOSES = _as_str_list(_diag.get("adult_excess_weight"))

# -----------------------------------------------------------------------------
# Comorbidities
# -----------------------------------------------------------------------------
COMORBIDITIES: dict[str, dict[str, Any]] = _SNAPSHOT.get("comorbidities") or {}

QUALIFYING_HYPERTENSION = _as_str_list((COMORBIDITIES.get("hypertension") or {}).get("accepted_strings"))
QUALIFYING_T2DM = _as_str_list((COMORBIDITIES.get("type2_diabetes") or {}).get("accepted_strings"))
QUALIFYING_LIPIDS = _as_str_list((COMORBIDITIES.get("dyslipidemia") or {}).get("accepted_strings"))
QUALIFYING_OSA = _as_str_list((COMORBIDITIES.get("obstructive_sleep_apnea") or {}).get("accepted_strings"))
QUALIFYING_CVD_PHRASES = _as_str_list((COMORBIDITIES.get("cardiovascular_disease") or {}).get("accepted_strings"))

# Anything that is a single token counts as a potential abbrev (CAD, MI, OSA, HFpEF, etc.)
QUALIFYING_CVD_ABBREVS = [p for p in QUALIFYING_CVD_PHRASES if len(p.split()) == 1]

# -----------------------------------------------------------------------------
# Safety exclusions
# -----------------------------------------------------------------------------
SAFETY_EXCLUSIONS: list[dict[str, Any]] = _as_list(_SNAPSHOT.get("safety_exclusions"))

def _category_terms(name_fragment: str) -> list[str]:
    frag = (name_fragment or "").lower()
    if not frag:
        return []
    terms: list[str] = []
    for entry in SAFETY_EXCLUSIONS:
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category") or "").lower()
        if frag in cat:
            terms.extend(_as_str_list(entry.get("accepted_strings")))
    return terms

SAFETY_MTC_MEN2 = _category_terms("medullary thyroid") + _category_terms("multiple endocrine neoplasia")
SAFETY_PREGNANCY_LACTATION = _category_terms("pregnan") + _category_terms("lactat") + _category_terms("breast")
SAFETY_HYPERSENSITIVITY = _category_terms("hypersens")
SAFETY_PANCREATITIS = _category_terms("pancreatitis")
SAFETY_SUICIDALITY = _category_terms("suicid")
SAFETY_GI_MOTILITY = _category_terms("motility") + _category_terms("gastroparesis")

# -----------------------------------------------------------------------------
# Drug conflicts
# -----------------------------------------------------------------------------
_drug_conflicts = _SNAPSHOT.get("drug_conflicts") or {}
PROHIBITED_GLP1 = _as_str_list(_drug_conflicts.get("glp1_or_glp1_gip_agents"))

# -----------------------------------------------------------------------------
# Ambiguities (exposed for guardrails/governance)
# -----------------------------------------------------------------------------
AMBIGUITIES: list[dict[str, Any]] = _as_list(_SNAPSHOT.get("ambiguities"))

def _ambiguity_patterns(keyword: str) -> list[str]:
    kw = (keyword or "").lower()
    if not kw:
        return []
    out: list[str] = []
    for a in AMBIGUITIES:
        if not isinstance(a, dict):
            continue
        patt = str(a.get("pattern") or "")
        if kw in patt.lower():
            out.append(patt)
    return out

AMBIGUOUS_DIABETES = _ambiguity_patterns("diabetes") + _ambiguity_patterns("prediabetes") + _ambiguity_patterns("glucose")
AMBIGUOUS_BP = _ambiguity_patterns("blood pressure") + _ambiguity_patterns("hypertension") + _ambiguity_patterns("bp")
AMBIGUOUS_OBESITY = _ambiguity_patterns("obesity") + _ambiguity_patterns("body mass index") + _ambiguity_patterns("bmi")
AMBIGUOUS_SLEEP_APNEA = _ambiguity_patterns("sleep apnea")
AMBIGUOUS_THYROID = _ambiguity_patterns("thyroid")

AMBIGUOUS_APPEAL_TERMS = AMBIGUOUS_BP + AMBIGUOUS_DIABETES + AMBIGUOUS_OBESITY + AMBIGUOUS_SLEEP_APNEA

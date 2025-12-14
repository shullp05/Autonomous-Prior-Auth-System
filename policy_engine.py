# policy_engine.py
"""
Deterministic Policy Engine for RX-WEG-2025 (Wegovy Chronic Weight Management).

All semantics are sourced from the canonical snapshot (policies/RX-WEG-2025.json).
The LLM is only used downstream to narrate reasoning; eligibility decisions here
are purely rule-based.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from policy_utils import normalize, has_word_boundary, matches_term, expand_safety_variants, is_snf_phrase
from policy_constants import (
    ADULT_OBESITY_DIAGNOSES,
    ADULT_OVERWEIGHT_DIAGNOSES,
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
    SAFETY_EXCLUSIONS,
)
from policy_snapshot import POLICY_ID, SNAPSHOT_PATH, load_policy_snapshot
from schema_validation import validate_policy_snapshot

logger = logging.getLogger(__name__)

_SNAPSHOT = load_policy_snapshot(SNAPSHOT_PATH, POLICY_ID)
validate_policy_snapshot(_SNAPSHOT)
AMBIGUITY_RULES = _SNAPSHOT.get("ambiguities", [])
DOCUMENTATION_REQUIREMENTS = _SNAPSHOT.get("documentation_requirements", [])

ELIGIBILITY_PATHWAYS: List[dict] = _SNAPSHOT["eligibility"]["pathways"]


@dataclass
class EligibilityResult:
    """Structured result from deterministic eligibility evaluation."""

    verdict: str  # APPROVED, DENIED_SAFETY, DENIED_CLINICAL, DENIED_MISSING_INFO, MANUAL_REVIEW
    bmi_numeric: Optional[float]
    safety_flag: str  # CLEAR or DETECTED
    comorbidity_category: str  # NONE, HYPERTENSION, LIPIDS, DIABETES, OSA, CVD
    evidence_quoted: str
    reasoning: str
    policy_path: str  # BMI30_OBESITY, BMI27_COMORBIDITY, BELOW_THRESHOLD, SAFETY_EXCLUSION, AMBIGUITY_MANUAL_REVIEW, UNKNOWN
    decision_type: str  # APPROVED, DENIED_HARD_STOP, DENIED_BMI_THRESHOLD, DENIED_NO_COMORBIDITY, DENIED_MISSING_INFO, FLAGGED_AMBIGUITY
    safety_exclusion_code: Optional[str] = None  # MTC, MEN2, PREGNANCY, BREASTFEEDING, HYPERSENSITIVITY, CONCURRENT_GLP1, PANCREATITIS, SUICIDALITY, GI_MOTILITY
    ambiguity_code: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "verdict": self.verdict,
            "bmi_numeric": self.bmi_numeric,
            "safety_flag": self.safety_flag,
            "comorbidity_category": self.comorbidity_category,
            "evidence_quoted": self.evidence_quoted,
            "reasoning": self.reasoning,
            "policy_path": self.policy_path,
            "decision_type": self.decision_type,
            "safety_exclusion_code": self.safety_exclusion_code,
            "ambiguity_code": self.ambiguity_code,
        }








def _parse_bmi(bmi_raw: str) -> Optional[float]:
    if not bmi_raw:
        return None
    raw = str(bmi_raw).strip()
    if raw.upper() == "MISSING_DATA":
        return None
    match = re.search(r"(\d+\.?\d*)", raw)
    if not match:
        return None
    try:
        bmi = float(match.group(1))
        if BMI_MIN_REASONABLE <= bmi <= BMI_MAX_REASONABLE:
            return round(bmi, 2)
        return None
    except (ValueError, TypeError):
        return None


def _check_safety_exclusions(conditions: List[str], meds: List[str]) -> Tuple[bool, str]:
    conditions_norm = [normalize(c) for c in conditions]
    meds_norm = [normalize(m) for m in meds]

    for cond in conditions_norm:
        for exclusion in SAFETY_EXCLUSIONS:
            for term in exclusion.get("accepted_strings", []):
                for variant in expand_safety_variants(term):
                    if "pregnan" in variant or "breast" in variant or "nursing" in variant or "lactat" in variant:
                        if is_snf_phrase(cond):
                            continue
                    if variant in cond or has_word_boundary(cond, variant):
                        return True, cond

    has_wegovy = any("wegovy" in m for m in meds_norm)
    for med in meds_norm:
        if "wegovy" in med:
            continue
        for bad in PROHIBITED_GLP1:
            for variant in expand_safety_variants(bad):
                if variant in med or has_word_boundary(med, variant):
                    if variant.startswith("semaglutide") and has_wegovy:
                        continue
                    return True, med

    return False, ""


def _find_ambiguity(conditions: List[str], predicate=None) -> Tuple[Optional[dict], str]:
    for cond in conditions:
        cond_lower = normalize(cond)
        for rule in AMBIGUITY_RULES:
            if predicate and not predicate(rule):
                continue
            pattern = rule["pattern"].lower()
            if pattern == "sleep apnea" and ("obstructive" in cond_lower or has_word_boundary(cond_lower, "osa")):
                continue
            if pattern in cond_lower or has_word_boundary(cond_lower, pattern):
                return rule, cond
    return None, ""


def _ambiguity_code(rule: dict) -> Optional[str]:
    pattern = rule.get("pattern", "").lower()
    if "prediabetes" in pattern or "borderline diabetes" in pattern or "impaired fasting glucose" in pattern:
        return "PREDIABETES"
    if "borderline hypertension" in pattern or "blood pressure" in pattern:
        return "BP_BORDERLINE"
    if "sleep apnea" in pattern:
        return "SLEEP_APNEA_GENERIC"
    if "thyroid" in pattern:
        return "THYROID_CANCER_UNSPECIFIED"
    return None


def _has_required_diagnosis(conditions: List[str], accepted_strings: List[str]) -> bool:
    for cond in conditions:
        cond_lower = normalize(cond)
        for term in accepted_strings:
            if matches_term(cond_lower, term):
                return True
    return False


def _find_qualifying_comorbidity(conditions: List[str]) -> Tuple[str, str]:
    for cond in conditions:
        cond_lower = normalize(cond)
        amb_rule, _ = _find_ambiguity([cond])
        if amb_rule:
            continue

        # Cardiovascular disease
        for phrase in QUALIFYING_CVD_PHRASES:
            if matches_term(cond_lower, phrase):
                return "CVD", cond
        for abbrev in QUALIFYING_CVD_ABBREVS:
            if has_word_boundary(cond_lower, normalize(abbrev)):
                return "CVD", cond

        # Hypertension
        for term in QUALIFYING_HYPERTENSION:
            if matches_term(cond_lower, term):
                return "HYPERTENSION", cond

        # Lipids
        for term in QUALIFYING_LIPIDS:
            if matches_term(cond_lower, term):
                return "LIPIDS", cond

        # T2DM
        for term in QUALIFYING_T2DM:
            if matches_term(cond_lower, term):
                return "DIABETES", cond

        # OSA (must explicitly indicate obstructive)
        if "obstructive" in cond_lower or has_word_boundary(cond_lower, "osa"):
            for term in QUALIFYING_OSA:
                if matches_term(cond_lower, term):
                    return "OSA", cond

    return "NONE", ""


def _safety_code_from_category(category: str) -> Optional[str]:
    cat = category.lower()
    if "medullary thyroid" in cat:
        return "MTC"
    if "multiple endocrine neoplasia" in cat:
        return "MEN2"
    if "pregnant" in cat or "nursing" in cat:
        return "PREGNANCY"
    if "hypersensitivity" in cat:
        return "HYPERSENSITIVITY"
    if "pancreatitis" in cat:
        return "PANCREATITIS"
    if "suicid" in cat:
        return "SUICIDALITY"
    if "motility" in cat or "gastroparesis" in cat:
        return "GI_MOTILITY"
    return None


def _apply_ambiguity_action(rule: dict, evidence: str, bmi: Optional[float]) -> EligibilityResult:
    action = rule["action"]
    notes = rule.get("notes", "")
    reasoning = f"Ambiguous term '{evidence}' matched policy pattern '{rule['pattern']}'. {notes}".strip()
    if action == "REQUEST_INFO":
        verdict = "DENIED_MISSING_INFO"
    elif action == "CONSERVATIVE_DENY":
        verdict = "DENIED_CLINICAL"
    else:
        verdict = "MANUAL_REVIEW"
    return EligibilityResult(
        verdict=verdict,
        bmi_numeric=bmi,
        safety_flag="CLEAR",
        comorbidity_category="NONE",
        evidence_quoted=evidence,
        reasoning=reasoning,
        policy_path="AMBIGUITY_MANUAL_REVIEW",
        decision_type="FLAGGED_AMBIGUITY" if verdict == "MANUAL_REVIEW" else "DENIED_NO_COMORBIDITY",
        safety_exclusion_code=None,
        ambiguity_code=_ambiguity_code(rule),
    )


def evaluate_eligibility(patient_data: dict) -> EligibilityResult:
    bmi_raw = patient_data.get("latest_bmi", "")
    conditions = patient_data.get("conditions", []) or []
    meds = patient_data.get("meds", []) or []

    bmi = _parse_bmi(bmi_raw)
    default_policy_path = "UNKNOWN"
    default_decision_type = "FLAGGED_AMBIGUITY" if bmi is None else "DENIED_NO_COMORBIDITY"
    ambiguity_code = None
    safety_code = None

    # Safety gate
    is_excluded, safety_evidence = _check_safety_exclusions(conditions, meds)
    if is_excluded:
        for exclusion in SAFETY_EXCLUSIONS:
            if exclusion["category"] and exclusion["category"].lower() in safety_evidence:
                safety_code = _safety_code_from_category(exclusion["category"])
        return EligibilityResult(
            verdict="DENIED_SAFETY",
            bmi_numeric=bmi,
            safety_flag="DETECTED",
            comorbidity_category="NONE",
            evidence_quoted=safety_evidence,
            reasoning=f"Safety exclusion detected: '{safety_evidence}'. Wegovy is contraindicated.",
            policy_path="SAFETY_EXCLUSION",
            decision_type="DENIED_HARD_STOP",
            safety_exclusion_code=safety_code,
            ambiguity_code=None,
        )

    # Ambiguous thyroid malignancy needs manual review even without MTC/MEN2
    thyroid_rule, thyroid_ev = _find_ambiguity(conditions, lambda r: "thyroid" in r["pattern"].lower())
    if thyroid_rule:
        result = _apply_ambiguity_action(thyroid_rule, thyroid_ev, bmi)
        result.policy_path = "AMBIGUITY_MANUAL_REVIEW"
        result.decision_type = "FLAGGED_AMBIGUITY"
        result.ambiguity_code = _ambiguity_code(thyroid_rule)
        return result

    # BMI presence
    if bmi is None:
        return EligibilityResult(
            verdict="DENIED_MISSING_INFO",
            bmi_numeric=None,
            safety_flag="CLEAR",
            comorbidity_category="NONE",
            evidence_quoted="",
            reasoning="BMI could not be determined. Document a current or baseline BMI.",
            policy_path="UNKNOWN",
            decision_type="DENIED_MISSING_INFO",
            safety_exclusion_code=None,
            ambiguity_code=None,
        )

    # BMI ≥ 30 pathway (requires adult obesity diagnosis string)
    if bmi >= BMI_OBESE_THRESHOLD:
        if _has_required_diagnosis(conditions, ADULT_OBESITY_DIAGNOSES):
            return EligibilityResult(
                verdict="APPROVED",
                bmi_numeric=bmi,
                safety_flag="CLEAR",
                comorbidity_category="NONE",
                evidence_quoted="",
                reasoning=f"BMI {bmi} meets obesity threshold (≥{BMI_OBESE_THRESHOLD}) with qualifying adult obesity diagnosis.",
                policy_path="BMI30_OBESITY",
                decision_type="APPROVED",
                safety_exclusion_code=None,
                ambiguity_code=None,
            )
        return EligibilityResult(
            verdict="DENIED_MISSING_INFO",
            bmi_numeric=bmi,
            safety_flag="CLEAR",
            comorbidity_category="NONE",
            evidence_quoted="",
            reasoning="BMI ≥ 30 requires a documented adult obesity diagnosis string.",
            policy_path="BMI30_OBESITY",
            decision_type="DENIED_NO_COMORBIDITY",
            safety_exclusion_code=None,
            ambiguity_code=None,
        )

    # Below minimum
    if bmi < BMI_OVERWEIGHT_THRESHOLD:
        return EligibilityResult(
            verdict="DENIED_CLINICAL",
            bmi_numeric=bmi,
            safety_flag="CLEAR",
            comorbidity_category="NONE",
            evidence_quoted="",
            reasoning=f"BMI {bmi} is below minimum threshold ({BMI_OVERWEIGHT_THRESHOLD}). Does not meet coverage criteria.",
            policy_path="BELOW_THRESHOLD",
            decision_type="DENIED_BMI_THRESHOLD",
            safety_exclusion_code=None,
            ambiguity_code=None,
        )

    # BMI 27–29.9 pathway
    # CRITICAL FIX: Check for qualifying comorbidities FIRST, before ambiguity checks.
    # If a patient has a valid qualifying comorbidity (e.g., Essential Hypertension),
    # they should be APPROVED even if they also have ambiguous terms (e.g., Prediabetes).
    
    # Step 1: Check for qualifying comorbidity first (highest priority after safety)
    category, comorbidity_evidence = _find_qualifying_comorbidity(conditions)
    
    # If we have a qualifying comorbidity, check for overweight diagnosis
    if category != "NONE":
        if _has_required_diagnosis(conditions, ADULT_OVERWEIGHT_DIAGNOSES):
            # Has both qualifying comorbidity AND overweight diagnosis = APPROVED
            return EligibilityResult(
                verdict="APPROVED",
                bmi_numeric=bmi,
                safety_flag="CLEAR",
                comorbidity_category=category,
                evidence_quoted=comorbidity_evidence,
                reasoning=f"BMI {bmi} (overweight) with qualifying comorbidity '{comorbidity_evidence}' ({category}). Approved.",
                policy_path="BMI27_COMORBIDITY",
                decision_type="APPROVED",
                safety_exclusion_code=None,
                ambiguity_code=None,
            )
        else:
            # Has qualifying comorbidity but missing overweight diagnosis string
            return EligibilityResult(
                verdict="DENIED_MISSING_INFO",
                bmi_numeric=bmi,
                safety_flag="CLEAR",
                comorbidity_category=category,
                evidence_quoted=comorbidity_evidence,
                reasoning=(
                    f"DOCUMENTATION REQUIRED: Patient has qualifying comorbidity '{comorbidity_evidence}' ({category}) "
                    f"and BMI {bmi} meets the overweight threshold. However, an explicit 'Overweight' or "
                    "'excess weight' diagnosis must be documented in the chart to complete policy criteria. "
                    "Please add an overweight diagnosis (e.g., 'Overweight', 'Overweight, adult', 'E66.3') to the problem list."
                ),
                policy_path="BMI27_COMORBIDITY",
                decision_type="DENIED_MISSING_INFO",
                safety_exclusion_code=None,
                ambiguity_code=None,
            )
    
    # Step 2: No qualifying comorbidity found - check for ambiguous terms
    # (only after confirming no valid comorbidities exist)
    ambiguity_rule, ambiguity_evidence = _find_ambiguity(conditions)
    if ambiguity_rule:
        result = _apply_ambiguity_action(ambiguity_rule, ambiguity_evidence, bmi)
        result.policy_path = "AMBIGUITY_MANUAL_REVIEW"
        result.decision_type = "FLAGGED_AMBIGUITY"
        result.ambiguity_code = _ambiguity_code(ambiguity_rule)
        return result

    # Step 3: No qualifying comorbidity AND no ambiguous terms = denial
    return EligibilityResult(
        verdict="DENIED_CLINICAL",
        bmi_numeric=bmi,
        safety_flag="CLEAR",
        comorbidity_category="NONE",
        evidence_quoted="",
        reasoning=f"BMI {bmi} (overweight) but no qualifying weight-related comorbidity documented. Does not meet criteria.",
        policy_path="BMI27_COMORBIDITY",
        decision_type="DENIED_NO_COMORBIDITY",
        safety_exclusion_code=None,
        ambiguity_code=None,
    )


def evaluate_from_patient_record(patient_data: dict) -> dict:
    """Wrapper that returns a dictionary for compatibility with agent_logic.py."""
    result = evaluate_eligibility(patient_data)
    return result.to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = {"latest_bmi": "32.4", "conditions": ["Hypertension"], "meds": []}
    print(evaluate_eligibility(sample))

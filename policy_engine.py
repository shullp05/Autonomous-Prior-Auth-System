# policy_engine.py
"""
Deterministic Policy Engine for RX-WEG-2025 (Wegovy Chronic Weight Management).

All semantics are sourced from the canonical snapshot (policies/RX-WEG-2025.json).
The LLM is only used downstream to narrate reasoning; eligibility decisions here
are purely rule-based.

Key alignment decisions (with chaos_monkey + pipeline behavior):
- BMI >= 30 is sufficient for clinical eligibility (no obesity diagnosis-string dependency).
  Reason: synthetic generation + ambiguity handling treats generic "Obesity" as non-authoritative,
  and payers commonly accept BMI documentation itself as objective evidence.
- For BMI 27–29.9, qualifying comorbidities take precedence over ambiguous terms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from policy_utils import (
    normalize,
    has_word_boundary,
    matches_term,
    expand_safety_variants,
    is_snf_phrase,
)
from policy_constants import (
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

AMBIGUITY_RULES = _SNAPSHOT.get("ambiguities", []) or []


from context_rules import classify_context
from context_rules import classify_context
from rules.coding_integrity_rules import check_admin_readiness
from audit_logger import get_audit_logger

_audit_logger = get_audit_logger()

@dataclass
class EligibilityResult:
    """Structured result from deterministic eligibility evaluation."""

    verdict: str  # APPROVED, DENIED_SAFETY, DENIED_CLINICAL, DENIED_MISSING_INFO, MANUAL_REVIEW, SAFETY_SIGNAL_NEEDS_REVIEW, CDI_REQUIRED
    bmi_numeric: Optional[float]
    safety_flag: str  # CLEAR or DETECTED (or SIGNAL)
    comorbidity_category: str  # NONE, HYPERTENSION, LIPIDS, DIABETES, OSA, CVD
    evidence_quoted: str
    reasoning: str
    policy_path: str  # BMI30_OBESITY, BMI27_COMORBIDITY, BELOW_THRESHOLD, SAFETY_EXCLUSION, AMBIGUITY_MANUAL_REVIEW, UNKNOWN
    decision_type: str  # APPROVED, DENIED_HARD_STOP, DENIED_BMI_THRESHOLD, DENIED_NO_COMORBIDITY, DENIED_MISSING_INFO, FLAGGED_AMBIGUITY, CDI_REQUIRED
    safety_exclusion_code: Optional[str] = None  # MTC, MEN2, PREGNANCY, BREASTFEEDING, HYPERSENSITIVITY, CONCURRENT_GLP1, PANCREATITIS, SUICIDALITY, GI_MOTILITY
    ambiguity_code: Optional[str] = None
    # Phase 9.3: Explicit Safety Evidence
    safety_context: Optional[str] = None  # ACTIVE, HISTORICAL, NEGATED, FAMILY_HISTORY, HYPOTHETICAL
    safety_confidence: Optional[str] = None  # HARD_STOP, SIGNAL
    # Phase 9.5: Coding Integrity
    clinical_eligible: bool = False
    admin_ready: bool = False
    missing_anchor_code: Optional[str] = None
    physician_query_text: Optional[str] = None

    def to_dict(self) -> dict:
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
            "safety_context": self.safety_context,
            "safety_confidence": self.safety_confidence,
            "clinical_eligible": self.clinical_eligible,
            "admin_ready": self.admin_ready,
            "missing_anchor_code": self.missing_anchor_code,
            "physician_query_text": self.physician_query_text,
        }


def _as_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]

def _as_str_list(val: Any) -> List[str]:
    out: List[str] = []
    for x in _as_list(val):
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out

def _parse_bmi(bmi_raw: Any) -> Optional[float]:
    """Parse BMI from various formats, handling '(Calculated)' suffix."""
    if bmi_raw is None:
        return None
    s = str(bmi_raw).lower().strip()
    # Remove suffix if present
    if "(" in s:
        s = s.split("(")[0].strip()
    try:
        val = float(s)
        if BMI_MIN_REASONABLE < val < BMI_MAX_REASONABLE:
            return val
    except ValueError:
        pass
    return None

def _safety_code_from_category_or_term(category: str, term: str) -> str:
    """Map safety category or term to a specific exclusion code."""
    cat = category.lower()
    t = term.lower()
    
    if "medullary thyroid" in cat or "mtc" in t:
        return "MTC_HISTORY"
    if "multiple endocrine" in cat or "men2" in t:
        return "MEN2_SYNDROME"
    if "pregnan" in cat or "pregnan" in t:
        return "PREGNANCY"
    if "breastfeed" in cat or "lactat" in cat or "nursing" in t:
        return "BREASTFEEDING"
    if "hypersens" in cat or "allerg" in cat:
        return "HYPERSENSITIVITY"
    if "pancreatitis" in cat:
        return "PANCREATITIS_HISTORY"
    if "suicid" in cat:
        return "SUICIDALITY_HISTORY"
    if "glp-1" in cat or "concurrent" in cat:
        return "CONCURRENT_GLP1"
    if "motility" in cat or "gastroparesis" in t:
        return "GI_MOTILITY"
        
    return "SAFETY_EXCLUSION_GENERIC"

def _check_safety_exclusions(conditions: List[str], meds: List[str]) -> Tuple[bool, str, Optional[str], Optional[str], Optional[str]]:
    """
    Check for safety exclusions.
    Returns: (is_hard_stop, evidence_quoted, exclusion_code, safety_context, safety_confidence)
    """
    # 1. Prohibited Meds (Concurrent GLP-1)
    for med in meds:
        m_norm = normalize(med)
        for prohibited in PROHIBITED_GLP1:
            if matches_term(m_norm, prohibited):
                return True, med, "CONCURRENT_GLP1", "ACTIVE", "HARD_STOP"

    # 2. Safety Conditions
    for entry in SAFETY_EXCLUSIONS:
        category = str(entry.get("category", ""))
        is_pregnancy_related = "breastfeeding" in category.lower() or "pregnancy" in category.lower()

        # Gather variants
        search_terms = []
        for term in _as_str_list(entry.get("accepted_strings")):
            search_terms.extend(expand_safety_variants(term))
            
        for term in search_terms:
            for cond in conditions:
                # Special skip for "nursing" in SNF context
                if is_pregnancy_related and "nursing" in term.lower() and is_snf_phrase(cond):
                    continue
                    
                if matches_term(cond, term):
                    # Found a textual match. Now classify context (Phase 9.3)
                    classification = classify_context(cond)
                    
                    code = _safety_code_from_category_or_term(category, term)
                    
                    if classification.confidence == "HARD_STOP":
                        return True, cond, code, classification.context_type, classification.confidence
                    elif classification.confidence == "SIGNAL":
                        # We found a match but it's not a Hard Stop (e.g. Historical)
                        # We return it as a signal, but continue checking for other Hard Stops?
                        # ACTUALLY: Hard Stop takes precedence. 
                        # But if we find a Signal, we should store it and keep looking for Hard Stops.
                        # For simplicity in this engine: checking order matters. 
                        # We'll return the first match, but if it's a Signal, we should ideally verify if a Hard Stop exists later.
                        # Optimization: Iterate all, prioritize proper Hard Stop.
                        pass # Continue loop to find a Hard Stop if possible
                    
    # Second pass: If no Hard Stop found, check for Signals
    for entry in SAFETY_EXCLUSIONS:
        category = str(entry.get("category", ""))
        is_pregnancy_related = "breastfeeding" in category.lower() or "pregnancy" in category.lower()
        search_terms = []
        for term in _as_str_list(entry.get("accepted_strings")):
            search_terms.extend(expand_safety_variants(term))
            
        for term in search_terms:
            for cond in conditions:
                if is_pregnancy_related and "nursing" in term.lower() and is_snf_phrase(cond):
                    continue
                if matches_term(cond, term):
                    classification = classify_context(cond)
                    code = _safety_code_from_category_or_term(category, term)
                    if classification.confidence == "SIGNAL":
                         return False, cond, code, classification.context_type, classification.confidence

    return False, "", None, None, None

def _find_ambiguity(conditions: List[str], filter_fn=None) -> Tuple[Optional[dict], str]:
    """Find if any condition matches an ambiguity rule."""
    for rule in AMBIGUITY_RULES:
        if filter_fn and not filter_fn(rule):
            continue
        
        pattern = rule.get("pattern")
        if not pattern:
            continue
            
        for cond in conditions:
            if matches_term(cond, pattern):
                # Context check is implicit in manual review requirement often, 
                # but we could skip negated ones. 
                # For ambiguity, usually even negated/historical fuzzy matches trigger review 
                # unless explicitly cleared.
                # Let's trust matches_term/normalize for now.
                return rule, cond
    return None, ""

def _ambiguity_code(rule: dict) -> str:
    return rule.get("id", "AMBIGUITY_GENERIC")

def _apply_ambiguity_action(rule: dict, evidence: str, bmi: float) -> EligibilityResult:
    """Construct result for an ambiguity trigger."""
    # Most actions are manual_review
    action = rule.get("action", "manual_review")
    reason = rule.get("reasoning_template", "Ambiguous clinical term detected.")
    
    return EligibilityResult(
        verdict="MANUAL_REVIEW",
        bmi_numeric=bmi,
        safety_flag="CLEAR", # Not a safety issue per se
        comorbidity_category="NONE",
        evidence_quoted=evidence,
        reasoning=f"{reason} (Term: '{evidence}')",
        policy_path="AMBIGUITY_MANUAL_REVIEW",
        decision_type="FLAGGED_AMBIGUITY",
        ambiguity_code=_ambiguity_code(rule)
    )

def _find_qualifying_comorbidity(conditions: List[str]) -> Tuple[str, str]:
    """
    Check if any condition meets the strict comorbidity criteria for BMI 27+.
    Returns (category_name, evidence_string).
    """
    # 1. Hypertension
    for cond in conditions:
        for term in QUALIFYING_HYPERTENSION:
            if matches_term(cond, term) and classify_context(cond).context_type == "ACTIVE":
                return "HYPERTENSION", cond
                
    # 2. Type 2 Diabetes
    for cond in conditions:
        for term in QUALIFYING_T2DM:
            if matches_term(cond, term) and classify_context(cond).context_type == "ACTIVE":
                return "TYPE2_DIABETES", cond

    # 3. Dyslipidemia
    for cond in conditions:
        for term in QUALIFYING_LIPIDS:
            if matches_term(cond, term) and classify_context(cond).context_type == "ACTIVE":
                return "DYSLIPIDEMIA", cond

    # 4. OSA
    for cond in conditions:
        for term in QUALIFYING_OSA:
            if matches_term(cond, term) and classify_context(cond).context_type == "ACTIVE":
                return "OSA", cond

    # 5. CVD
    for cond in conditions:
        # Check phrases
        for term in QUALIFYING_CVD_PHRASES:
             if matches_term(cond, term) and classify_context(cond).context_type == "ACTIVE":
                return "CARDIOVASCULAR_DISEASE", cond
        # Check abbreviations (strict word boundary already handled by matches_term logic for single tokens)
        for term in QUALIFYING_CVD_ABBREVS:
             if matches_term(cond, term) and classify_context(cond).context_type == "ACTIVE":
                return "CARDIOVASCULAR_DISEASE", cond

    return "NONE", ""




def evaluate_eligibility(patient_data: dict) -> EligibilityResult:
    """
    Core deterministic decision function.
    Expects patient_data keys:
      - latest_bmi: str/float (may include "(Calculated)")
      - conditions: List[str]
      - meds: List[str]
    """
    bmi_raw = patient_data.get("latest_bmi", "")
    conditions = patient_data.get("conditions", []) or []
    meds = patient_data.get("meds", []) or []

    bmi = _parse_bmi(bmi_raw)

    # 1) Safety gate
    is_hard_stop, safety_evidence, safety_code, safety_context, safety_confidence = _check_safety_exclusions(conditions, meds)
    
    if is_hard_stop:
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
            safety_context=safety_context,
            safety_confidence=safety_confidence
        )
        
    # Phase 9.1: If not Hard Stop but we found a Safety Signal (e.g. Historical), return Signal verdict.
    if safety_code is not None:
         return EligibilityResult(
            verdict="SAFETY_SIGNAL_NEEDS_REVIEW",
            bmi_numeric=bmi,
            safety_flag="DETECTED", # Flag remains DETECTED so UI lights up, but verdict triggers specific handling
            comorbidity_category="NONE",
            evidence_quoted=safety_evidence,
            reasoning=f"Safety signal detected: '{safety_evidence}' ({safety_context}). Manual review required to confirm contraindication.",
            policy_path="SAFETY_EXCLUSION",
            decision_type="FLAGGED_SAFETY_WARNING", # New decision type
            safety_exclusion_code=safety_code,
            ambiguity_code=None,
            safety_context=safety_context,
            safety_confidence=safety_confidence
        )

    # 2) Ambiguous thyroid malignancy needs manual review even without explicit MTC/MEN2
    thyroid_rule, thyroid_ev = _find_ambiguity(conditions, lambda r: "thyroid" in str(r.get("pattern", "")).lower())
    if thyroid_rule:
        result = _apply_ambiguity_action(thyroid_rule, thyroid_ev, bmi)
        result.policy_path = "AMBIGUITY_MANUAL_REVIEW"
        result.decision_type = "FLAGGED_AMBIGUITY"
        result.ambiguity_code = _ambiguity_code(thyroid_rule)
        return result

    # 3) BMI presence
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

    # 4) Clinical Decision Logic
    clinical_verdict = None
    policy_path = None
    comorbidity_category = "NONE"
    evidence_quoted = ""
    pathway_name = None # For Coding Integrity Check

    # 4a) BMI ≥ 30 pathway
    if bmi >= BMI_OBESE_THRESHOLD:
        pathway_name = "BMI30_OBESITY"
        clinical_verdict = "APPROVED"
        reasoning_base = f"BMI {bmi} meets obesity threshold (≥{BMI_OBESE_THRESHOLD})."
        policy_path = "BMI30_OBESITY"
        
    # 4b) Below minimum threshold
    elif bmi < BMI_OVERWEIGHT_THRESHOLD:
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
        
    # 4c) BMI 27–29.9 pathway
    else:
        # Critical ordering: comorbidity check BEFORE ambiguity check.
        category, comorbidity_ev = _find_qualifying_comorbidity(conditions)
        if category != "NONE":
            pathway_name = "BMI27_COMORBIDITY"
            clinical_verdict = "APPROVED"
            comorbidity_category = category
            evidence_quoted = comorbidity_ev
            reasoning_base = f"BMI {bmi} (overweight range) with qualifying comorbidity '{comorbidity_ev}' ({category})."
            policy_path = "BMI27_COMORBIDITY"
        else:
            # Only if no qualifying comorbidity exists: check ambiguities
            ambiguity_rule, ambiguity_evidence = _find_ambiguity(conditions)
            if ambiguity_rule:
                result = _apply_ambiguity_action(ambiguity_rule, ambiguity_evidence, bmi)
                result.policy_path = "AMBIGUITY_MANUAL_REVIEW"
                result.decision_type = "FLAGGED_AMBIGUITY" if result.verdict == "MANUAL_REVIEW" else result.decision_type
                result.ambiguity_code = _ambiguity_code(ambiguity_rule)
                return result

            # No comorbidity + no ambiguity => denial
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

    # 5) Phase 9.5: Coding Integrity Overlay
    # If we reached here, patient is Clinically Eligible (clinical_verdict == "APPROVED")
    
    is_admin_ready, missing_code = check_admin_readiness(conditions, pathway_name)
    
    if is_admin_ready:
        return EligibilityResult(
            verdict="APPROVED",
            bmi_numeric=bmi,
            safety_flag="CLEAR",
            comorbidity_category=comorbidity_category,
            evidence_quoted=evidence_quoted,
            reasoning=f"{reasoning_base} Approved.",
            policy_path=policy_path,
            decision_type="APPROVED",
            safety_exclusion_code=None,
            ambiguity_code=None,
            clinical_eligible=True,
            admin_ready=True
        )
    else:
        # CDI REQUIRED
        # Construct Physician Query Text
        if pathway_name == "BMI30_OBESITY":
             rec_codes = "E66.9 (Obesity, unspecified) or specific E66.x code"
        else:
             rec_codes = "E66.3 (Overweight)"
             
        query_text = (
            f"Attributes Summary: Clinical criteria met (BMI {bmi}).\n"
            f"Missing Documentation: Payer requires specific ICD-10 anchor code for weight diagnosis.\n"
            f"Recommended Action: Add diagnosis code {rec_codes} to problem list."
        )
             
        result = EligibilityResult(
            verdict="CDI_REQUIRED",
            bmi_numeric=bmi,
            safety_flag="CLEAR", # Safety is clear, administratively blocked
            comorbidity_category=comorbidity_category,
            evidence_quoted=evidence_quoted,
            reasoning=f"{reasoning_base} Clinical criteria met, but missing required administrative anchor code ({missing_code}).",
            policy_path=policy_path, # Keep original clinical path
            decision_type="CDI_REQUIRED",
            safety_exclusion_code=None,
            ambiguity_code=None,
            clinical_eligible=True,
            admin_ready=False,
            missing_anchor_code=missing_code,
            physician_query_text=query_text
        )
    
    # Audit Log the decision (centralized logging)
    _audit_logger.log_event(
        event_type="DECISION",
        actor="system",
        patient_id=patient_data.get("patient_id"), 
        details={
            "inputs": {
                "bmi": str(bmi),
                "conditions_count": len(conditions),
                "meds_count": len(meds)
            },
            "output": {
                "verdict": result.verdict,
                "reasoning": result.reasoning,
                "decision_type": result.decision_type
            }
        }
    )
    
    return result


def evaluate_from_patient_record(patient_data: dict) -> dict:
    """Wrapper that returns a dictionary for compatibility with agent_logic.py."""
    return evaluate_eligibility(patient_data).to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = {"latest_bmi": "32.4", "conditions": ["Hypertension"], "meds": []}
    print(evaluate_eligibility(sample))

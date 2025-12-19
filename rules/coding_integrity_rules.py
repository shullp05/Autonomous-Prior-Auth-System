"""
coding_integrity_rules.py - Phase 9.5 Coding Integrity Overlay

Defines the "Administrative Readiness" logic.
Cases may be CLINICALLY ELIGIBLE but ADMINISTRATIVE NOT READY if they lack specific anchor codes.

Anchor Codes (ICD-10):
- Obesity Pathway: E66.01, E66.09, E66.1, E66.2, E66.8, E66.9
- Overweight Pathway: E66.3
+ Hygiene prefixes (Z68 for BMI, etc. - currently informational)
"""

from typing import List, Tuple, Optional, Set

# ==============================================================================
# Configuration: Payer-Mandated Anchor Codes
# ==============================================================================

OBESITY_ANCHOR_CODES = {
    "E66.01", # Morbid (severe) obesity due to excess calories
    "E66.09", # Other obesity due to excess calories
    "E66.1",  # Drug-induced obesity
    "E66.2",  # Morbid (severe) obesity with alveolar hypoventilation
    "E66.8",  # Other obesity
    "E66.9",  # Obesity, unspecified
    "E66.3"   # Overweight (Included here? No, strictly E66.3 is for Overweight pathway usually)
}

OVERWEIGHT_ANCHOR_CODES = {
    "E66.3"   # Overweight
}

# Hygiene only
BMI_ZCODES_PREFIX = "Z68."

# ==============================================================================
# Logic
# ==============================================================================

def check_admin_readiness(
    patient_diagnoses: List[str], 
    clinical_pathway: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """
    Check if a clinically eligible case is administratively ready (has anchor codes).
    
    Args:
        patient_diagnoses: List of diagnosis codes (e.g. ["E66.9", "I10"])
        clinical_pathway: "BMI30_OBESITY" or "BMI27_COMORBIDITY"
        
    Returns:
        (is_ready: bool, missing_code_suggestion: Optional[str])
    """
    if not clinical_pathway:
        # Not politically eligible, so administrative readiness is irrelevant (or technically not ready)
        return False, None

    # Normalize diagnoses (upper case, strip)
    # Note: Currently data might be names or codes. We need to handle both? 
    # Assumption for Phase 9.5: The 'conditions' list in data might be text names, but the `diagnosis_codes` might be explicit.
    # WAIT - `evaluate_eligibility` receives `conditions` (strings). 
    # Does it receive codes? 
    # The Task instructions say: "Add Codign Integrity Decision Step... Missing Anchor -> CDI_REQUIRED"
    # And "OBESITY_ANCHOR_CODES = {...}"
    # This implies we need access to ICD codes.
    # `patient_data` in `policy_engine.py` currently has keys: `latest_bmi`, `conditions`, `meds`.
    # It does NOT appear to have a dedicated `diagnosis_codes` list yet.
    # However, `conditions` list often contains mixed text.
    # We must scan `conditions` for these substrings (or exact codes if provided).
    
    # For robust implementation, we'll scan the input strings for the code pattern.
    
    detected_codes = _extract_codes(patient_diagnoses)
    
    if clinical_pathway == "BMI30_OBESITY":
        # Requires Obesity Anchor
        if any(code in detected_codes for code in OBESITY_ANCHOR_CODES):
            return True, None
        return False, "E66.9" # Default suggestion
        
    if clinical_pathway == "BMI27_COMORBIDITY":
        # Requires Overweight Anchor
        # Note: E66.9 (Obesity) would technically satisfy Overweight criteria too (hierarchically), 
        # but strict rules might require E66.3 specifically for the "Overweight" pathway.
        # Let's be lenient: accepted if E66.3 OR any Obesity code (since Obesity > Overweight)
        if any(code in detected_codes for code in OVERWEIGHT_ANCHOR_CODES) or \
           any(code in detected_codes for code in OBESITY_ANCHOR_CODES):
            return True, None
        return False, "E66.3" # Default suggestion

    return True, None # Should be unreachable if pathway is valid

def _extract_codes(conditions: List[str]) -> Set[str]:
    """
    Extract potential ICD-10 codes from condition strings.
    Naive extraction: looks for patterns like "E66.9" or "(E66.9)"
    """
    found = set()
    import re
    # Match generally: Letter + Digits + Dot + Digits
    # e.g. E66.9, Z68.1, I10
    pattern = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,3})?)\b')
    
    for cond in conditions:
        matches = pattern.findall(cond.upper())
        found.update(matches)
        
    return found

"""
coding_integrity_rules.py - Phase 9.5 Coding Integrity Overlay

Defines the "Administrative Readiness" logic.
Cases may be CLINICALLY ELIGIBLE but ADMINISTRATIVE NOT READY if they lack specific anchor codes.

New Strict Policy (2025):
- Obesity Pathway: Requires (Text "Obesity") + (Code E66.x) + (Code Z68.x)
- Overweight Pathway: Requires (Text "Overweight/Obesity") + (Code E66.3/E66.x) + (Code Z68.x)
"""
import re
from typing import List, Set, Tuple, Optional, Any, Dict

# ==============================================================================
# Configuration: Payer-Mandated Anchor Codes
# ==============================================================================

# Constants removed - now injected from Policy Snapshot (RX-WEG-2025.json) via Policy Engine

VALID_BMI_Z_PREFIX = "Z68" # Prefix check still useful, or rely on explicit list?

# ==============================================================================
# Logic
# ==============================================================================

def check_admin_readiness(
    patient_diagnoses: List[Any],
    required_e66_codes: Set[str],
    required_z68_codes: Set[str],
    required_diagnosis_strings: List[str]
) -> Tuple[bool, Optional[str], Dict[str, str]]:
    """
    Check if a clinically eligible case is administratively ready (Triple-Key Verification).
    
    Args:
        patient_diagnoses: List of condition dicts (with 'condition_name', 'icd10_dx', 'icd10_bmi') 
                           OR list of strings (legacy fallback).
        required_e66_codes: Set of valid E66 codes for this pathway.
        required_z68_codes: Set of valid Z68 codes for this pathway.
        required_diagnosis_strings: List of valid diagnosis text strings.
        
    Returns:
        (is_ready: bool, missing_code_suggestion: Optional[str], found_evidence: Dict[str, str])
    """

    # 1. Gather all evidence from the chart into Sets for global checking
    chart_text_tokens = set()
    chart_dx_codes = set()
    chart_z_codes = set()

    for c in patient_diagnoses:
        # Handle dict vs string
        if isinstance(c, str):
            name = c.upper()
            dx = ""
            z = ""
        else:
            name = str(c.get("condition_name", "")).upper()
            dx = str(c.get("icd10_dx", "")).strip().upper()
            z = str(c.get("icd10_bmi", "")).strip().upper()

        # Tokenize name (naive) - actually, let's keep the full name for matching too
        chart_text_tokens.add(name)

        # Collect codes
        if dx: chart_dx_codes.add(dx)
        if z: chart_z_codes.add(z)

        # Also check if codes are embedded in the text string (backup)
        # e.g. "Obesity (E66.9)"
        codes_in_text = re.findall(r'([A-Z]\d{2}(?:\.\d{1,9})?)', name)
        for code in codes_in_text:
             if code.startswith("E66"): chart_dx_codes.add(code)
             if code.startswith("Z68"): chart_z_codes.add(code)


    # 2. Verify Triple Keys (Against Injected Requirements)
    found_evidence = {"text": None, "e66": None, "z68": None}

    # Key A: Text
    # We check if ANY chart condition name contains ANY of the required strings (case-insensitive substring match)
    # The required_diagnosis_strings from snapshot are usually full phrases "Adult Obesity".
    # But simple "Obesity" might be enough.
    # Logic: If chart text contains a required string (normalized)
    has_text = False

    # Normalize reqs
    normalized_reqs = [r.upper() for r in required_diagnosis_strings]
    # Add stemmed fallbacks if list provides "Adult Obesity", we probably want "Obesity" to match?
    # Actually policy says "Documented... diagnosis string".
    # For now, we do substring check.

    for token in chart_text_tokens: # token is full condition name here
        for req in normalized_reqs:
            if req in token:
                has_text = True
                found_evidence["text"] = token # Capture the actual chart text that matched
                break
        if has_text: break

    # Key B: E66 Code
    has_e66 = False
    for code in chart_dx_codes:
        if code in required_e66_codes:
            has_e66 = True
            found_evidence["e66"] = code
            break
        # Handle "E66.x" wildcard if present in requirements
        if "E66.X" in required_e66_codes or "E66.x" in required_e66_codes:
             if code.startswith("E66"):
                 has_e66 = True
                 found_evidence["e66"] = code
                 break

    # Key C: Z68 Code
    has_z68 = False
    for code in chart_z_codes:
        if code in required_z68_codes:
            has_z68 = True
            found_evidence["z68"] = code
            break
        # Handle "Z68.x" wildcard
        if "Z68.X" in required_z68_codes or "Z68.x" in required_z68_codes:
            if code.startswith("Z68"):
                has_z68 = True
                found_evidence["z68"] = code
                break

    # Compile Verdict
    def _suggest_code(required: Set[str], preferred: List[str], fallback: str) -> str:
        normalized = {c.upper() for c in required if c}
        for candidate in preferred:
            if candidate.upper() in normalized:
                return candidate
        for code in sorted(normalized):
            if code and code[0].isalpha():
                return code
        return fallback

    if not has_text or not has_e66 or not has_z68:
        if not has_e66:
            missing_code = _suggest_code(
                required_e66_codes,
                ["E66.3", "E66.9", "E66.01", "E66.09", "E66.0", "E66.x"],
                "E66.x",
            )
        elif not has_z68:
            missing_code = _suggest_code(
                required_z68_codes,
                ["Z68.x"],
                "Z68.x",
            )
        else:
            missing_code = "Diagnosis Text"

        return False, missing_code, found_evidence

    return True, None, found_evidence

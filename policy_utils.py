# policy_utils.py

from __future__ import annotations

import math
import re
from typing import Any, Optional


def normalize(text: Any) -> str:
    """
    Normalize any incoming value to a safe lowercase string.

    CRITICAL FIX:
    - Avoid `text or ""` because pandas.NA raises on boolean evaluation.
    - Treat None and NaN as empty.
    """
    if text is None:
        return ""
    # Handle float NaN safely
    if isinstance(text, float) and math.isnan(text):
        return ""
    s = str(text)
    # Common pandas-ish missing markers
    if s.strip().lower() in {"nan", "<na>", "none"}:
        return ""
    return s.strip().lower()


def has_word_boundary(haystack: str, needle: str) -> bool:
    """
    True if `needle` appears in `haystack` with word boundaries.
    """
    hay = normalize(haystack)
    ndl = normalize(needle)
    if not hay or not ndl:
        return False
    pattern = rf"\b{re.escape(ndl)}\b"
    return bool(re.search(pattern, hay, re.IGNORECASE))


def matches_term(text: str, term: str) -> bool:
    """
    Policy-safe match for comorbidity/safety terms.

    Improvements:
    - For single tokens: strict word boundary match.
    - For multi-word phrases: allow flexible separators between words (space, hyphen, slash, punctuation),
      while still enforcing boundaries at the ends.
      Example: "type 2 diabetes" matches "type-2 diabetes" and "type 2 diabetes".
    """
    t = normalize(text)
    q = normalize(term)
    if not t or not q:
        return False

    # Single token => strict boundary
    if " " not in q and "/" not in q and "-" not in q:
        return has_word_boundary(t, q)

    # Multi-token => allow non-word separators between tokens
    tokens = [tok for tok in re.split(r"[\s/]+", q) if tok]
    if not tokens:
        return False

    # Join tokens with flexible separators (one or more non-word chars or underscores)
    inner = r"(?:[\W_]+)".join(re.escape(tok) for tok in tokens)
    pattern = rf"\b{inner}\b"
    return bool(re.search(pattern, t, re.IGNORECASE))


def expand_safety_variants(term: str) -> list[str]:
    """
    Expand complex safety strings into individual checkable variants.
    Examples handled:
      - "X/Y"
      - "X or Y"
      - "Something (ABBR)"
      - "personal or family history of X"
    """
    base = normalize(term)
    if not base:
        return []

    variants = {base}

    # Split on common separators
    if "/" in base:
        variants.update(part.strip() for part in base.split("/") if part.strip())
    if " or " in base:
        variants.update(part.strip() for part in base.split(" or ") if part.strip())
    if "," in base:
        variants.update(part.strip() for part in base.split(",") if part.strip())

    # Remove "syndrome" suffix variant
    if base.endswith(" syndrome"):
        variants.add(base.replace(" syndrome", "").strip())

    # Parenthetical abbreviation extraction
    if "(" in base and ")" in base:
        inside = base[base.find("(") + 1 : base.find(")")]
        inside = inside.strip()
        if len(inside) > 1:
            variants.add(inside)
        variants.add(base[: base.find("(")].strip())

    # History prefix normalization
    prefix = "personal or family history of "
    if base.startswith(prefix):
        variants.add(base.split(prefix, 1)[1].strip())

    # Final cleanup
    out = [v for v in (normalize(v) for v in variants) if v]
    # Deduplicate while preserving order
    seen = set()
    ordered: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def is_snf_phrase(text: str) -> bool:
    """
    Detect skilled nursing facility phrases that should NOT trigger pregnancy/nursing exclusions.
    """
    t = normalize(text)
    if not t:
        return False

    snf_terms = [
        "skilled nursing facility",
        "nursing facility",
        "long-term nursing facility",
        "ltc facility",
        "snf",
        "ltc",
    ]

    # For abbreviations, use boundary; for phrases, substring is fine
    for term in snf_terms:
        if len(term) <= 4:  # snf/ltc
            if has_word_boundary(t, term):
                return True
        else:
            if term in t:
                return True
    return False


def format_criteria_list(
    bmi_val: Optional[float],
    found_text: Optional[str],
    found_e66: Optional[str],
    found_z68: Optional[str],
    found_comorb: Optional[str],
    missing_anchor: Optional[str],
    ambiguity_code: Optional[str]
) -> str:
    """
    Generate the 5-point criteria list for letters, highlighting actions needed.
    """
    lines = ["Medical Necessity & Coverage Criteria:"]

    # 1. BMI
    if bmi_val is not None:
        lines.append(f"- Documented BMI Value: {bmi_val}")
    else:
        lines.append("- Documented BMI Value: MISSING [Action: Document current BMI]")

    # 2. Diagnosis String
    if found_text:
        lines.append(f"- Diagnosis String used: {found_text}")
    elif missing_anchor and "Diagnosis Text" in missing_anchor:
        lines.append("- Diagnosis String used: MISSING [Action: Document 'Obesity' or 'Overweight']")
    else:
        lines.append(f"- Diagnosis String used: {found_text if found_text else 'N/A'}")

    # 3. E66 Code
    if found_e66:
        lines.append(f"- ICD-10 E66 Code found: {found_e66}")
    elif missing_anchor and "E66" in missing_anchor:
        lines.append("- ICD-10 E66 Code found: MISSING [Action: Add E66 code]")
    else:
        lines.append(f"- ICD-10 E66 Code found: {found_e66 if found_e66 else 'N/A'}")

    # 4. Z68 Code
    if found_z68:
        lines.append(f"- ICD-10 Z68 Code found: {found_z68}")
    elif missing_anchor and "Z68" in missing_anchor:
        lines.append("- ICD-10 Z68 Code found: MISSING [Action: Add Z68 code matching BMI]")
    else:
        lines.append(f"- ICD-10 Z68 Code found: {found_z68 if found_z68 else 'N/A'}")

    # 5. Comorbidity
    if bmi_val and bmi_val >= 30:
        lines.append("- Qualifying Documented Comorbidity: Not Applicable (BMI >= 30)")
    else:
        if ambiguity_code:
             term_display = found_comorb if found_comorb else ambiguity_code
             lines.append(f"- Qualifying Documented Comorbidity: AMBIGUOUS TERM '{term_display}' [Action: Clarify diagnosis]")
        elif found_comorb:
            lines.append(f"- Qualifying Documented Comorbidity: {found_comorb}")
        elif bmi_val and bmi_val < 27:
             lines.append("- Qualifying Documented Comorbidity: N/A (BMI < 27)")
        else:
             lines.append("- Qualifying Documented Comorbidity: NOT FOUND [Action: Document qualifying comorbidity for BMI 27-29.9]")

    return "\\n".join(lines)

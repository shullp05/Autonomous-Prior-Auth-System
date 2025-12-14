import re
from typing import List, Set

def normalize(text: str) -> str:
    """Coerce any incoming value to string to avoid attribute errors on floats/NaNs."""
    return str(text or "").strip().lower()

def has_word_boundary(haystack: str, needle: str) -> bool:
    pattern = rf"\b{re.escape(needle)}\b"
    return bool(re.search(pattern, haystack, re.IGNORECASE))

def matches_term(text: str, term: str) -> bool:
    """Boundary-aware match for comorbidity/safety terms. Always uses word boundaries."""
    return has_word_boundary(text, term)

def expand_safety_variants(term: str) -> List[str]:
    """
    Expand complex safety strings (e.g. 'X or Y') into individual terms.
    Useful for checking safety exclusions against specific keywords.
    """
    base = term.lower()
    variants = {base}
    if "/" in base:
        variants.update(part.strip() for part in base.split("/") if part.strip())
    if " or " in base:
        variants.update(part.strip() for part in base.split(" or ") if part.strip())
    if " syndrome" in base:
        variants.add(base.replace(" syndrome", ""))
    if "(" in base and ")" in base:
        inside = base[base.find("(") + 1 : base.find(")")]
        if len(inside) > 2:
            variants.add(inside)
        variants.add(base[: base.find("(")].strip())
    if base.startswith("personal or family history of "):
        variants.add(base.split("personal or family history of ", 1)[1])
    return [v for v in variants if v]

def is_snf_phrase(text: str) -> bool:
    """Detect skilled nursing facility phrases that should NOT trigger pregnancy/nursing exclusions."""
    t = normalize(text)
    snf_terms = ["skilled nursing facility", "snf", "nursing facility", "long-term nursing facility", "ltc facility"]
    return any(term in t for term in snf_terms)

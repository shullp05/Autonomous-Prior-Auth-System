"""
context_rules.py - Deterministic Context Classification for Safety Signals

Implements "boring" but strict rules to classify safety terms as:
- ACTIVE (Current, contraindication)
- HISTORICAL (Past, resolved)
- NEGATED (Explicitly denied)
- FAMILY_HISTORY (Not the patient)
- HYPOTHETICAL (Uncertain, possible)

Semantics:
ACTIVE -> Hard Stop (DENIED_SAFETY)
Others -> Safety Signal (SAFETY_SIGNAL_NEEDS_REVIEW)
"""

import re

# Pre-compiled regex patterns for performance
# \b ensures word boundaries so "his" doesn't match "history"

# 1. Negation
# "denies diabetes", "negative for cancer", "ruled out mtc", "no history of"
REGEX_NEGATION = re.compile(
    r"\b(denies|denied|negative|ruled out|no evidence of|not detected|resolved|no history|no family|no)\b",
    re.IGNORECASE
)

# 2. Historical/Temporality
# "history of", "prior", "remote", "status post", "s/p", "postpartum"
REGEX_HISTORICAL = re.compile(
    r"\b(history of|hx of|prior|past|remote|resolved|status post|s/p|postpartum|former)\b",
    re.IGNORECASE
)

# 3. Family History
# "family history", "mother had", "father has", "fhx"
REGEX_FAMILY = re.compile(
    r"\b(family history|fam hx|fhx|mother|father|brother|sister|grandparent|aunt|uncle)\b",
    re.IGNORECASE
)

# 4. Uncertainty/Hypothetical
# "possible", "concern for", "monitor for", "risk of"
REGEX_UNCERTAINTY = re.compile(
    r"\b(possible|probable|suspected|concern for|monitor for|risk of|evaluate for|check for)\b",
    re.IGNORECASE
)

from dataclasses import dataclass


@dataclass
class ContextClassification:
    context_type: str  # ACTIVE, HISTORICAL, NEGATED, FAMILY_HISTORY, HYPOTHETICAL
    confidence: str    # HARD_STOP, SIGNAL, CLEARED

def classify_context(text: str, term: str = "") -> ContextClassification:
    """
    Classify the context of a matched safety term within a snippet.
    
    Args:
        text: The snippet containing the term (e.g. "Patient denies history of MTC")
        term: The specific safety term matched (e.g. "MTC") - Optional in signature for bw compatibility but used for nuances if needed.
        
    Returns:
        ContextClassification object
    """
    # Normalize for basic matching
    lower_text = text.lower()

    # Check explicitly defined order of precedence:

    # 1. Negation (Strongest signal - "denies history of" is negated, not historical per se)
    # "denies history of" -> Negated.
    if _has_pattern(REGEX_NEGATION, lower_text):
        return ContextClassification('NEGATED', 'CLEARED')

    # 2. Family History (Strong signal - "family history of" is not patient)
    if _has_pattern(REGEX_FAMILY, lower_text):
        return ContextClassification('FAMILY_HISTORY', 'SIGNAL')

    # 3. Historical (Past events)
    if _has_pattern(REGEX_HISTORICAL, lower_text):
        return ContextClassification('HISTORICAL', 'SIGNAL')

    # 4. Uncertainty (Ambiguous/Hypothetical)
    if _has_pattern(REGEX_UNCERTAINTY, lower_text):
        return ContextClassification('HYPOTHETICAL', 'SIGNAL')

    # Default: If it exists and isn't negated/historical/family/uncertain, assume Active/Current.
    return ContextClassification('ACTIVE', 'HARD_STOP')

def _has_pattern(pattern: re.Pattern, text: str) -> bool:
    return bool(pattern.search(text))


import pytest
from rules.coding_integrity_rules import check_admin_readiness, _extract_codes as extract_codes

def test_extract_codes_basic():
    """Verify regex extraction of ICD-10 like codes."""
    conditions = ["Type 2 diabetes mellitus without complications (E11.9)", "Essential (primary) hypertension (I10)"]
    codes = extract_codes(conditions)
    assert "E11.9" in codes
    assert "I10" in codes

def test_extract_codes_no_codes():
    """Verify behavior when no codes are present."""
    conditions = ["Headache", "Obesity due to excess calories"]
    codes = extract_codes(conditions)
    assert len(codes) == 0

def test_admin_readiness_obesity_success():
    """Verify BMI30_OBESITY pathway with valid anchor code."""
    diags = ["Obesity, unspecified (E66.9)"]
    is_ready, missing = check_admin_readiness(diags, "BMI30_OBESITY")
    assert is_ready is True
    assert missing is None

def test_admin_readiness_obesity_failure():
    """Verify BMI30_OBESITY pathway without anchor code."""
    diags = ["Type 2 Diabetes (E11.9)"]
    is_ready, missing = check_admin_readiness(diags, "BMI30_OBESITY")
    assert is_ready is False
    assert missing == "E66.9"

def test_admin_readiness_overweight_success():
    """Verify BMI27_COMORBIDITY pathway with valid anchor code."""
    diags = ["Overweight (E66.3)"]
    is_ready, missing = check_admin_readiness(diags, "BMI27_COMORBIDITY")
    assert is_ready is True
    assert missing is None

def test_admin_readiness_overweight_failure():
    """Verify BMI27_COMORBIDITY pathway without anchor code."""
    diags = ["Sleep Apnea (G47.33)"]
    is_ready, missing = check_admin_readiness(diags, "BMI27_COMORBIDITY")
    assert is_ready is False
    assert missing == "E66.3"

def test_admin_readiness_irrelevant_pathway():
    """Verify other pathways are ignored (always ready)."""
    diags = []
    is_ready, missing = check_admin_readiness(diags, "OTHER_PATHWAY")
    assert is_ready is True
    assert missing is None

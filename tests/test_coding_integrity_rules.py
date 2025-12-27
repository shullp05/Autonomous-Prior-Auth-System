
from rules.coding_integrity_rules import check_admin_readiness

# Mock Requirements similar to Snapshot
REQ_E66_OBESITY = {"E66.01", "E66.09", "E66.1", "E66.2", "E66.8", "E66.9", "E66.0", "E66.x"}
REQ_Z68_OBESITY = {"Z68.30", "Z68.31", "Z68.32", "Z68.33", "Z68.34", "Z68.35", "Z68.36", "Z68.37", "Z68.38", "Z68.39", "Z68.41", "Z68.42", "Z68.43", "Z68.44", "Z68.45"}
REQ_STRINGS_OBESITY = ["Adult Obesity", "Morbid Obesity", "Obesity"]

REQ_E66_OVERWEIGHT = {"E66.3", "E66.0", "E66.x"}
REQ_Z68_OVERWEIGHT = {"Z68.27", "Z68.28", "Z68.29"}
REQ_STRINGS_OVERWEIGHT = ["Overweight", "Adult Overweight"]

def test_admin_readiness_obesity_success():
    """Verify success with valid Triple Key (Text + E66 + Z68)."""
    diags = [
        {"condition_name": "Obesity due to excess calories", "icd10_dx": "E66.0", "icd10_bmi": "Z68.32"}
    ]
    # Pass explicit requirements
    is_ready, missing, found = check_admin_readiness(diags, REQ_E66_OBESITY, REQ_Z68_OBESITY, REQ_STRINGS_OBESITY)
    assert is_ready is True
    assert missing is None
    assert found["text"] == "OBESITY DUE TO EXCESS CALORIES"
    assert found["e66"] == "E66.0"
    assert found["z68"] == "Z68.32"

def test_admin_readiness_missing_z68():
    """Verify failure if Z68 is missing."""
    diags = [
        {"condition_name": "Obesity", "icd10_dx": "E66.9", "icd10_bmi": ""}
    ]
    is_ready, missing, _ = check_admin_readiness(diags, REQ_E66_OBESITY, REQ_Z68_OBESITY, REQ_STRINGS_OBESITY)
    assert is_ready is False
    assert "BMI Z-Code" in missing

def test_admin_readiness_missing_e66():
    """Verify failure if E66 is missing."""
    diags = [
        {"condition_name": "Obesity", "icd10_dx": "R63.5", "icd10_bmi": "Z68.35"}
    ]
    is_ready, missing, _ = check_admin_readiness(diags, REQ_E66_OBESITY, REQ_Z68_OBESITY, REQ_STRINGS_OBESITY)
    assert is_ready is False
    assert "ICD-10 Code" in missing

def test_admin_readiness_missing_text():
    """Verify failure if diagnosis text doesn't match required strings."""
    diags = [
        {"condition_name": "High Blood Pressure", "icd10_dx": "E66.9", "icd10_bmi": "Z68.35"}
    ]
    is_ready, missing, _ = check_admin_readiness(diags, REQ_E66_OBESITY, REQ_Z68_OBESITY, REQ_STRINGS_OBESITY)
    assert is_ready is False
    assert "Diagnosis Text" in missing

def test_wildcard_e66_logic():
    """Verify E66.x matches any startswith E66 if allowed."""
    # REQ_E66_OBESITY contains "E66.x"
    diags = [
        {"condition_name": "Obesity", "icd10_dx": "E66.999", "icd10_bmi": "Z68.32"}
    ]
    is_ready, _, found = check_admin_readiness(diags, REQ_E66_OBESITY, REQ_Z68_OBESITY, REQ_STRINGS_OBESITY)
    assert is_ready is True
    assert found["e66"] == "E66.999"

def test_legacy_string_format_extraction():
    """Verify legacy string list input with embedded codes working."""
    # "Obesity (E66.9)" -> Should extract E66.9 and match text "Obesity"
    diags = ["Obesity (E66.9) (Z68.32)"]
    is_ready, missing, found = check_admin_readiness(diags, REQ_E66_OBESITY, REQ_Z68_OBESITY, REQ_STRINGS_OBESITY)
    assert is_ready is True
    assert found["e66"] == "E66.9"
    assert found["z68"] == "Z68.32"


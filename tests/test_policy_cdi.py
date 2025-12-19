
import pytest
from policy_engine import evaluate_eligibility

def test_cdi_required_bmi30_no_anchor():
    """Verify that a clinically eligible BMI>30 patient without E66.9 gets CDI_REQUIRED."""
    patient_data = {
        "latest_bmi": "35.0",
        "conditions": ["Hypertension (I10)"], # Missing E66.x
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "CDI_REQUIRED"
    assert result.decision_type == "CDI_REQUIRED"
    assert result.clinical_eligible is True
    assert result.admin_ready is False
    assert result.missing_anchor_code == "E66.9"
    assert "E66.9" in result.physician_query_text

def test_cdi_approved_bmi30_with_anchor():
    """Verify that a clinically eligible BMI>30 patient WITH E66.9 gets APPROVED."""
    patient_data = {
        "latest_bmi": "35.0",
        "conditions": ["Hypertension (I10)", "Obesity, unspecified (E66.9)"],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "APPROVED"
    assert result.decision_type == "APPROVED"
    assert result.clinical_eligible is True
    assert result.admin_ready is True
    assert result.missing_anchor_code is None

def test_cdi_required_bmi27_no_anchor():
    """Verify that a clinically eligible BMI>27+Comorb patient without E66.3 gets CDI_REQUIRED."""
    patient_data = {
        "latest_bmi": "28.0",
        "conditions": ["Hypertension (I10)"], # Valid comorb, but Missing E66.3
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "CDI_REQUIRED"
    assert result.decision_type == "CDI_REQUIRED"

    assert result.clinical_eligible is True
    assert result.admin_ready is False
    assert result.missing_anchor_code == "E66.3"

def test_cdi_denied_not_eligible():
    """Verify that a clinically ineligible patient (BMI < 27) gets DENIED, not CDI_REQUIRED."""
    patient_data = {
        "latest_bmi": "25.0",
        "conditions": ["Hypertension (I10)"],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "DENIED_CLINICAL"
    assert result.clinical_eligible is False
    # admin_ready is not computed for denied cases, or defaults to False/True. 
    # Current logic: loop breaks before CDI check.

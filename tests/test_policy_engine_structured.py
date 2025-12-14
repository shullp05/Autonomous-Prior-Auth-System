import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_engine import evaluate_eligibility
from policy_constants import ADULT_OVERWEIGHT_DIAGNOSES


def test_snf_not_pregnancy_safety():
    """Skilled nursing facility phrases must not trigger pregnancy/nursing safety exclusion."""
    data = {
        "latest_bmi": "31.0",
        "conditions": ["Patient transfer to skilled nursing facility (procedure)"],
        "meds": [],
    }
    res = evaluate_eligibility(data)
    assert res.verdict != "DENIED_SAFETY"
    assert res.safety_exclusion_code is None


def test_overweight_no_comorbidity_true_denial():
    """BMI 27-29.9 with overweight diagnosis but no comorbidity is a clinical denial, not missing info."""
    data = {
        "latest_bmi": "28.4",
        "conditions": [ADULT_OVERWEIGHT_DIAGNOSES[0]],
        "meds": [],
    }
    res = evaluate_eligibility(data)
    assert res.verdict == "DENIED_CLINICAL"
    assert res.decision_type == "DENIED_NO_COMORBIDITY"
    assert res.policy_path == "BMI27_COMORBIDITY"


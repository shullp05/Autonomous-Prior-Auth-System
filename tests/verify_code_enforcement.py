
import sys

import pytest

# Add project root to path
sys.path.append("/root/projects/PriorAuth")

from policy_engine import evaluate_eligibility
from policy_snapshot import parse_guidelines, write_policy_snapshot

# from etl_pipeline import PatientData # Not needed if passing dict

def test_full_enforcement_flow():
    # 1. Regenerate Snapshot to ensure it matches current codebase
    try:
        snap = parse_guidelines()
        write_policy_snapshot(snap)
        print("Snapshot regenerated successfully.")
    except Exception as e:
        pytest.fail(f"Snapshot generation failed: {e}")

    # 2. No Class Init needed for functional engine
    # engine = PolicyEngine()

    # 3. Test Case 1: Obesity Pathway - Success
    # Needs BMI >= 30, Text "Obesity", E66.x, Z68.x
    patient_success = {
        "conditions": [
            {"condition_name": "Obesity", "icd10_dx": "E66.9", "icd10_bmi": "Z68.32"}
        ],
        "biometrics": {
             # Derived BMI will be calculated by engine if height/weight present,
             # logic usually prioritizes `latest_bmi` if present in input dict?
             # Let's provide explicit latest_bmi string as agents often do.
             "bmi": 32.0,
        },
        "latest_bmi": 32.0, # Explicit
        "meds": []
    }

    result = evaluate_eligibility(patient_success)
    assert result.verdict == "APPROVED"
    assert result.admin_ready is True

    # 4. Test Case 2: Obesity Pathway - Missing Z68
    patient_fail_z68 = {
        "conditions": [
            {"condition_name": "Obesity", "icd10_dx": "E66.9", "icd10_bmi": ""} # Missing Z code
        ],
        "latest_bmi": 32.0,
        "meds": []
    }
    result_fail = evaluate_eligibility(patient_fail_z68)
    # Verdict should be CDI_REQUIRED because clinical is APPROVED but admin is NOT READY
    assert result_fail.verdict == "CDI_REQUIRED"
    assert "BMI Z-Code" in result_fail.decision_type or "CDI_REQUIRED" in str(result_fail.verdict)

    print("Integration test passed!")

if __name__ == "__main__":
    # Allow running as script
    test_full_enforcement_flow()


import json
import pytest
from pathlib import Path
from policy_engine import evaluate_eligibility, EligibilityResult

# Locating the fixture file relative to this test file
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gold_cases.json"

def load_gold_cases():
    if not FIXTURE_PATH.exists():
        return []
    with open(FIXTURE_PATH, "r") as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_gold_cases())
def test_gold_case_regression(case):
    """
    Run each gold case through the policy engine and verify exact match on key fields.
    """
    case_id = case["id"]
    patient_data = case["data"]
    expected = case["expected"]
    
    # Execute Engine
    result: EligibilityResult = evaluate_eligibility(patient_data)
    
    # Assertions
    # 1. Verdict
    assert result.verdict == expected["verdict"], \
        f"[{case_id}] Verdict mismatch. Expected {expected['verdict']}, got {result.verdict}. Reasoning: {result.reasoning}"
        
    # 2. Decision Type (Granular reason)
    assert result.decision_type == expected["decision_type"], \
        f"[{case_id}] Decision Type mismatch. Expected {expected['decision_type']}, got {result.decision_type}"
        
    # 3. Safety Flag
    assert result.safety_flag == expected["safety_flag"], \
        f"[{case_id}] Safety Flag mismatch. Expected {expected['safety_flag']}, got {result.safety_flag}"
        
    # Optional: Check if reasoning contains key phrases if provided in expected
    if "reasoning_contains" in expected:
        for phrase in expected["reasoning_contains"]:
            assert phrase.lower() in result.reasoning.lower(), \
                f"[{case_id}] Reasoning missing phrase '{phrase}'. Got: {result.reasoning}"

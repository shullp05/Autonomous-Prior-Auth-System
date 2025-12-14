"""
Test suite for policy guardrails in agent_logic.py

Tests the _apply_policy_guardrails() function which enforces hard policy rules
after the LLM's reasoning to catch edge cases like:
- Approving BMI 29.1 with no comorbidities
- Treating prediabetes as valid comorbidity
- Treating generic "sleep apnea" as qualifying OSA
- Treating generic thyroid cancer as MTC safety exclusion
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from policy_constants import (
    BMI_OBESE_THRESHOLD,
    BMI_OVERWEIGHT_THRESHOLD,
    AMBIGUOUS_DIABETES,
    AMBIGUOUS_BP,
    AMBIGUOUS_THYROID,
)


# Import the function under test
# Note: We need to mock the data loading since agent_logic tries to load CSVs on import
@pytest.fixture(scope="module")
def guardrails_func():
    """Import _apply_policy_guardrails after mocking data dependencies."""
    # Create minimal mock data files if they don't exist
    import pandas as pd
    
    test_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create minimal test data if not present
    if not os.path.exists(os.path.join(test_data_dir, "data_patients.csv")):
        pd.DataFrame({"patient_id": ["TEST001"], "name": ["Test"], "dob": ["2000-01-01"]}).to_csv(
            os.path.join(test_data_dir, "data_patients.csv"), index=False
        )
    if not os.path.exists(os.path.join(test_data_dir, "data_medications.csv")):
        pd.DataFrame({"patient_id": ["TEST001"], "medication_name": ["Metformin"]}).to_csv(
            os.path.join(test_data_dir, "data_medications.csv"), index=False
        )
    if not os.path.exists(os.path.join(test_data_dir, "data_conditions.csv")):
        pd.DataFrame({"patient_id": ["TEST001"], "condition_name": ["Hypertension"]}).to_csv(
            os.path.join(test_data_dir, "data_conditions.csv"), index=False
        )
    if not os.path.exists(os.path.join(test_data_dir, "data_observations.csv")):
        pd.DataFrame({"patient_id": ["TEST001"], "type": ["BMI"], "value": [30.0], "date": ["2024-01-01"]}).to_csv(
            os.path.join(test_data_dir, "data_observations.csv"), index=False
        )
    
    # Mock ollama and other dependencies to avoid import errors
    sys.modules["ollama"] = type("Mock", (object,), {})
    sys.modules["langchain_ollama"] = type("Mock", (object,), {})
    sys.modules["langchain"] = type("Mock", (object,), {})
    sys.modules["langgraph"] = type("Mock", (object,), {"graph": type("Mock", (object,), {"StateGraph": type("Mock", (object,), {})})})
    
    # We only need _apply_policy_guardrails, which doesn't depend on these libs
    # But agent_logic imports them at module level.
    # We can use unittest.mock to patch them if sys.modules trick isn't enough
    # or just wrap the import in a try/except block in agent_logic if feasible,
    # but here we are modifying the test.
    
    try:
        from agent_logic import _apply_policy_guardrails
    except ImportError:
        # If imports fail despite mocks, we might need to be more aggressive
        # For now, let's see if the sys.modules trick works for simple module imports
        # If agent_logic uses them immediately, we might need to mock attributes too.
        # Let's try to mock enough to satisfy import.
        import types
        m = types.ModuleType("ollama")
        sys.modules["ollama"] = m
        
        m2 = types.ModuleType("langchain_ollama")
        m2.ChatOllama = type("Mock", (object,), {})
        sys.modules["langchain_ollama"] = m2
        
        m3 = types.ModuleType("langchain")
        sys.modules["langchain"] = m3
        
        # Mock langgraph as a package structure
        m4 = types.ModuleType("langgraph")
        m4.graph = types.ModuleType("graph")
        m4.graph.StateGraph = lambda *args: None
        m4.graph.END = "END"
        m4.prebuilt = types.ModuleType("prebuilt")
        m4.prebuilt.ToolNode = lambda *args: None
        sys.modules["langgraph"] = m4
        sys.modules["langgraph.graph"] = m4.graph
        sys.modules["langgraph.prebuilt"] = m4.prebuilt
        
        from agent_logic import _apply_policy_guardrails

    return _apply_policy_guardrails


class TestBMIThresholds:
    """Tests for BMI-based approval/denial logic."""

    def test_approve_bmi_30_no_comorbidity(self, guardrails_func):
        """BMI >= 30 should be approved without comorbidity."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 30.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "BMI is 30.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"

    def test_approve_bmi_35_obese(self, guardrails_func):
        """BMI 35 (clearly obese) should be approved."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "BMI is 35.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"

    def test_reject_bmi_26_approval(self, guardrails_func):
        """BMI < 27 should NEVER be approved, even if LLM says so."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 26.5,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM incorrectly approved.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "DENIED_CLINICAL"

    def test_reject_bmi_missing_approval(self, guardrails_func):
        """Missing BMI should not be approved."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": None,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM incorrectly approved without BMI.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "DENIED_MISSING_INFO"

    def test_bmi_27_with_valid_comorbidity(self, guardrails_func):
        """BMI 27-29.9 with valid comorbidity should stay approved."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 28.5,
            "safety_flag": "CLEAR",
            "comorbidity_category": "HYPERTENSION",
            "evidence_quoted": "Hypertension",
            "reasoning": "BMI 28.5 with HTN.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"

    def test_bmi_29_no_comorbidity_rejected(self, guardrails_func):
        """BMI 27-29.9 without comorbidity should be denied."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 29.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM incorrectly approved.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "DENIED_CLINICAL"


class TestAmbiguousTerms:
    """Tests for ambiguous term detection."""

    def test_prediabetes_not_qualifying(self, guardrails_func):
        """Prediabetes should NOT qualify as diabetes comorbidity."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 28.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "DIABETES",
            "evidence_quoted": "Prediabetes",
            "reasoning": "LLM incorrectly treated prediabetes as diabetes.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "MANUAL_REVIEW"

    def test_borderline_diabetes_not_qualifying(self, guardrails_func):
        """Borderline diabetes should NOT qualify."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 27.5,
            "safety_flag": "CLEAR",
            "comorbidity_category": "DIABETES",
            "evidence_quoted": "borderline diabetes",
            "reasoning": "LLM incorrectly treated borderline diabetes.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "MANUAL_REVIEW"

    def test_generic_sleep_apnea_not_qualifying(self, guardrails_func):
        """Generic 'sleep apnea' should NOT qualify (must be OSA)."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 28.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "OSA",
            "evidence_quoted": "Sleep Apnea",
            "reasoning": "LLM incorrectly treated generic sleep apnea as OSA.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "MANUAL_REVIEW"

    def test_obstructive_sleep_apnea_qualifies(self, guardrails_func):
        """Obstructive Sleep Apnea SHOULD qualify."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 28.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "OSA",
            "evidence_quoted": "Obstructive Sleep Apnea",
            "reasoning": "OSA is documented.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"

    def test_elevated_bp_counts_per_policy(self, guardrails_func):
        """Elevated BP is included in hypertension strings for BMI 27-29.9."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 28.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "HYPERTENSION",
            "evidence_quoted": "elevated blood pressure",
            "reasoning": "LLM incorrectly treated elevated BP.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"


class TestSafetyExclusions:
    """Tests for safety exclusion logic."""

    def test_safety_flag_overrides_approval(self, guardrails_func):
        """Safety exclusion should override any approval."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "DETECTED",
            "comorbidity_category": "NONE",
            "evidence_quoted": "Pregnancy",
            "reasoning": "LLM incorrectly approved despite safety flag.",
        }
        patient_data = {"conditions": ["Medullary Thyroid Carcinoma"], "meds": []}
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] == "DENIED_SAFETY"

    def test_generic_thyroid_cancer_not_safety_exclusion(self, guardrails_func):
        """Generic thyroid cancer should NOT trigger safety denial."""
        audit = {
            "verdict": "DENIED_SAFETY",
            "bmi_numeric": 30.0,
            "safety_flag": "DETECTED",
            "comorbidity_category": "NONE",
            "evidence_quoted": "Thyroid cancer",
            "reasoning": "LLM incorrectly treated thyroid cancer as MTC.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "MANUAL_REVIEW"
        assert result["safety_flag"] == "CLEAR"

    def test_papillary_thyroid_not_safety_exclusion(self, guardrails_func):
        """Papillary thyroid carcinoma should NOT trigger safety denial."""
        audit = {
            "verdict": "DENIED_SAFETY",
            "bmi_numeric": 32.0,
            "safety_flag": "DETECTED",
            "comorbidity_category": "NONE",
            "evidence_quoted": "Papillary thyroid carcinoma",
            "reasoning": "LLM incorrectly treated papillary as MTC.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "MANUAL_REVIEW"

    def test_mtc_is_safety_exclusion(self, guardrails_func):
        """Medullary Thyroid Carcinoma IS a valid safety exclusion."""
        audit = {
            "verdict": "DENIED_SAFETY",
            "bmi_numeric": 30.0,
            "safety_flag": "DETECTED",
            "comorbidity_category": "NONE",
            "evidence_quoted": "Medullary Thyroid Carcinoma",
            "reasoning": "MTC is a safety exclusion.",
        }
        patient_data = {"conditions": ["Multiple Endocrine Neoplasia type 2"], "meds": []}
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] == "DENIED_SAFETY"

    def test_men2_is_safety_exclusion(self, guardrails_func):
        """MEN2 IS a valid safety exclusion."""
        audit = {
            "verdict": "DENIED_SAFETY",
            "bmi_numeric": 30.0,
            "safety_flag": "DETECTED",
            "comorbidity_category": "NONE",
            "evidence_quoted": "Multiple Endocrine Neoplasia type 2 (MEN2)",
            "reasoning": "MEN2 is a safety exclusion.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "DENIED_SAFETY"

    def test_hypersensitivity_safety_exclusion(self, guardrails_func):
        """Hypersensitivity to semaglutide is a safety exclusion."""
        # Test via patient_data override
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM missed hypersensitivity.",
        }
        patient_data = {
            "conditions": ["History of anaphylaxis to semaglutide"],
            "meds": []
        }
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] == "DENIED_SAFETY"

    def test_pancreatitis_safety_exclusion(self, guardrails_func):
        """History of pancreatitis is a safety exclusion."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM missed pancreatitis.",
        }
        patient_data = {
            "conditions": ["prior acute pancreatitis"],
            "meds": []
        }
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] == "DENIED_SAFETY"

    def test_renal_disease_not_safety_exclusion(self, guardrails_func):
        """Renal disease/dialysis is not a Wegovy safety exclusion."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM considered renal disease.",
        }
        patient_data = {
            "conditions": ["end-stage renal disease"],
            "meds": []
        }
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] != "DENIED_SAFETY"

    def test_suicidality_safety_exclusion(self, guardrails_func):
        """Suicidality history is a safety exclusion."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM missed suicidality.",
        }
        patient_data = {
            "conditions": ["History of suicide attempt"],
            "meds": []
        }
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] == "DENIED_SAFETY"

    def test_gi_motility_safety_exclusion(self, guardrails_func):
        """Severe GI motility disorder is a safety exclusion."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 35.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM missed gastroparesis.",
        }
        patient_data = {
            "conditions": ["Severe gastroparesis"],
            "meds": []
        }
        result = guardrails_func(audit, patient_data)
        assert result["verdict"] == "DENIED_SAFETY"


class TestBoundaryConditions:
    """Tests for exact boundary values."""

    def test_bmi_exactly_27(self, guardrails_func):
        """BMI exactly 27.0 with valid comorbidity should be approved."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 27.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "LIPIDS",
            "evidence_quoted": "Dyslipidemia",
            "reasoning": "BMI 27 with lipids.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"

    def test_bmi_exactly_30(self, guardrails_func):
        """BMI exactly 30.0 should be approved without comorbidity."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 30.0,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "BMI is 30.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "APPROVED"

    def test_bmi_29_99_needs_comorbidity(self, guardrails_func):
        """BMI 29.99 (just under 30) without comorbidity should be denied."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 29.99,
            "safety_flag": "CLEAR",
            "comorbidity_category": "NONE",
            "evidence_quoted": "",
            "reasoning": "LLM rounded up to 30.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "DENIED_CLINICAL"

    def test_bmi_26_99_always_denied(self, guardrails_func):
        """BMI 26.99 should always be denied."""
        audit = {
            "verdict": "APPROVED",
            "bmi_numeric": 26.99,
            "safety_flag": "CLEAR",
            "comorbidity_category": "HYPERTENSION",
            "evidence_quoted": "Hypertension",
            "reasoning": "LLM incorrectly approved BMI 26.99.",
        }
        result = guardrails_func(audit)
        assert result["verdict"] == "DENIED_CLINICAL"

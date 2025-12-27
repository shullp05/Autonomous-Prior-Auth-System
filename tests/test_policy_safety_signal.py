"""
test_policy_safety_signal.py - Integration tests for Phase 9 Safety Signals
"""

from policy_engine import evaluate_eligibility


class TestPolicySafetySignals:

    def test_active_safety_hard_stop(self):
        """Active condition -> DENIED_SAFETY"""
        patient = {
            "latest_bmi": "22.0",
            "conditions": ["Active Medullary Thyroid Carcinoma"],
            "meds": []
        }
        result = evaluate_eligibility(patient)
        assert result.verdict == "DENIED_SAFETY"
        assert result.safety_flag == "DETECTED"
        assert result.safety_confidence == "HARD_STOP"
        assert result.safety_context == "ACTIVE"

    def test_historical_safety_signal(self):
        """Historical condition -> SAFETY_SIGNAL_NEEDS_REVIEW"""
        patient = {
            "latest_bmi": "22.0",
            "conditions": ["History of pancreatitis"],
            "meds": []
        }
        result = evaluate_eligibility(patient)
        assert result.verdict == "SAFETY_SIGNAL_NEEDS_REVIEW"
        assert result.safety_flag == "DETECTED" # Flag is raised
        assert result.safety_confidence == "SIGNAL" # But verification is Signal
        assert result.safety_context == "HISTORICAL"
        assert result.decision_type == "FLAGGED_SAFETY_WARNING"

    def test_negated_safety_signal(self):
        """Negated condition -> SAFETY_SIGNAL_NEEDS_REVIEW
        (Conservative: even negated terms are flagged for human review to confirm the negation context is correct)
        """
        patient = {
            "latest_bmi": "35.0", # Eligible BMI
            "conditions": ["Patient denies history of MTC"],
            "meds": []
        }
        result = evaluate_eligibility(patient)

        # Even though BMI is eligible, Safety Signal preempts Approval?
        # Yes, logic: 1) Safety Gate -> Signal -> Return Signal.
        # So it blocks approval until manual review clears the signal.
        assert result.verdict == "SAFETY_SIGNAL_NEEDS_REVIEW"
        assert result.safety_context == "NEGATED"
        assert result.safety_confidence == "SIGNAL"

    def test_concurrent_glp1_hard_stop(self):
        """Concurrent GLP-1 -> DENIED_SAFETY (Hard Stop)"""
        patient = {
            "latest_bmi": "35.0",
            "conditions": [],
            "meds": ["Ozempic"] # Prohibited
        }
        result = evaluate_eligibility(patient)
        assert result.verdict == "DENIED_SAFETY"
        assert result.safety_context == "ACTIVE"
        assert result.safety_exclusion_code == "CONCURRENT_GLP1"

    def test_clean_case_approved(self):
        """No safety terms -> APPROVED"""
        patient = {
            "latest_bmi": "35.0",
            "conditions": ["Hypertension"],
            "meds": []
        }
        result = evaluate_eligibility(patient)
        assert result.verdict == "APPROVED"
        assert result.safety_flag == "CLEAR"
        assert result.safety_context is None

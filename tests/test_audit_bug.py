from unittest.mock import MagicMock, patch

import pytest

import policy_engine


def test_audit_logging_gap():
    """
    Bug Reproduction: Verify that evaluate_eligibility logs detailed decisions 
    to the audit logger for ALL verdicts, not just CDI_REQUIRED.
    """
    # Mock the module-level _audit_logger
    mock_logger = MagicMock()
    with patch('policy_engine._audit_logger', mock_logger):

        # Case 1: Clear Approval (BMI 35)
        # Should return APPROVED and LOG the event
        # E66.9 ensures admin_readiness so it hits the APPROVED return path
        patient_data = {
            "latest_bmi": "35.0",
            "conditions": ["Obesity", "Obesity, unspecified (E66.9)"],
            "meds": [],
            "patient_id": "TEST_PATIENT_1"
        }

        result = policy_engine.evaluate_eligibility(patient_data)

        assert result.verdict == "APPROVED"

        # FAIL condition: If bug exists, this call count will be 0
        if mock_logger.log_event.call_count == 0:
            pytest.fail("Audit logger was NOT called for APPROVED decision!")

        mock_logger.log_event.assert_called_once()
        args, kwargs = mock_logger.log_event.call_args
        assert kwargs['event_type'] == "DECISION"
        assert kwargs['patient_id'] == "TEST_PATIENT_1"
        assert kwargs['details']['output']['verdict'] == "APPROVED"

def test_audit_logging_safety_denial():
    """Verify safety denial also logs."""
    mock_logger = MagicMock()
    with patch('policy_engine._audit_logger', mock_logger):
         # Case 2: Safety Denial (Pregnancy)
        patient_data = {
            "latest_bmi": "35.0",
            "conditions": ["Pregnancy"],
            "meds": [],
            "patient_id": "TEST_PATIENT_2"
        }

        result = policy_engine.evaluate_eligibility(patient_data)
        assert result.verdict == "DENIED_SAFETY"

        if mock_logger.log_event.call_count == 0:
            pytest.fail("Audit logger was NOT called for DENIED_SAFETY decision!")

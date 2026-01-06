
import unittest
from unittest.mock import MagicMock, patch

from priorauth.agent_logic import make_decision


class TestAuditFix(unittest.TestCase):
    @patch("priorauth.agent_logic.generate_approved_letter")
    @patch("priorauth.agent_logic.generate_appeal_letter")
    @patch("priorauth.agent_logic.get_audit_logger")
    def test_decision_logs_event(self, mock_get_logger, mock_appeal, mock_approve):
        # Setup mock logger
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        # Setup state for an APPROVED decision
        state = {
            "audit_findings": {
                "verdict": "APPROVED",
                "reasoning": "Meets criteria",
                "evidence_quoted": "BMI 31.0"
            },
            "patient_data": {"patient_id": "TEST_PATIENT"},
        }

        # Run make_decision
        result = make_decision(state)

        # Assert log_event was called
        mock_get_logger.assert_called()
        mock_logger_instance.log_event.assert_called_once()

        args, kwargs = mock_logger_instance.log_event.call_args
        self.assertEqual(kwargs['event_type'], "DECISION")
        self.assertEqual(kwargs['patient_id'], "TEST_PATIENT")
        self.assertEqual(kwargs['details']['verdict'], "APPROVED")

if __name__ == "__main__":
    unittest.main()

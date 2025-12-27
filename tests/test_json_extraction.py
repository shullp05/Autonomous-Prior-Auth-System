"""
Test suite for _extract_json_object() in agent_logic.py

Tests the robust JSON extraction function that handles:
- Clean JSON
- Markdown-fenced JSON (```json ... ```)
- JSON with surrounding text
- Malformed JSON that should raise errors
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope="module")
def extract_func():
    """Import _extract_json_object after mocking data dependencies."""
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

    from agent_logic import _extract_json_object
    return _extract_json_object


class TestCleanJSON:
    """Tests for properly formatted JSON."""

    def test_simple_object(self, extract_func):
        """Simple JSON object should parse correctly."""
        text = '{"verdict": "APPROVED", "bmi_numeric": 30.5}'
        result = extract_func(text)
        assert result["verdict"] == "APPROVED"
        assert result["bmi_numeric"] == 30.5

    def test_complex_object(self, extract_func):
        """Full audit result object should parse correctly."""
        text = '''
        {
            "bmi_numeric": 28.5,
            "safety_flag": "CLEAR",
            "comorbidity_category": "HYPERTENSION",
            "evidence_quoted": "Hypertension documented",
            "verdict": "APPROVED",
            "reasoning": "BMI 28.5 with HTN qualifies."
        }
        '''
        result = extract_func(text)
        assert result["verdict"] == "APPROVED"
        assert result["bmi_numeric"] == 28.5
        assert result["comorbidity_category"] == "HYPERTENSION"

    def test_null_values(self, extract_func):
        """JSON with null values should parse correctly."""
        text = '{"bmi_numeric": null, "verdict": "DENIED_MISSING_INFO"}'
        result = extract_func(text)
        assert result["bmi_numeric"] is None
        assert result["verdict"] == "DENIED_MISSING_INFO"


class TestMarkdownFenced:
    """Tests for markdown-fenced JSON."""

    def test_json_fence(self, extract_func):
        """JSON in ```json ... ``` fence should parse."""
        text = '''Here is the result:
```json
{"verdict": "APPROVED", "bmi_numeric": 31.0}
```
That's my analysis.'''
        result = extract_func(text)
        assert result["verdict"] == "APPROVED"
        assert result["bmi_numeric"] == 31.0

    def test_plain_fence(self, extract_func):
        """JSON in plain ``` ... ``` fence should parse."""
        text = '''
```
{"verdict": "DENIED_SAFETY", "safety_flag": "DETECTED"}
```
'''
        result = extract_func(text)
        assert result["verdict"] == "DENIED_SAFETY"
        assert result["safety_flag"] == "DETECTED"


class TestSurroundingText:
    """Tests for JSON with surrounding prose."""

    def test_text_before_json(self, extract_func):
        """JSON with text before it should parse."""
        text = 'Based on my analysis, the result is: {"verdict": "APPROVED", "bmi_numeric": 30.0}'
        result = extract_func(text)
        assert result["verdict"] == "APPROVED"

    def test_text_after_json(self, extract_func):
        """JSON with text after it should parse."""
        text = '{"verdict": "DENIED_CLINICAL", "bmi_numeric": 25.0} This patient does not qualify.'
        result = extract_func(text)
        assert result["verdict"] == "DENIED_CLINICAL"
        assert result["bmi_numeric"] == 25.0

    def test_text_both_sides(self, extract_func):
        """JSON with text on both sides should parse."""
        text = 'Here is the decision: {"verdict": "MANUAL_REVIEW", "evidence_quoted": "Prediabetes"} End of analysis.'
        result = extract_func(text)
        assert result["verdict"] == "MANUAL_REVIEW"


class TestMalformedJSON:
    """Tests for malformed JSON that should raise errors."""

    def test_missing_closing_brace(self, extract_func):
        """Missing closing brace should raise error."""
        text = '{"verdict": "APPROVED"'
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

    def test_single_quotes(self, extract_func):
        """Single quotes (Python-style) should raise error."""
        text = "{'verdict': 'APPROVED'}"
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

    def test_trailing_comma(self, extract_func):
        """Trailing comma should raise error."""
        text = '{"verdict": "APPROVED",}'
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

    def test_no_json_at_all(self, extract_func):
        """No JSON content should raise error."""
        text = "I cannot determine the verdict."
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

    def test_array_not_object(self, extract_func):
        """JSON array (not object) should raise error."""
        text = '["APPROVED", 30.0]'
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

    def test_empty_string(self, extract_func):
        """Empty string should raise error."""
        text = ""
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_nested_braces_in_string(self, extract_func):
        """JSON with braces inside string values should parse."""
        text = '{"evidence_quoted": "BMI {calculated}", "verdict": "APPROVED"}'
        result = extract_func(text)
        assert result["verdict"] == "APPROVED"
        assert "{calculated}" in result["evidence_quoted"]

    def test_unicode_characters(self, extract_func):
        """JSON with unicode should parse."""
        text = '{"reasoning": "BMI ≥ 30 kg/m²", "verdict": "APPROVED"}'
        result = extract_func(text)
        assert "≥" in result["reasoning"]

    def test_escaped_quotes(self, extract_func):
        """JSON with escaped quotes should parse."""
        text = '{"evidence_quoted": "Patient said \\"I have hypertension\\"", "verdict": "APPROVED"}'
        result = extract_func(text)
        assert "hypertension" in result["evidence_quoted"]

    def test_whitespace_only(self, extract_func):
        """Whitespace only should raise error."""
        text = "   \n\t  "
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

    def test_multiple_json_objects_raises_error(self, extract_func):
        """Multiple JSON objects should raise error (strict parsing)."""
        text = '{"verdict": "APPROVED"} {"verdict": "DENIED"}'
        # json.loads correctly rejects multiple objects
        with pytest.raises(json.JSONDecodeError):
            extract_func(text)

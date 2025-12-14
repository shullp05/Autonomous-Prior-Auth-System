"""
Adversarial Test Harness for PriorAuth System

This module provides systematic evaluation of edge cases to verify:
1. Zero false approvals for safety exclusions
2. Correct handling of BMI boundary conditions
3. Proper distinction between qualifying and ambiguous terms
4. Appeals only generated when evidence exists

Run with: pytest tests/test_adversarial.py -v
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from policy_engine import evaluate_eligibility, EligibilityResult
from policy_constants import ADULT_OBESITY_DIAGNOSES, ADULT_OVERWEIGHT_DIAGNOSES, BMI_OBESE_THRESHOLD

OBESITY_DX = ADULT_OBESITY_DIAGNOSES[0]
OVERWEIGHT_DX = ADULT_OVERWEIGHT_DIAGNOSES[0]


def run_eval(patient_data: dict) -> EligibilityResult:
    data = dict(patient_data)
    conds = data.get("conditions", []) or []
    try:
        bmi_val = float(data.get("latest_bmi"))
    except Exception:
        bmi_val = None
    diagnosis = OBESITY_DX if bmi_val is not None and bmi_val >= BMI_OBESE_THRESHOLD else OVERWEIGHT_DX
    data["conditions"] = [diagnosis] + conds
    data.setdefault("meds", [])
    return evaluate_eligibility(data)


class TestBMIBoundaryConditions:
    """Test exact BMI boundary values to ensure correct threshold handling."""
    
    @pytest.mark.parametrize("bmi,expected_verdict", [
        # Below minimum threshold - always denied
        (26.0, "DENIED_CLINICAL"),
        (26.5, "DENIED_CLINICAL"),
        (26.9, "DENIED_CLINICAL"),
        (26.99, "DENIED_CLINICAL"),
        # At overweight threshold - needs comorbidity (denied without)
        (27.0, "DENIED_CLINICAL"),
        (27.01, "DENIED_CLINICAL"),
        # Overweight range without comorbidity
        (28.0, "DENIED_CLINICAL"),
        (29.0, "DENIED_CLINICAL"),
        (29.9, "DENIED_CLINICAL"),
        (29.99, "DENIED_CLINICAL"),
        # At obesity threshold - approved without comorbidity
        (30.0, "APPROVED"),
        (30.01, "APPROVED"),
        # Clearly obese
        (35.0, "APPROVED"),
        (40.0, "APPROVED"),
        (45.0, "APPROVED"),
    ])
    def test_bmi_thresholds_no_comorbidity(self, bmi, expected_verdict):
        """BMI thresholds without comorbidity."""
        result = run_eval({
            "latest_bmi": str(bmi),
            "conditions": [],
            "meds": [],
        })
        assert result.verdict == expected_verdict, f"BMI {bmi}: expected {expected_verdict}, got {result.verdict}"
    
    @pytest.mark.parametrize("bmi", [27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 29.9])
    def test_overweight_with_valid_comorbidity_approved(self, bmi):
        """BMI 27-29.9 with valid comorbidity should be approved."""
        result = run_eval({
            "latest_bmi": str(bmi),
            "conditions": ["Hypertension"],
            "meds": [],
        })
        assert result.verdict == "APPROVED"
        assert result.comorbidity_category == "HYPERTENSION"


class TestSafetyExclusionZeroFalseApprovals:
    """
    CRITICAL: Verify zero false approvals when safety exclusions are present.
    Even with perfect eligibility (BMI 40+), safety exclusions MUST deny.
    """
    
    @pytest.mark.parametrize("safety_condition", [
        "Medullary Thyroid Carcinoma",
        "Medullary Thyroid Carcinoma (MTC)",
        "MTC",
        "Multiple Endocrine Neoplasia type 2",
        "MEN2",
        "MEN 2",
        "Pregnant",
        "Currently pregnant",
        "Pregnancy",
        "Breastfeeding",
        "Nursing",
        "Lactating",
        "Serious hypersensitivity reaction to semaglutide",
        "Anaphylaxis to Wegovy",
        "Angioedema due to semaglutide",
        "Severe allergic reaction to semaglutide",
        "History of pancreatitis",
        "Prior acute pancreatitis",
        "Chronic pancreatitis",
        "Pancreatitis while on GLP-1",
        "History of suicide attempt",
        "Active suicidal ideation",
        "Self-harm behavior",
        "Severe gastroparesis",
        "Major GI motility disorder",
    ])
    def test_safety_exclusion_denies_despite_perfect_eligibility(self, safety_condition):
        """Safety exclusions MUST deny even with BMI 40+."""
        result = run_eval({
            "latest_bmi": "40.0",  # Clearly obese
            "conditions": [safety_condition, "Hypertension", "Type 2 Diabetes"],  # Perfect eligibility
            "meds": [],
        })
        assert result.verdict == "DENIED_SAFETY", f"Failed to deny for '{safety_condition}'"
        assert result.safety_flag == "DETECTED"
    
    @pytest.mark.parametrize("glp1_med", [
        "Ozempic",
        "ozempic",
        "OZEMPIC",
        "Ozempic (semaglutide) injection",
        "Trulicity",
        "Dulaglutide",
        "Victoza",
        "Saxenda",
        "Liraglutide",
        "Byetta",
        "Bydureon",
        "Exenatide",
        "Rybelsus",
        "semaglutide injection",
        "semaglutide tablets",
    ])
    def test_concurrent_glp1_denies_despite_eligibility(self, glp1_med):
        """Concurrent GLP-1 use MUST deny."""
        result = run_eval({
            "latest_bmi": "35.0",
            "conditions": ["Hypertension"],
            "meds": [glp1_med],
        })
        assert result.verdict == "DENIED_SAFETY", f"Failed to deny for concurrent '{glp1_med}'"
    
    def test_wegovy_itself_is_allowed(self):
        """Wegovy itself should NOT trigger concurrent GLP-1 denial."""
        result = run_eval({
            "latest_bmi": "32.0",
            "conditions": [],
            "meds": ["Wegovy 2.4mg injection"],
        })
        assert result.verdict == "APPROVED"


class TestAmbiguousTermsManualReview:
    """
    Verify ambiguous terms trigger MANUAL_REVIEW, not approval or denial.
    These terms are clinically meaningful but don't meet strict policy criteria.
    """
    
    @pytest.mark.parametrize("ambiguous_term", [
        "Prediabetes",
        "Pre-diabetes",
        "Borderline diabetes",
        "Impaired fasting glucose",
    ])
    def test_ambiguous_diabetes_triggers_manual_review(self, ambiguous_term):
        """Prediabetes variants should trigger manual review, not approval."""
        result = run_eval({
            "latest_bmi": "28.5",  # Overweight
            "conditions": [ambiguous_term],
            "meds": [],
        })
        assert result.verdict == "MANUAL_REVIEW", f"'{ambiguous_term}' should trigger MANUAL_REVIEW"
    
    @pytest.mark.parametrize("ambiguous_term", [
        "Elevated blood pressure",
        "Borderline hypertension",
    ])
    def test_ambiguous_bp_triggers_manual_review(self, ambiguous_term):
        """
        Per policy:
        - 'elevated blood pressure' is in hypertension accepted strings -> APPROVED
        - 'Elevated BP' is in ambiguities -> MANUAL_REVIEW
        - 'Borderline hypertension' is in ambiguities -> MANUAL_REVIEW
        """
        result = run_eval({
            "latest_bmi": "28.5",
            "conditions": [ambiguous_term],
            "meds": [],
        })
        if "borderline" in ambiguous_term.lower():
            assert result.verdict == "MANUAL_REVIEW", f"'{ambiguous_term}' should trigger MANUAL_REVIEW"
        else:
            # "elevated blood pressure" is explicitly in hypertension accepted strings
            assert result.verdict == "APPROVED"
    
    def test_elevated_bp_abbreviation_is_ambiguous(self):
        """Per policy, 'Elevated BP' (abbreviation) is in ambiguities, not accepted strings."""
        result = run_eval({
            "latest_bmi": "28.5",
            "conditions": ["Elevated BP"],
            "meds": [],
        })
        assert result.verdict == "MANUAL_REVIEW", "'Elevated BP' is ambiguous per policy"
    
    def test_generic_sleep_apnea_triggers_manual_review(self):
        """Generic 'sleep apnea' without 'obstructive' should trigger manual review."""
        result = run_eval({
            "latest_bmi": "28.5",
            "conditions": ["Sleep apnea"],
            "meds": [],
        })
        assert result.verdict == "MANUAL_REVIEW"
    
    def test_obstructive_sleep_apnea_approves(self):
        """Explicit 'Obstructive Sleep Apnea' should approve."""
        result = run_eval({
            "latest_bmi": "28.5",
            "conditions": ["Obstructive Sleep Apnea"],
            "meds": [],
        })
        assert result.verdict == "APPROVED"
        assert result.comorbidity_category == "OSA"
    
    def test_osa_abbreviation_approves(self):
        """OSA abbreviation should approve."""
        result = run_eval({
            "latest_bmi": "28.5",
            "conditions": ["OSA"],
            "meds": [],
        })
        assert result.verdict == "APPROVED"
        assert result.comorbidity_category == "OSA"


class TestThyroidDistinction:
    """
    Critical: Distinguish between MTC (safety exclusion) and other thyroid cancers (manual review).
    """
    
    @pytest.mark.parametrize("mtc_term", [
        "Medullary Thyroid Carcinoma",
        "MTC",
        "Multiple Endocrine Neoplasia type 2",
        "MEN2",
        "MEN 2",
    ])
    def test_mtc_men2_is_safety_exclusion(self, mtc_term):
        """MTC/MEN2 MUST be safety exclusion."""
        result = run_eval({
            "latest_bmi": "35.0",
            "conditions": [mtc_term],
            "meds": [],
        })
        assert result.verdict == "DENIED_SAFETY"
    
    @pytest.mark.parametrize("non_mtc_thyroid", [
        "Thyroid cancer",
        "Thyroid carcinoma",
        "Papillary thyroid carcinoma",
        "Follicular thyroid carcinoma",
        "Malignant tumor of thyroid",
    ])
    def test_non_mtc_thyroid_is_manual_review(self, non_mtc_thyroid):
        """Non-MTC thyroid cancers should be MANUAL_REVIEW, not safety denial."""
        result = run_eval({
            "latest_bmi": "35.0",
            "conditions": [non_mtc_thyroid],
            "meds": [],
        })
        # Should be manual review, NOT safety denial
        assert result.verdict == "MANUAL_REVIEW", f"'{non_mtc_thyroid}' should be MANUAL_REVIEW, not {result.verdict}"
        assert result.safety_flag == "CLEAR"


class TestQualifyingComorbidities:
    """Verify all qualifying comorbidities are properly recognized."""
    
    @pytest.mark.parametrize("condition,expected_category", [
        # Hypertension variants
        ("Hypertension", "HYPERTENSION"),
        ("Essential hypertension", "HYPERTENSION"),
        ("HTN", "HYPERTENSION"),
        ("High blood pressure", "HYPERTENSION"),
        # Lipid disorders
        ("Dyslipidemia", "LIPIDS"),
        ("Hyperlipidemia", "LIPIDS"),
        ("High cholesterol", "LIPIDS"),
        ("Mixed hyperlipidemia", "LIPIDS"),
        # Type 2 Diabetes
        ("Type 2 Diabetes", "DIABETES"),
        ("Type 2 diabetes mellitus", "DIABETES"),
        ("Type II Diabetes", "DIABETES"),
        ("T2DM", "DIABETES"),
        # OSA (explicit only)
        ("Obstructive Sleep Apnea", "OSA"),
        ("OSA", "OSA"),
        # CVD
        ("Coronary Artery Disease", "CVD"),
        ("CAD", "CVD"),
        ("Myocardial Infarction", "CVD"),
        ("MI", "CVD"),
        ("Heart Attack", "CVD"),
        ("Stroke", "CVD"),
        ("Ischemic Stroke", "CVD"),
        ("Peripheral Arterial Disease", "CVD"),
        ("PAD", "CVD"),
    ])
    def test_qualifying_comorbidity_approves(self, condition, expected_category):
        """Qualifying comorbidities with BMI 27-29.9 should approve."""
        result = run_eval({
            "latest_bmi": "28.0",
            "conditions": [condition],
            "meds": [],
        })
        assert result.verdict == "APPROVED", f"'{condition}' should approve"
        assert result.comorbidity_category == expected_category


class TestMissingDataHandling:
    """Test handling of missing or invalid data."""
    
    @pytest.mark.parametrize("bmi_value", [
        "MISSING_DATA",
        "",
        None,
    ])
    def test_missing_bmi_denied(self, bmi_value):
        """Missing BMI should be DENIED_MISSING_INFO."""
        result = run_eval({
            "latest_bmi": bmi_value,
            "conditions": ["Hypertension"],
            "meds": [],
        })
        assert result.verdict == "DENIED_MISSING_INFO"
    
    def test_unreasonable_bmi_treated_as_missing(self):
        """BMI outside reasonable range should be treated as missing."""
        result = run_eval({
            "latest_bmi": "150.0",  # Unreasonably high
            "conditions": ["Hypertension"],
            "meds": [],
        })
        assert result.verdict == "DENIED_MISSING_INFO"
    
    def test_bmi_parsing_from_various_formats(self):
        """BMI should be correctly parsed from various formats."""
        test_cases = [
            ("32.4", 32.4),
            ("32.4 (Source: EMR)", 32.4),
            ("28.1 (Calculated)", 28.1),
            ("30", 30.0),
        ]
        for bmi_str, expected in test_cases:
            result = run_eval({
                "latest_bmi": bmi_str,
                "conditions": [],
                "meds": [],
            })
            assert result.bmi_numeric == expected, f"Failed to parse '{bmi_str}'"


class TestCaseInsensitivity:
    """Verify case-insensitive matching for conditions and medications."""
    
    @pytest.mark.parametrize("condition", [
        "hypertension",
        "HYPERTENSION",
        "Hypertension",
        "HyPeRtEnSiOn",
    ])
    def test_condition_case_insensitive(self, condition):
        """Condition matching should be case-insensitive."""
        result = run_eval({
            "latest_bmi": "28.0",
            "conditions": [condition],
            "meds": [],
        })
        assert result.verdict == "APPROVED"
    
    @pytest.mark.parametrize("med", [
        "OZEMPIC",
        "ozempic",
        "Ozempic",
    ])
    def test_medication_case_insensitive(self, med):
        """Medication matching should be case-insensitive."""
        result = run_eval({
            "latest_bmi": "32.0",
            "conditions": [],
            "meds": [med],
        })
        assert result.verdict == "DENIED_SAFETY"


class TestComplexScenarios:
    """Test complex real-world scenarios with multiple conditions."""
    
    def test_multiple_comorbidities_approves(self):
        """Multiple valid comorbidities should still approve."""
        result = run_eval({
            "latest_bmi": "28.5",
            "conditions": ["Hypertension", "Type 2 Diabetes", "Dyslipidemia"],
            "meds": [],
        })
        assert result.verdict == "APPROVED"
    
    def test_safety_exclusion_overrides_comorbidities(self):
        """Safety exclusion should override any comorbidities."""
        result = run_eval({
            "latest_bmi": "35.0",
            "conditions": ["Hypertension", "Type 2 Diabetes", "MTC"],
            "meds": [],
        })
        assert result.verdict == "DENIED_SAFETY"
    
    def test_ambiguous_and_valid_term_uses_valid(self):
        """If both ambiguous and valid comorbidity present, should approve."""
        result = run_eval({
            "latest_bmi": "28.0",
            "conditions": ["Prediabetes", "Hypertension"],  # Ambiguous + Valid
            "meds": [],
        })
        assert result.verdict == "APPROVED"
        assert result.comorbidity_category == "HYPERTENSION"
    
    def test_obese_bmi_overrides_need_for_comorbidity(self):
        """BMI ≥ 30 should approve regardless of comorbidity presence."""
        result = run_eval({
            "latest_bmi": "30.0",
            "conditions": [],  # No comorbidities
            "meds": [],
        })
        assert result.verdict == "APPROVED"
        assert result.comorbidity_category == "NONE"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

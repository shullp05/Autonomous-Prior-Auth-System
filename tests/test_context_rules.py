"""
test_context_rules.py - Verify 9.2 context classification logic
"""

import pytest

from context_rules import classify_context


class TestContextRules:

    # 1. NEGATED (Strongest)
    @pytest.mark.parametrize("text", [
        "Patient denies history of MTC",
        "MTC ruled out",
        "No evidence of MEN2",
        "Negative for medullary thyroid cancer",
        "denies previous pancreatitis",
        "History of MTC: Negative",
        "MTC: Not detected"
    ])
    def test_negated(self, text):
        assert classify_context(text, "MTC") == "NEGATED"

    # 2. FAMILY HISTORY
    @pytest.mark.parametrize("text", [
        "Family history of MTC",
        "Mother had MTC",
        "Father has MEN2",
        "Brother with thyroid cancer",
        "Grandparent had pancreatitis",
        "Strong fam hx of MTC",
        "FHx: MTC"
    ])
    def test_family_history(self, text):
        assert classify_context(text, "MTC") == "FAMILY_HISTORY"

    # 3. HISTORICAL
    @pytest.mark.parametrize("text", [
        "History of pancreatitis",
        "Hx of suicidality",
        "Prior MTC",
        "Past medical history: pancreatitis",
        "Remote history of suicide attempt",
        "Status post thyroidectomy for MTC",
        "s/p pancreatitis",
        "Postpartum thyroiditis"  # Contains 'postpartum' -> Historical context for pregnancy? Or generally past.
    ])
    def test_historical(self, text):
        # Note: "Postpartum" effectively means pregnancy is over/historical
        assert classify_context(text, "MTC") == "HISTORICAL"

    # 4. HYPOTHETICAL
    @pytest.mark.parametrize("text", [
        "Possible MTC",
        "Concern for MEN2",
        "Monitor for pancreatitis",
        "Risk of suicidality",
        "Evaluate for thyroid cancer",
        "Probable MTC",
        "Suspected MEN2"
    ])
    def test_hypothetical(self, text):
        assert classify_context(text, "MTC") == "HYPOTHETICAL"

    # 5. ACTIVE (Default / No Modifiers)
    @pytest.mark.parametrize("text", [
        "MTC detected",
        "Diagnosis: MTC",
        "Active pancreatitis",
        "Patient has MEN2",
        "Pregnant",
        "Breastfeeding",
        "Contraindication: MTC",
        "Assessment: Medullary thyroid carcinoma"
    ])
    def test_active(self, text):
        assert classify_context(text, "MTC") == "ACTIVE"

    # Edge cases / Conflicts
    def test_precedence_negation_over_historical(self):
        # "Patient denies history of MTC" -> Should be NEGATED, not Historical
        assert classify_context("Patient denies history of MTC", "MTC") == "NEGATED"

    def test_precedence_family_over_historical(self):
        # "Family history of MTC" -> Should be FAMILY, not Historical
        assert classify_context("Family history of MTC", "MTC") == "FAMILY_HISTORY"

    def test_complex_phrasing(self):
        # "No family history of MTC" -> Negation + Family?
        # Actually our rule checks negation first. "No ..." matches negation.
        # Ideally "No family history" means NO RISK. So NEGATED is correct outcome (not active).
        assert classify_context("No family history of MTC", "MTC") == "NEGATED"

    def test_current_pregnancy(self):
        assert classify_context("Patient is pregnant (12 weeks)", "PREGNANCY") == "ACTIVE"

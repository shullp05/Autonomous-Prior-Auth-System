
from governance_audit import _detect_safety_exclusion, _is_cvd, _is_t2_diabetes, two_proportion_z_pvalue, wilson_ci


def test_governance_safety_exclusion_mtc():
    # MTC
    assert _detect_safety_exclusion(["medullary thyroid carcinoma"], []) == (True, "medullary thyroid carcinoma")
    assert _detect_safety_exclusion(["headache"], []) == (False, "")

def test_governance_safety_exclusion_pregnancy():
    # Pregnancy
    # Pregnancy terms are checked against CONDITIONS, not MEDS in the implementation
    assert _detect_safety_exclusion(["pregnant"], []) == (True, "pregnant")
    assert _detect_safety_exclusion(["headache"], ["pregnancy"]) == (False, "")

def test_governance_cvd_detection():
    # CVD terms via policy_constants
    assert _is_cvd(["myocardial infarction"]) == "myocardial infarction"
    assert _is_cvd(["MI"]) == "MI" # Abbreviation boundary check
    assert _is_cvd(["migraine"]) is None # "mi" inside migraine should not match if boundary works

def test_governance_t2dm_detection():
    assert _is_t2_diabetes(["Type 2 Diabetes Mellitus"]) == "Type 2 Diabetes Mellitus"
    assert _is_t2_diabetes(["T2DM"]) == "T2DM"
    assert _is_t2_diabetes(["Type 1 Diabetes"]) is None


# --- Statistical Function Edge Cases ---

class TestWilsonCI:
    """Edge case tests for Wilson confidence interval calculation."""

    def test_wilson_ci_zero_successes(self):
        """k=0 should produce valid CI starting at 0."""
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert 0 < hi < 0.05

    def test_wilson_ci_all_successes(self):
        """k=n should produce valid CI ending at 1.0."""
        lo, hi = wilson_ci(100, 100)
        assert 0.95 < lo < 1.0
        assert hi == 1.0

    def test_wilson_ci_empty_sample(self):
        """n=0 should return NaN."""
        import math
        lo, hi = wilson_ci(0, 0)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_wilson_ci_small_sample(self):
        """Small sample (n=5) should still produce valid bounds."""
        lo, hi = wilson_ci(2, 5)
        assert 0 <= lo < hi <= 1


class TestTwoProportionZPvalue:
    """Edge case tests for two-proportion z-test."""

    def test_identical_proportions(self):
        """Identical proportions should give p-value close to 1."""
        pval = two_proportion_z_pvalue(50, 100, 50, 100)
        assert pval is not None
        assert pval >= 0.99

    def test_significantly_different(self):
        """Clearly different proportions should give low p-value."""
        pval = two_proportion_z_pvalue(90, 100, 10, 100)
        assert pval is not None
        assert pval < 0.001

    def test_empty_groups(self):
        """Empty group should return None."""
        assert two_proportion_z_pvalue(0, 0, 50, 100) is None
        assert two_proportion_z_pvalue(50, 100, 0, 0) is None

    def test_all_or_none_success(self):
        """Edge case: all successes or all failures."""
        pval = two_proportion_z_pvalue(100, 100, 0, 100)
        # When pooled proportion is 0.5 and groups are maximally different
        assert pval is not None
        assert pval < 0.001


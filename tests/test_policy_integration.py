
import pytest
from governance_audit import _detect_safety_exclusion, _is_cvd, _is_t2_diabetes
from policy_utils import normalize

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


from policy_engine import evaluate_eligibility


# T01_Strict_Neg: BMI 32.0, Missing codes -> FLAG/CDI_REQUIRED
def test_t01_strict_neg():
    patient_data = {
        "latest_bmi": "32.0",
        "conditions": [
            {"condition_name": "Obesity", "icd10_dx": None, "icd10_bmi": None}
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    # Expect CDI_REQUIRED because clinically eligible (BMI>30) but admin not ready
    assert result.verdict == "CDI_REQUIRED"
    assert "Missing explicit ICD-10" in str(result.missing_anchor_code)

# T02_Strict_Pos: BMI 32.0, Present Codes -> APPROVED
def test_t02_strict_pos():
    patient_data = {
        "latest_bmi": "32.0",
        "conditions": [
            {"condition_name": "Obesity", "icd10_dx": "E66.9", "icd10_bmi": "Z68.32"}
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "APPROVED"
    assert result.admin_ready is True

# T03_Underweight: BMI 26.9 -> DENIED (Use Clinical Logic)
def test_t03_underweight():
    patient_data = {
        "latest_bmi": "26.9",
        "conditions": [
            {"condition_name": "Obesity", "icd10_dx": "E66.9", "icd10_bmi": "Z68.26"}
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "DENIED_CLINICAL"

# T04_Overweight_Fail: BMI 28.0, No Comorbidity -> DENIED
def test_t04_overweight_fail():
    patient_data = {
        "latest_bmi": "28.0",
        "conditions": [
            {"condition_name": "Overweight", "icd10_dx": "E66.3", "icd10_bmi": "Z68.28"}
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    # Rejection because Overweight alone isn't enough, need comorbidity
    assert result.verdict == "DENIED_CLINICAL"

# T05_Overweight_Pass: BMI 28.0, HTN -> APPROVED
def test_t05_overweight_pass():
    patient_data = {
        "latest_bmi": "28.0",
        "conditions": [
            {"condition_name": "Overweight", "icd10_dx": "E66.3", "icd10_bmi": "Z68.28"},
            {"condition_name": "Essential hypertension", "icd10_dx": None, "icd10_bmi": None} # HTN doesn't need codes?
            # Wait, check_admin_readiness for BMI27_COMORBIDITY checks for Overweight Anchor.
            # It iterates ALL conditions. One should satisfy validate_obesity_condition.
            # The "Overweight" condition satisfies it.
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "APPROVED"
    assert result.comorbidity_category == "HYPERTENSION"

# T06_Ambiguity: BMI 28.0, Prediabetes -> MANUAL_REVIEW
def test_t06_ambiguity():
    patient_data = {
        "latest_bmi": "28.0",
        "conditions": [
            {"condition_name": "Prediabetes", "icd10_dx": None, "icd10_bmi": None},
            {"condition_name": "Overweight", "icd10_dx": "E66.3", "icd10_bmi": "Z68.28"}
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "MANUAL_REVIEW"
    assert result.decision_type == "FLAGGED_AMBIGUITY"

# T07_Safety_Stop: BMI 35.0, Pregnancy -> DENIED_SAFETY
def test_t07_safety_stop():
    patient_data = {
        "latest_bmi": "35.0",
        "conditions": [
            {"condition_name": "Obesity", "icd10_dx": "E66.9", "icd10_bmi": "Z68.35"},
            {"condition_name": "Pregnancy", "icd10_dx": None, "icd10_bmi": None}
        ],
        "meds": []
    }
    result = evaluate_eligibility(patient_data)
    assert result.verdict == "DENIED_SAFETY"


from policy_engine import evaluate_eligibility


def test_dict_conditions_capture_anchor_codes_obesity():
    patient_data = {
        "latest_bmi": "35.0",
        "conditions": [
            {
                "condition_name": "Obesity",
                "icd10_dx": "E66.9",
                "icd10_bmi": "Z68.35",
            }
        ],
        "meds": [],
    }

    result = evaluate_eligibility(patient_data)

    assert result.verdict == "APPROVED"
    assert result.admin_ready is True
    assert result.found_e66_code == "E66.9"
    assert result.found_z68_code == "Z68.35"


def test_dict_conditions_capture_anchor_codes_overweight():
    patient_data = {
        "latest_bmi": "28.4",
        "conditions": [
            {
                "condition_name": "Overweight",
                "icd10_dx": "E66.3",
                "icd10_bmi": "Z68.28",
            },
            {
                "condition_name": "Hypertension",
                "icd10_dx": None,
                "icd10_bmi": None,
            },
        ],
        "meds": [],
    }

    result = evaluate_eligibility(patient_data)

    assert result.verdict == "APPROVED"
    assert result.admin_ready is True
    assert result.found_e66_code == "E66.3"
    assert result.found_z68_code == "Z68.28"


def test_dict_conditions_missing_e66_triggers_cdi():
    patient_data = {
        "latest_bmi": "35.0",
        "conditions": [
            {
                "condition_name": "Obesity",
                "icd10_dx": None,
                "icd10_bmi": "Z68.35",
            }
        ],
        "meds": [],
    }

    result = evaluate_eligibility(patient_data)

    assert result.verdict == "CDI_REQUIRED"
    assert result.admin_ready is False
    assert result.missing_anchor_code == "E66.9"
    assert result.found_z68_code == "Z68.35"


def test_dict_conditions_missing_z68_triggers_cdi():
    patient_data = {
        "latest_bmi": "28.4",
        "conditions": [
            {
                "condition_name": "Overweight",
                "icd10_dx": "E66.3",
                "icd10_bmi": None,
            },
            {
                "condition_name": "Hypertension",
                "icd10_dx": None,
                "icd10_bmi": None,
            },
        ],
        "meds": [],
    }

    result = evaluate_eligibility(patient_data)

    assert result.verdict == "CDI_REQUIRED"
    assert result.admin_ready is False
    assert result.missing_anchor_code == "Z68.x"
    assert result.found_e66_code == "E66.3"

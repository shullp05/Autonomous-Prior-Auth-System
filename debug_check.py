
import pandas as pd

from config import VALID_BMI_Z_PREFIX, VALID_OBESITY_DX_CODES


# Mocking the logic content for reproduction
def check_row(row):
    print("--- Checking Row ---")
    print(f"Raw Row: {row}")

    name = str(row.get("condition_name", "")).upper()
    dx = str(row.get("icd10_dx", "")).strip()
    bmi_code = str(row.get("icd10_bmi", "")).strip()

    print(f"Name (Upper): '{name}'")
    print(f"DX (Strip): '{dx}'")
    print(f"BMI Code (Strip): '{bmi_code}'")

    # 1. Text Check
    has_text = "OBESITY" in name or "OVERWEIGHT" in name
    print(f"Has Text (OBESITY/OVERWEIGHT): {has_text}")

    # 2. DX Code Check
    # Verify explicitly against set
    valid_dx = dx in VALID_OBESITY_DX_CODES
    print(f"Valid DX ({dx} in {list(VALID_OBESITY_DX_CODES)[:3]}...): {valid_dx}")

    # 3. BMI Code Check
    valid_bmi = bmi_code.startswith(VALID_BMI_Z_PREFIX)
    print(f"Valid BMI ({bmi_code} starts with {VALID_BMI_Z_PREFIX}): {valid_bmi}")

    result = has_text and valid_dx and valid_bmi
    print(f"Triple-Key Result: {result}")
    return result

def run_debug():
    try:
        df = pd.read_csv("output/data_conditions.csv")
    except FileNotFoundError:
        print("CSV not found.")
        return

    pid = "e53d9998-b3db-eb20-e78d-76dab38ded12"
    print(f"Loading conditions for {pid}...")

    # Filter
    rows = df[df["patient_id"] == pid].to_dict(orient="records")

    if not rows:
        print("No conditions found for patient!")
        return

    found_valid = False
    for row in rows:
        if check_row(row):
            found_valid = True

    print(f"\nFinal Verdict: {'APPROVED' if found_valid else 'CDI_REQUIRED'}")

if __name__ == "__main__":
    run_debug()

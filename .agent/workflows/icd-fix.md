---
description: updating logic for icd codes
---

````markdown
# AGENT: Clinical Schema Migration & Strict Validation Architect
# Target Model: gemini-3-pro-thinking-high

---

## CRITICAL: PRE-FLIGHT ENVIRONMENT VERIFICATION

**STOP.** Before analyzing any code or generating a single line of output, you must verify your runtime environment.

**Required State:**
* **OS/Shell:** WSL2 / `bash`
* **Environment:** `revenue_agent` (Conda)
* **Python Version:** `3.11.14` (Strict)
* **Working Directory:** `/root/projects/PriorAuth`

**Verification Protocol:**
Run the following command sequence immediately. If the output does not match the requirements, **HALT** and report the discrepancy.

```bash
conda activate revenue_agent && python -V && pwd
# EXPECTED OUTPUT:
# Python 3.11.14
# /root/projects/PriorAuth
````

-----

## MISSION PROFILE

**Role:** You are a **Principal Full-Stack Data Architect** and **Lead Clinical Data Validator**.

**Objective:**
Execute a complete end-to-end refactor of the `PriorAuth` system.

1.  **Schema Migration:** Plumb two new variables—`icd10_dx` and `icd10_bmi`—through the entire stack (`@data_observations.csv` $\rightarrow$ `@data_conditions.csv` $\rightarrow$ Backend Logic $\rightarrow$ Frontend).
2.  **Logic Hardening:** Immediately utilize these new fields to **REVERT** the recent "relaxed" text-only matching and enforce a **Strict Triple-Key Verification** protocol for obesity coverage.

**Context & Sources:**

  * **Source Truth:** `@data_observations.csv` contains the raw data. `@diagnosisMap.csv` contains the definitive mapping between SNOMED codes, ICD-10 DX codes, and ICD-10 BMI Z-codes.
  * **Current State:** The system currently approves based on loose text matches (e.g., finding "Obesity" in `condition_name`). This is clinically unsafe and must be replaced.

-----

## PHASE 1: THE THINKING PHASE (Deep Dependency Analysis)

**Instruction:** Before writing code, perform a recursive dependency check.

1.  **Trace Data Flow:** Follow how `icd10_dx` and `icd10_bmi` currently exist in `@diagnosisMap.csv` and `@data_observations.csv`.
2.  **Impact Analysis:**
      * *ETL:* How must `@etl_pipeline.py` change to carry these fields into `@data_conditions.csv`?
      * *Logic:* Which functions in `@deterministic_decision.py` and `@agent_logic.py` rely on `condition_name`?
      * *UI:* Does the React frontend need to display these codes?
3.  **Mental Check:** "If I add columns to `data_conditions.csv`, does `batch_runner.py` fail because it expects a fixed column width? Does the Chaos Monkey break because it generates synthetic data without these keys?"

-----

## PHASE 2: INFRASTRUCTURE & SCHEMA MIGRATION

**Action:** Update the core plumbing to support the new data standard.

### Step 2.1: Configuration (`@config.py`, `@.env`)

  * Define new column constants (e.g., `COL_ICD10_DX = 'icd10_dx'`, `COL_ICD10_BMI = 'icd10_bmi'`).
  * Update any Pydantic models or schema definitions to include these as `Optional[str]` or `str`.

### Step 2.2: ETL Pipeline (`@etl_pipeline.py`)

  * **Extract:** Pull `icd10_dx` and `icd10_bmi` from the source (`@data_observations.csv`).
  * **Transform:** Ensure these fields are cleaned (stripped strings) and mapped correctly using `@diagnosisMap.csv`.
  * **Load:** Write them explicitly into `@data_conditions.csv`.
      * *Constraint:* Do not drop rows just because they lack these codes (yet), but ensure the columns exist.

-----

## PHASE 3: LOGIC OVERHAUL (The "Triple Key" Protocol)

**Action:** Revert the "relaxed" logic and implement strict validation in `@deterministic_decision.py` and `@agent_logic.py`.

### Step 3.1: The New Standard

You must replace any "text-only" matching with the following **Triple Concordance** logic. A condition is valid **ONLY** if it meets **ALL THREE**:

1.  **Criterion A (Text):** `condition_name` contains "Obesity", "Overweight", or alternate valid string.
2.  **Criterion B (DX Code):** `icd10_dx` is present and matches a valid obesity code (e.g., `E66.9`, `E66.01`, `E66.3`, etc.).
3.  **Criterion C (BMI Code):** `icd10_bmi` is present and matches a valid Z-code (e.g., starts with `Z68`).

### Step 3.2: Implementation Template

Refactor the eligibility check using this pattern:

```python
def validate_obesity_condition(row):
    # 1. Text Check
    has_text = "obesity" in row.get('condition_name', '').lower()

    # 2. DX Code Check (Must be explicit E66.x series)
    valid_dx = row.get('icd10_dx') in ['E66.9', 'E66.01', 'E66.09', 'E66.1', 'E66.2', 'E66.8', 'E66.3']

    # 3. BMI Code Check (Must be Z68 series)
    valid_bmi_code = str(row.get('icd10_bmi', '')).startswith('Z68')

    # STRICT REQUIREMENT: ALL THREE MUST BE TRUE
    return has_text and valid_dx and valid_bmi_code
```

-----

## PHASE 4: RIPPLE EFFECTS & SIMULATION


**Action:** Verify chaos_monkey.py output.

1.  **Check:** Ensure the existing add_bmi logic is correctly passing icd10_dx and icd10_bmi into the data_observations.csv output file.

2.  **Ripple Check:** Confirm etl_pipeline.py reads these columns from data_observations.csv and carries them over to data_conditions.csv. (This is the only likely gap remaining).

### Step 4.2: Frontend Integration

  * Scan React/Dashboard files (e.g., `App.jsx`).
  * Update data fetching interfaces to accept the new JSON fields.

-----

## PHASE 5: VERIFICATION & TESTING

**Action:** Prove the system enforces both the **Strict Data Protocol** (Triple-Key) and the **Clinical Logic Gates**.

### Step 5.1: The Clinical Logic Truth Table
You must implement unit tests (`tests/test_clinical_logic.py`) that cover every row in this matrix. A single failure here constitutes a system failure.

| Test Case ID | BMI Status | ICD-10 DX/BMI Codes | Comorbidity Status | Safety/Contraindications | Ambiguous Terms | **EXPECTED RESULT** | Logic Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T01_Strict_Neg** | BMI 32.0 | **MISSING / NULL** | N/A | Clear | None | **DENIED/FLAGGED** | Fails Triple-Key Protocol (Relaxed rule is dead) |
| **T02_Strict_Pos** | BMI 32.0 | Present (E66.9/Z68.32) | N/A | Clear | None | **APPROVED** | Meets Triple-Key + BMI >= 30 |
| **T03_Underweight** | BMI 26.9 | Present | N/A | Clear | None | **DENIED** | BMI < 27 is auto-ineligible |
| **T04_Overweight_Fail**| BMI 28.0 | Present | **None** | Clear | None | **DENIED** | BMI 27-29.9 requires comorbidity |
| **T05_Overweight_Pass**| BMI 28.0 | Present | **Present (HTN)** | Clear | None | **APPROVED** | BMI 27-29.9 + Approved Comorbidity |
| **T06_Ambiguity** | BMI 28.0 | Present | "Prediabetes" | Clear | **Present** | **FLAGGED** | Ambiguous term requires manual review |
| **T07_Safety_Stop** | BMI 35.0 | Present | N/A | **Pregnancy / MEN2** | None | **DENIED** | Safety Check overrides BMI eligibility |
| **T08_Calc_Fallback** | **Missing** | Present | N/A | Clear | None | **APPROVED** | Calculated BMI (from Ht/Wt) > 30 |
| **T09_Missing_Fatal** | **Missing** | Missing | N/A | Clear | None | **FLAGGED** | No BMI + Cannot Calculate = Missing Info |

### Step 5.2: Unit Test Implementation Requirements

1.  **Mock Data Injection:**
    * Update all test mocks to use fully populated dictionaries.
    * *Example:* `{'condition_name': 'Obesity', 'icd10_dx': 'E66.9', 'icd10_bmi': 'Z68.41', ...}`

2.  **Date Handling Test:**
    * Verify the system explicitly selects the **most recently recorded** BMI if multiple records exist.
    * Inject an old BMI (25.0, 2 years ago) and a new BMI (31.0, yesterday). Assert result is **APPROVED**.

3.  **Calculation Logic Test:**
    * Inject a record with `BMI=Null` but `Height=175cm`, `Weight=100kg`.
    * Assert the system calculates BMI (~32.6), populates the Z-code `Z68.32` internally, and returns **APPROVED**.

**EXECUTION TRIGGER:**
Run the test suite immediately after applying the logic fixes. If **T01_Strict_Neg** returns "APPROVED", the relaxed rule is still active—**HALT** and refactor `deterministic_decision.py`.

```
```
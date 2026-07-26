# Governance

## Purpose
Governance evaluates whether the system treats protected groups consistently for clinically eligible cases. The current focus is **equal opportunity** via false negative rate (FNR) parity.

## Methodology (current implementation)
Implemented in `src/priorauth/governance_audit.py` and run against `output/dashboard_data.json` plus the supporting CSVs in `output/`.

- **Ground truth**: Deterministic policy logic (BMI + qualifying comorbidity + safety exclusions) derived from `src/priorauth/policy_constants.py`.
- **Metric**: FNR parity among *truly eligible* patients.
  - `fn_access`: eligible patients not marked APPROVED (includes DENIED/FLAGGED/etc.).
  - `fn_denied_only`: eligible patients whose outcome starts with DENIED.
- **Statistical checks**:
  - Wilson 95% confidence intervals for rates.
  - Two-proportion z-tests with Bonferroni correction across groups.
- **Attributes audited**: `race` and `gender` (from patient data).
- **Intersectional audit**: `race × gender`, with the same minimum-sample and uncertainty controls.
- **Calibration**: Brier score and confidence bins when results include numeric `decision_confidence`.
- **Drift**: Status-distribution deltas when `PA_GOVERNANCE_BASELINE_PATH` points to a locked prior result set.
- **Clinically reviewed truth**: `PA_CLINICAL_GROUND_TRUTH_PATH` may supply reviewer-attributed prospective labels that supersede synthetic labels for matching cases.
- **Stop-ship rule**: A group is flagged if disparity exceeds the configured threshold and is statistically significant.

## Inputs and outputs
- Inputs:
  - `output/dashboard_data.json`
  - `output/data_patients.csv`
  - `output/data_observations.csv`
  - `output/data_conditions.csv`
  - `output/data_medications.csv`
- Output:
  - `output/governance_report.json`

## How to run
```bash
python -m priorauth.governance_audit
```

## Limitations
- The audit depends on **synthetic or staged data**; results should not be interpreted as real-world bias measurements.
- Cases with **unknown ground truth** (e.g., missing BMI) are excluded from eligible denominators.
- The parity checks do not account for clinical confounders or distribution shifts.
- Small groups are retained with confidence intervals and marked insufficient rather than treated as evidence of parity.
- Calibration is reported only when callers provide numeric confidence; the audit still does not establish causal fairness or clinical utility.

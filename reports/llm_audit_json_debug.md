# LLM Audit JSON Debug (qwen3, mistral)

Date: 2025-12-31
Updated: 2026-01-01

## Goal
Investigate advisory audit JSON parse warnings observed during benchmark runs.

## Methods
Ran targeted Python calls to `agent_logic._make_llm()` with
`PA_AUDIT_MODEL_FLAVOR` set to `qwen3` and `mistral`. Used the same
system/user prompts as `clinical_audit`, with both short and full policy
evidence.

## Findings
- qwen3 (full policy evidence) returned JSON with a nested `reasoning` object
  and additional keys (e.g., `policy_criteria`, `deterministic_decision`).
  This does not match `AuditResult` (expects `reasoning` as a string) and can
  trigger validation errors.
- qwen3 (short policy evidence) returned JSON matching the expected schema.
- mistral (short policy evidence) returned a JSON echo of the input payload
  (top-level keys `policy_evidence`, `patient`, `deterministic_decision_source_of_truth`,
  `output_schema`) rather than the requested schema.
- mistral (full policy evidence) returned a partial schema
  (`verdict`, `bmi_numeric`, `safety_flag`, `comorbidity_category`) with other
  fields omitted; Pydantic defaults will fill missing fields but the output is
  low-signal.
- The exact parse error `Expecting value: line 1 column 1 (char 0)` was not
  reproduced in these targeted calls; likely occurs when the model emits
  non-JSON or empty content in some runs.

## Schema-Enforced Spot Check
After switching to JSON schema formatting via `format=AUDIT_RESULT_SCHEMA`,
both qwen3 and mistral produced JSON matching the expected `AuditResult`
shape in spot tests.

- qwen3 output keys: `bmi_numeric`, `safety_flag`, `comorbidity_category`,
  `evidence_quoted`, `verdict`, `reasoning`
- mistral output keys: `bmi_numeric`, `safety_flag`, `comorbidity_category`,
  `evidence_quoted`, `verdict`, `reasoning`

## Benchmark Follow-up
During qwen3 sample=50 benchmark runs on 2025-12-31 and 2026-01-01, advisory
audit parse warnings (`Expecting value: line 1 column 1 (char 0)`) still
appeared in the console output despite schema enforcement, indicating
occasional empty or non-JSON responses can still occur at runtime.

## Notes
- These advisory outputs do not affect deterministic outcomes; they only impact
  `llm_*` fields in audit findings.

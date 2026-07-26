# Prospective Clinical Validation Protocol

The repository's synthetic and retrospective tests establish software behavior, not clinical validity, regulatory clearance, or fitness for autonomous patient-care use.

Before production use, a protocol approved by clinical, compliance, privacy, security, and statistical owners must prospectively evaluate consecutive cases without influencing care. At minimum, it must pre-register the intended population, endpoints, subgroup analyses, sample size, stopping rules, adjudication process, and acceptance thresholds.

Each reference label must include `patient_id`, `eligible`, `reviewer_id`, and `reviewed_at`. Two independent qualified reviewers should adjudicate disagreements while blinded to model output. The locked system version, policy hash, model version, input provenance, exclusions, missingness, overrides, and deviations must be retained.

Required endpoints include sensitivity, specificity, false-negative rate, calibration, abstention/manual-review rate, safety-event recall, inter-rater agreement, intersectional subgroup performance, and operational time. Confidence intervals and results for small groups must be reported without suppressing uncertainty.

Deployment remains prohibited until the pre-registered thresholds are met and an accountable medical director approves the validation report. Post-deployment monitoring must include drift thresholds, periodic revalidation, incident review, and a documented rollback and stop-ship procedure.

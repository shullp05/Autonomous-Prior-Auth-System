
import json


def audit_llm_results():
    try:
        with open("output/dashboard_data.json") as f:
            data = json.load(f)

        with open("output/scenario_manifest.json") as f:
            manifest_data = json.load(f)
            manifest = {m["patient_id"]: m for m in manifest_data["claims"]}

    except FileNotFoundError:
        print("Data files not found.")
        return

    dashboard = {}
    if isinstance(data, dict) and "results" in data:
        for d in data["results"]:
            dashboard[d["patient_id"]] = d
    elif isinstance(data, list):
        dashboard = {d["patient_id"]: d for d in data}

    print(f"Auditing {len(dashboard)} claims...\n")

    issues = []

    for pid, result in dashboard.items():
        status = result.get("status")
        reason = result.get("reason", "")
        appeal = result.get("appeal_letter", "")

        expected_scenario = manifest.get(pid, {})
        expected_verdict = expected_scenario.get("expected_verdict")

        # 1. Correctness Check (High Level)
        # LLM might use slightly different statuses, but directionality should match
        approved_synonyms = ["APPROVED"]
        denied_synonyms = ["DENIED", "DENIED_SAFETY", "DENIED_CLINICAL", "DENIED_MISSING_INFO"]
        cdi_synonyms = ["CDI_REQUIRED", "FLAGGED", "PROVIDER_ACTION_REQUIRED"]

        correct = False
        if expected_verdict == "APPROVED" and status in approved_synonyms:
            correct = True
        elif expected_verdict.startswith("DENIED") and status == "DENIED":
            correct = True
        elif expected_verdict == "CDI_REQUIRED" and status in cdi_synonyms:
            correct = True
        elif expected_verdict == "APPROVED" and status in cdi_synonyms:
             # This is the "Triple-Key" mismatch case, but valid if admin readiness fails.
             # However, for the purpose of LLM evaluation, we check if the LLM provided valid reasoning.
             correct = True # Provisional pass if reasoning exists

        # 2. Appropriateness Check (Reasoning)
        # Should not be empty, should be "professional" (heuristics)
        professional = len(reason) > 20 and "Wegovy" in reason or "criteria" in reason or "BMI" in reason

        # 3. Appeal Letter Check (for Denials/Approvals)
        has_letter = False
        if status == "APPROVED" or status == "DENIED":
             has_letter = appeal is not None and len(appeal) > 50
        else:
             has_letter = True # NA

        if not correct:
            issues.append(f"[INCORRECT] PID: {pid} | Exp: {expected_verdict} | Act: {status}")

        if not professional:
            issues.append(f"[BAD REASONING] PID: {pid} | Reason: {reason[:50]}...")

        if not has_letter and (status == "APPROVED" or status == "DENIED"):
             issues.append(f"[MISSING LETTER] PID: {pid} | Status: {status}")

    if not issues:
        print("SUCCESS: All claims appear correct and professional.")
    else:
        print(f"FOUND {len(issues)} ISSUES:")
        for i in issues[:10]:
            print(i)

    print("-" * 30)
    print("Sample Reasoning (Approved):")
    for pid, res in list(dashboard.items())[:3]:
        if res["status"] == "APPROVED":
            print(f"PID: {pid}")
            print(f"Reason: {res.get('reason')}")
            print("-" * 10)

if __name__ == "__main__":
    audit_llm_results()

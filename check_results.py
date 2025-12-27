
import json


def check_results():
    try:
        with open("output/dashboard_data.json") as f:
            data = json.load(f)

        with open("output/scenario_manifest.json") as f:
            manifest = json.load(f)

    except FileNotFoundError:
        print("Data files not found.")
        return

    dashboard = {}
    if isinstance(data, dict) and "results" in data:
        for d in data["results"]:
            dashboard[d["patient_id"]] = d
    elif isinstance(data, list):
        dashboard = {d["patient_id"]: d for d in data}
    else:
        print("Unknown dashboard data format.")
        return
    scenarios = manifest["claims"]

    print(f"Total Claims in Dashboard: {len(dashboard)}")
    print(f"Total Scenarios: {len(scenarios)}")

    approved_count = 0
    cdi_count = 0
    mismatch_count = 0

    for s in scenarios:
        pid = s["patient_id"]
        expected = s["expected_verdict"]

        if pid not in dashboard:
            print(f"WARNING: Patient {pid} not in dashboard results.")
            continue

        result = dashboard[pid]
        status = result.get("status")
        verdict = result.get("raw_verdict")

        if status == "APPROVED":
            approved_count += 1
        elif status == "CDI_REQUIRED":
            cdi_count += 1

        # Comparison logic
        # expected_verdict in scenarios is typically "APPROVED", "DENIED_SAFETY", etc.
        # Dashboard outcome mapping

        match = False
        if expected == "APPROVED" and status == "APPROVED":
            match = True
        elif expected == "DENIED_SAFETY" and status == "DENIED":
            match = True
        elif expected == "DENIED_CLINICAL" and status == "DENIED":
            match = True # Or check reason
        elif expected.startswith("DENIED") and status == "DENIED":
             match = True

        if not match:
            # Check specifically for APPROVED vs CDI_REQUIRED mismatch
            if expected == "APPROVED" and status == "CDI_REQUIRED":
                mismatch_count += 1
                if mismatch_count <= 5:
                    print(f"Mismatch {mismatch_count}: Patient {pid}")
                    print(f"  Expected: {expected}, Got: {status}")
                    print(f"  Reason: {result.get('reason')}")
                    print(f"  Policy Path: {result.get('policy_path')}")

    print("-" * 30)
    print("Stats:")
    print(f"  APPROVED: {approved_count}")
    print(f"  CDI_REQUIRED: {cdi_count}")
    print(f"  Mismatches (Expected APPROVED, Got CDI): {mismatch_count}")

    if mismatch_count == 0 and approved_count > 0:
        print("SUCCESS: All expected approvals matched.")
    elif approved_count == 0:
        print("FAILURE: No approvals found.")

if __name__ == "__main__":
    check_results()

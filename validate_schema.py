
import json


def validate():
    try:
        with open("output/dashboard_data.json") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    results = data.get("results", [])
    if not results:
        print("No results found.")
        return

    errors = []

    # Check results items
    for i, res in enumerate(results):
        pid = res.get("patient_id", f"Index {i}")

        # Check required fields
        if "duration_ms" not in res:
            errors.append(f"{pid}: Missing 'duration_ms'")
        elif not isinstance(res["duration_ms"], int):
            errors.append(f"{pid}: 'duration_ms' is not integer (Got {type(res['duration_ms'])})")

        if "duration_sec" in res:
            errors.append(f"{pid}: Found 'duration_sec' (should be removed)")

    # Check metadata
    meta = data.get("metadata", {})
    if "avg_duration_ms" in meta:
        # Check if it makes sense (should be around 2000-3000 for approved, 20 for denied)
        val = meta["avg_duration_ms"]
        print(f"Metadata avg_duration_ms: {val}")

    if "cdi_required_count" not in meta:
        errors.append("Metadata: Missing 'cdi_required_count'")
    if "revenue_at_risk_usd" not in meta:
        errors.append("Metadata: Missing 'revenue_at_risk_usd'")

    if errors:
        print(f"FAILED: Found {len(errors)} errors.")
        for e in errors[:5]:
            print(e)
    else:
        print("SUCCESS: usage of duration_ms and new CDI/Risk metrics verified.")

if __name__ == "__main__":
    validate()

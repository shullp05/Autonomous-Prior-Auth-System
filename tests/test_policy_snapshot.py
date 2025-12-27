import hashlib
import json
from pathlib import Path

from policy_engine import evaluate_eligibility
from policy_snapshot import GUIDELINE_PATH, SNAPSHOT_PATH, canonical_dumps, parse_guidelines
from schema_validation import validate_policy_snapshot

GOLDEN_PATH = Path("tests/golden/policies/RX-WEG-2025.json")


def _canonical_json(data: dict) -> dict:
    return json.loads(canonical_dumps(data))


def _current_guideline_hash() -> str:
    return hashlib.sha256(Path(GUIDELINE_PATH).read_bytes()).hexdigest()


def test_parse_matches_golden_snapshot():
    golden = json.loads(GOLDEN_PATH.read_text())
    guideline_hash = _current_guideline_hash()
    assert (
        golden["source_hash"] == guideline_hash
    ), "Guidelines have changed; regenerate policies/RX-WEG-2025.json and the golden fixture."

    generated = parse_guidelines()
    validate_policy_snapshot(generated)
    assert _canonical_json(generated) == _canonical_json(golden)


def test_committed_snapshot_matches_parse_output():
    committed = json.loads(Path(SNAPSHOT_PATH).read_text())
    guideline_hash = _current_guideline_hash()
    assert (
        committed["source_hash"] == guideline_hash
    ), "Committed snapshot is stale relative to UpdatedPAGuidelines.txt; regenerate via setup_rag.py."

    generated = parse_guidelines()
    validate_policy_snapshot(committed)
    assert _canonical_json(generated) == _canonical_json(committed)


def test_missing_diagnosis_requires_documentation():
    result = evaluate_eligibility({"latest_bmi": "31.0", "conditions": [], "meds": []})
    assert result.verdict == "CDI_REQUIRED"


def test_no_renal_dialysis_safety_entries():
    snapshot = parse_guidelines()
    for entry in snapshot["safety_exclusions"]:
        cat = entry["category"].lower()
        assert "renal" not in cat and "dialysis" not in cat
        for s in entry.get("accepted_strings", []):
            s_low = s.lower()
            assert "renal" not in s_low and "dialysis" not in s_low

def test_ambiguity_parsing():
    snapshot = parse_guidelines()
    ambiguities = snapshot.get("ambiguities", [])
    assert len(ambiguities) > 0, "No ambiguity rules parsed"

    # Check for specific known rule
    sleep_apnea = next((r for r in ambiguities if r["pattern"] == "sleep apnea"), None)
    assert sleep_apnea is not None
    assert sleep_apnea["action"] == "MANUAL_REVIEW"
    assert "obstructive" in sleep_apnea.get("notes", "").lower()


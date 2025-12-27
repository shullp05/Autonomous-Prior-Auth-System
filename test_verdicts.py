
from agent_logic import make_decision


def test_make_decision_verdicts():
    # Mock states

    # 1. CDI Case
    state_cdi = {
        "audit_findings": {
            "verdict": "CDI_REQUIRED",
            "reasoning": "Clinical logic OK, admin logic failed",
            "physician_query_text": "QUERY: Add E66.9",
        },
        "patient_data": {}
    }

    res_cdi = make_decision(state_cdi)
    print("--- CDI Case ---")
    print(f"Final Status: {res_cdi['final_decision']}")
    print(f"Letter Content: {res_cdi['appeal_letter']}")

    if res_cdi['final_decision'] == "CDI_REQUIRED" and "QUERY: Add E66.9" in res_cdi['appeal_letter']:
        print("PASS")
    else:
        print("FAIL")

    # 2. Safety Signal Case
    state_safety = {
        "audit_findings": {
            "verdict": "SAFETY_SIGNAL_NEEDS_REVIEW",
            "reasoning": "Historical pancreatitis detected",
            "safety_context": "HISTORICAL",
            "safety_confidence": "SIGNAL",
            "safety_exclusion_code": "PANCREATITIS_HISTORY",
            "evidence_quoted": "history of pancreatitis"
        },
        "patient_data": {}
    }

    res_safety = make_decision(state_safety)
    print("\n--- Safety Signal Case ---")
    print(f"Final Status: {res_safety['final_decision']}")
    print(f"Appeal Note: {res_safety['appeal_note']}")

    if res_safety['final_decision'] == "SAFETY_SIGNAL_NEEDS_REVIEW" and "Safety Signal Detected" in res_safety['appeal_note']:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_make_decision_verdicts()


from pydantic import ValidationError

from agent_logic import AuditResult


def test_audit_coercion():
    print("Testing 'CARDIOVASCULAR_DISEASE' coercion to 'CVD'...")
    try:
        # This currently fails. We want it to succeed and convert to CVD.
        res = AuditResult(comorbidity_category="CARDIOVASCULAR_DISEASE")

        if res.comorbidity_category == "CVD":
            print("Success: Coerced to CVD")
        else:
            print(f"Failed: Accepted but not coerced? Value: {res.comorbidity_category}")

    except ValidationError:
        print("Failed: Validation Error (No coercion implemented yet)")

    print("\nTesting 'CVD' input...")
    try:
        res = AuditResult(comorbidity_category="CVD")
        if res.comorbidity_category == "CVD":
            print("Success: CVD preserved")
    except ValidationError as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_audit_coercion()

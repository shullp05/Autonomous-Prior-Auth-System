import json
import hashlib
import sys
import os

AUDIT_LOG_FILE = "audit_log.jsonl"

def calculate_hash(prev_hash, timestamp, event_type, details_str):
    payload = f"{prev_hash}|{timestamp}|{event_type}|{details_str}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def verify_log(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Log file '{filepath}' not found.")
        return False

    print(f"Verifying integrity of: {filepath}")
    prev_hash = "0" * 64
    line_num = 0
    valid_count = 0
    errors = []

    with open(filepath, 'r') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Invalid JSON")
                continue

            # Check structure
            required_keys = ['timestamp', 'event_type', 'details', 'prev_hash', 'hash']
            if not all(k in entry for k in required_keys):
                errors.append(f"Line {line_num}: Missing required keys")
                continue

            # Verify chain
            if entry['prev_hash'] != prev_hash:
                errors.append(f"Line {line_num}: BROKEN CHAIN. 'prev_hash' does not match previous entry hash.\n  Expected: {prev_hash}\n  Found:    {entry['prev_hash']}")
                # Cannot continue chain verification meaningfully if link is broken
                break 

            # Re-calculate hash
            # Need to ensure details serialization matches exactly how it was logged (audit_logger uses sort_keys=True)
            details_str = json.dumps(entry['details'], sort_keys=True)
            calculated_hash = calculate_hash(
                entry['prev_hash'], 
                entry['timestamp'], 
                entry['event_type'], 
                details_str
            )

            if calculated_hash != entry['hash']:
                errors.append(f"Line {line_num}: INVALID SIGNATURE.\n  Calculated: {calculated_hash}\n  Stored:     {entry['hash']}")
                break
            
            # Advancing chain
            prev_hash = entry['hash']
            valid_count += 1

    if errors:
        print("\n❌ INTEGRITY CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\n✅ INTEGRITY VERIFIED. {valid_count} entries checked. Chain is valid.")
        return True

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else AUDIT_LOG_FILE
    success = verify_log(filepath)
    sys.exit(0 if success else 1)

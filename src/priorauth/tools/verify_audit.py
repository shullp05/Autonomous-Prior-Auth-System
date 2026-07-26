"""Verify audit hash chains and signed checkpoint envelopes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

from priorauth import paths
from priorauth.audit_logger import GENESIS_HASH, canonical_json

AUDIT_LOG_FILE = str(paths.OUTPUT_DIR / "audit_log.jsonl")


def calculate_hash(prev_hash: str, timestamp: str, event_type: str, payload: str) -> str:
    return hashlib.sha256(f"{prev_hash}|{timestamp}|{event_type}|{payload}".encode()).hexdigest()


def verify_checkpoint(path: str | Path, signing_key: str | bytes) -> bool:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = envelope.pop("signature", None)
    key = signing_key.encode() if isinstance(signing_key, str) else signing_key
    expected = hmac.new(key, canonical_json(envelope).encode(), hashlib.sha256).hexdigest()
    return bool(signature and hmac.compare_digest(signature, expected))


def verify_log(filepath: str) -> bool:
    if not os.path.exists(filepath):
        print(f"Error: Log file '{filepath}' not found.")
        return False
    prev_hash = GENESIS_HASH
    valid_count = 0
    with open(filepath, encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {line_num}: invalid JSON")
                return False
            required = {"timestamp", "event_type", "details", "prev_hash", "hash"}
            if not required.issubset(entry):
                print(f"Line {line_num}: missing required keys")
                return False
            if entry["prev_hash"] != prev_hash:
                print(f"Line {line_num}: broken chain")
                return False
            if {"actor_roles", "decision_id", "idempotency_key"}.intersection(entry):
                payload = canonical_json(
                    {key: value for key, value in entry.items() if key not in {"prev_hash", "hash"}}
                )
            else:
                payload = json.dumps(entry["details"], sort_keys=True)
            expected = calculate_hash(prev_hash, entry["timestamp"], entry["event_type"], payload)
            if not hmac.compare_digest(str(entry["hash"]), expected):
                print(f"Line {line_num}: invalid entry hash")
                return False
            prev_hash = entry["hash"]
            valid_count += 1
    print(f"Integrity verified: {valid_count} entries; chain head {prev_hash}")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else AUDIT_LOG_FILE
    raise SystemExit(0 if verify_log(target) else 1)

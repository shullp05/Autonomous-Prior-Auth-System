import threading
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Dict, Any, Optional


class AuditLogger:
    """
    Singleton Audit Logger that implements cryptographic hash chaining.
    Each log entry includes a SHA-256 signature calculated from:
    hash = SHA256(prev_hash + timestamp + event_type + details_json)
    
    This ensures the log is append-only and tamper-evident.
    """
    _instance = None
    LOG_FILE = "audit_log.jsonl"

    def __new__(cls, log_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(AuditLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_file: Optional[str] = None):
        if self._initialized:
            return

        self.log_file = log_file or self.LOG_FILE
        self.prev_hash = self._get_last_hash()
        self._initialized = True

    def _get_last_hash(self) -> str:
        """
        Reads the last line of the log file to get the last hash.
        Returns a genesis hash if file is empty or missing.
        """
        if not os.path.exists(self.log_file):
            return "0" * 64  # Genesis hash

        try:
            with open(self.log_file, 'rb') as f:
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)

                last_line = f.readline().decode().strip()
                if not last_line:
                    return "0" * 64

                try:
                    entry = json.loads(last_line)
                    return entry.get('hash', "0" * 64)
                except json.JSONDecodeError:
                    return "0" * 64
        except Exception:
            return "0" * 64

    def _calculate_hash(self, prev_hash: str, timestamp: str, event_type: str, details_str: str) -> str:
        """
        Calculates SHA-256 hash of the entry.
        """
        payload = f"{prev_hash}|{timestamp}|{event_type}|{details_str}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def log_event(self, event_type: str, details: Dict[str, Any], actor: str = "system", patient_id: Optional[str] = None) -> str:
        """
        Logs an event with a cryptographic signature.
        Returns the hash of the new entry.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Minimize PHI: If patient_id provided, hash it or keep it minimal if it's just a UUID
        # Assuming patient_id is already a UUID (reference), logging it is fine.
        # If it was a name, we would hash it.
        safe_patient_mask = patient_id if patient_id else "N/A"

        entry_data = {
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": actor,
            "patient_id": safe_patient_mask,
            "details": details,
            "prev_hash": self.prev_hash
        }

        # Serialize details for consistent hashing
        details_str = json.dumps(details, sort_keys=True)

        # Calculate hash
        entry_hash = self._calculate_hash(self.prev_hash, timestamp, event_type, details_str)
        entry_data['hash'] = entry_hash

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry_data) + "\n")

        # Update state
        self.prev_hash = entry_hash
        return entry_hash

# Global instance
_logger = AuditLogger()

def get_audit_logger() -> AuditLogger:
    return _logger

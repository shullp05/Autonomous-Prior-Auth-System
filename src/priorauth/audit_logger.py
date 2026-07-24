"""Durable, authenticated audit logging and independently anchored checkpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from priorauth import paths

GENESIS_HASH = "0" * 64


class AuditError(RuntimeError):
    """Base class for audit-control failures."""


class AuditConfigurationError(AuditError):
    """Raised when a required audit control is not configured."""


class AuditWriteError(AuditError):
    """Raised when an audit record cannot be durably persisted."""


@dataclass(frozen=True)
class AuditReceipt:
    event_hash: str
    checkpoint_path: str | None
    anchor_path: str | None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AuditLogger:
    """Append-only hash chain with HMAC-signed, externally anchored checkpoints."""

    LOG_FILE = str(paths.OUTPUT_DIR / "audit_log.jsonl")

    def __init__(
        self,
        log_file: str | None = None,
        *,
        signing_key: str | bytes | None = None,
        checkpoint_dir: str | Path | None = None,
        anchor_dir: str | Path | None = None,
        require_signature: bool | None = None,
        require_external_anchor: bool | None = None,
    ) -> None:
        self.log_file = str(log_file or self.LOG_FILE)
        configured_key = signing_key if signing_key is not None else os.getenv("PA_AUDIT_SIGNING_KEY")
        signing_key_file = os.getenv("PA_AUDIT_SIGNING_KEY_FILE")
        if configured_key is None and signing_key_file:
            try:
                configured_key = Path(signing_key_file).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise AuditConfigurationError(f"Could not read audit signing key file: {exc}") from exc
        self.signing_key = configured_key.encode() if isinstance(configured_key, str) else configured_key
        self.require_signature = (
            require_signature
            if require_signature is not None
            else os.getenv("PA_AUDIT_REQUIRE_SIGNATURE", "false").lower() == "true"
        )
        self.require_external_anchor = (
            require_external_anchor
            if require_external_anchor is not None
            else os.getenv("PA_AUDIT_REQUIRE_EXTERNAL_ANCHOR", "false").lower() == "true"
        )
        base = Path(self.log_file).parent
        self.checkpoint_dir = Path(checkpoint_dir or os.getenv("PA_AUDIT_CHECKPOINT_DIR", base / "audit_checkpoints"))
        configured_anchor = anchor_dir or os.getenv("PA_AUDIT_ANCHOR_DIR")
        self.anchor_dir = Path(configured_anchor) if configured_anchor else None
        if self.require_signature and not self.signing_key:
            raise AuditConfigurationError("PA_AUDIT_SIGNING_KEY is required when signatures are enforced")
        if self.require_external_anchor and self.anchor_dir is None:
            raise AuditConfigurationError("PA_AUDIT_ANCHOR_DIR is required when external anchoring is enforced")
        self._lock = threading.RLock()
        self._faulted = False
        self.prev_hash = self._get_last_hash()
        self.last_receipt: AuditReceipt | None = None

    def _get_last_hash(self) -> str:
        path = Path(self.log_file)
        if not path.exists():
            return GENESIS_HASH
        try:
            last = next((line for line in reversed(path.read_text(encoding="utf-8").splitlines()) if line.strip()), "")
            return str(json.loads(last).get("hash", GENESIS_HASH)) if last else GENESIS_HASH
        except (OSError, json.JSONDecodeError) as exc:
            raise AuditWriteError(f"Cannot resume audit chain from {path}: {exc}") from exc

    @staticmethod
    def _calculate_hash(prev_hash: str, timestamp: str, event_type: str, payload: str) -> str:
        return hashlib.sha256(f"{prev_hash}|{timestamp}|{event_type}|{payload}".encode()).hexdigest()

    def _signature(self, checkpoint: dict[str, Any]) -> str | None:
        if not self.signing_key:
            return None
        return hmac.new(self.signing_key, canonical_json(checkpoint).encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _durable_append(path: Path, data: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise AuditWriteError(f"Durable append failed for {path}: {exc}") from exc

    @staticmethod
    def _write_once(directory: Path, name: str, data: str) -> Path:
        path = directory / name
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o440)
            return path
        except OSError as exc:
            raise AuditWriteError(f"Append-only checkpoint write failed for {path}: {exc}") from exc

    def checkpoint(self, event_hash: str, timestamp: str) -> AuditReceipt:
        checkpoint = {
            "algorithm": "HMAC-SHA256" if self.signing_key else "UNSIGNED",
            "chain_head": event_hash,
            "created_at": timestamp,
            "log_file": Path(self.log_file).name,
        }
        signature = self._signature(checkpoint)
        envelope = {**checkpoint, "signature": signature}
        if self.require_signature and not signature:
            raise AuditWriteError("Required checkpoint signature was not produced")
        encoded = canonical_json(envelope) + "\n"
        name = f"{timestamp.replace(':', '').replace('+', '_')}-{event_hash}.json"
        checkpoint_path = self._write_once(self.checkpoint_dir, name, encoded)
        anchor_path = self._write_once(self.anchor_dir, name, encoded) if self.anchor_dir else None
        if self.require_external_anchor and anchor_path is None:
            raise AuditWriteError("Required external checkpoint anchor was not persisted")
        return AuditReceipt(event_hash, str(checkpoint_path), str(anchor_path) if anchor_path else None)

    def log_event(
        self,
        event_type: str,
        details: dict[str, Any],
        actor: str = "system",
        patient_id: str | None = None,
        *,
        actor_roles: list[str] | None = None,
        decision_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if self._faulted:
                raise AuditWriteError("Audit logger is faulted after an incomplete durable write")
            entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "actor": actor,
                "actor_roles": sorted(actor_roles or []),
                "patient_id": patient_id or "N/A",
                "decision_id": decision_id,
                "idempotency_key": idempotency_key,
                "details": details,
                "prev_hash": self.prev_hash,
            }
            payload = canonical_json({key: value for key, value in entry.items() if key not in {"prev_hash"}})
            entry_hash = self._calculate_hash(self.prev_hash, timestamp, event_type, payload)
            entry["hash"] = entry_hash
            self._durable_append(Path(self.log_file), canonical_json(entry) + "\n")
            try:
                receipt = self.checkpoint(entry_hash, timestamp)
            except AuditError:
                self._faulted = True
                raise
            self.prev_hash = entry_hash
            self.last_receipt = receipt
            return entry_hash


_logger: AuditLogger | None = None
_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    global _logger
    with _logger_lock:
        if _logger is None:
            _logger = AuditLogger()
        return _logger

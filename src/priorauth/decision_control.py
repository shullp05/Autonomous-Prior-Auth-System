"""Identity, authorization, idempotency, and human-override controls."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import uuid
from base64 import urlsafe_b64decode
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from priorauth import paths
from priorauth.audit_logger import AuditLogger, AuditWriteError, get_audit_logger


class AuthorizationError(PermissionError):
    pass


class DuplicateDecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Actor:
    subject: str
    roles: tuple[str, ...]
    auth_source: str

    @classmethod
    def from_environment(cls) -> "Actor":
        token = os.getenv("PA_ACTOR_TOKEN", "").strip()
        if token:
            verification_key = os.getenv("PA_AUTH_VERIFICATION_KEY", "")
            if not verification_key:
                raise AuthorizationError("PA_AUTH_VERIFICATION_KEY is required for PA_ACTOR_TOKEN")
            return cls.from_signed_token(token, verification_key)
        subject = os.getenv("PA_ACTOR_ID", "service:batch-runner").strip()
        roles = tuple(sorted(filter(None, (role.strip() for role in os.getenv("PA_ACTOR_ROLES", "service").split(",")))))
        if not subject or not roles:
            raise AuthorizationError("PA_ACTOR_ID and PA_ACTOR_ROLES must identify the authenticated principal")
        return cls(subject, roles, os.getenv("PA_AUTH_SOURCE", "trusted-runtime"))

    @classmethod
    def from_signed_token(cls, token: str, verification_key: str | bytes) -> "Actor":
        try:
            payload_part, signature_part = token.split(".", 1)
            key = verification_key.encode() if isinstance(verification_key, str) else verification_key
            expected = hmac.new(key, payload_part.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature_part, expected):
                raise AuthorizationError("Actor token signature is invalid")
            padded = payload_part + "=" * (-len(payload_part) % 4)
            payload = json.loads(urlsafe_b64decode(padded).decode())
            if float(payload.get("exp", 0)) <= datetime.now(timezone.utc).timestamp():
                raise AuthorizationError("Actor token has expired")
            subject = str(payload["sub"]).strip()
            roles = tuple(sorted(str(role).strip() for role in payload["roles"] if str(role).strip()))
            if not subject or not roles:
                raise AuthorizationError("Actor token has no subject or roles")
            return cls(subject, roles, str(payload.get("iss", "signed-token")))
        except AuthorizationError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AuthorizationError("Actor token is malformed") from exc

    def require(self, *allowed_roles: str) -> None:
        if not set(self.roles).intersection(allowed_roles):
            raise AuthorizationError(f"Actor {self.subject} requires one of roles: {', '.join(allowed_roles)}")


def decision_identifiers(patient_id: str, policy_hash: str, request_key: str | None = None) -> tuple[str, str]:
    idempotency_key = request_key or hashlib.sha256(f"{patient_id}|{policy_hash}".encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"priorauth:{idempotency_key}")), idempotency_key


class IdempotencyStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.OUTPUT_DIR / "decision_idempotency.jsonl"
        self._lock = threading.Lock()

    def reserve(self, decision_id: str, idempotency_key: str) -> None:
        with self._lock:
            seen: set[str] = set()
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        seen.add(str(json.loads(line)["idempotency_key"]))
            if idempotency_key in seen:
                raise DuplicateDecisionError(f"Decision request already processed: {idempotency_key}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"decision_id": decision_id, "idempotency_key": idempotency_key}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise AuditWriteError(f"Could not durably reserve idempotency key: {exc}") from exc


def apply_human_override(
    *,
    decision_id: str,
    patient_id: str,
    original_verdict: str,
    replacement_verdict: str,
    reason: str,
    actor: Actor,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    actor.require("clinical_reviewer", "medical_director")
    if not reason.strip():
        raise ValueError("Human overrides require a non-empty reason")
    if original_verdict == replacement_verdict:
        raise ValueError("Replacement verdict must differ from the original verdict")
    logger = audit_logger or get_audit_logger()
    override = {
        "decision_id": decision_id,
        "patient_id": patient_id,
        "original_verdict": original_verdict,
        "replacement_verdict": replacement_verdict,
        "reason": reason.strip(),
        "actor": actor.subject,
        "actor_roles": list(actor.roles),
        "auth_source": actor.auth_source,
        "overridden_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.log_event(
        "HUMAN_OVERRIDE",
        override,
        actor=actor.subject,
        actor_roles=list(actor.roles),
        patient_id=patient_id,
        decision_id=decision_id,
    )
    return override

import hashlib
import hmac
import json
from base64 import urlsafe_b64encode
from datetime import datetime, timezone

import pytest

from priorauth.audit_logger import AuditConfigurationError, AuditLogger, AuditWriteError
from priorauth.decision_control import (
    Actor,
    AuthorizationError,
    DuplicateDecisionError,
    IdempotencyStore,
    apply_human_override,
    decision_identifiers,
)
from priorauth.tools.verify_audit import verify_checkpoint, verify_log


def test_signed_checkpoint_is_anchored_and_verifiable(tmp_path):
    logger = AuditLogger(
        str(tmp_path / "audit.jsonl"),
        signing_key="test-key",
        checkpoint_dir=tmp_path / "checkpoints",
        anchor_dir=tmp_path / "external-anchor",
        require_signature=True,
        require_external_anchor=True,
    )
    event_hash = logger.log_event("DECISION", {"verdict": "APPROVED"}, decision_id="decision-1")

    assert logger.last_receipt is not None
    assert logger.last_receipt.event_hash == event_hash
    assert verify_log(logger.log_file)
    assert verify_checkpoint(logger.last_receipt.checkpoint_path, "test-key")
    assert verify_checkpoint(logger.last_receipt.anchor_path, "test-key")


def test_required_signature_and_anchor_fail_closed(tmp_path):
    with pytest.raises(AuditConfigurationError):
        AuditLogger(str(tmp_path / "audit.jsonl"), require_signature=True)

    logger = AuditLogger(
        str(tmp_path / "audit.jsonl"),
        signing_key="test-key",
        checkpoint_dir=tmp_path / "checkpoints",
        require_external_anchor=True,
        anchor_dir=tmp_path / "not-a-directory",
    )
    (tmp_path / "not-a-directory").write_text("occupied", encoding="utf-8")
    with pytest.raises(AuditWriteError):
        logger.log_event("DECISION", {"verdict": "APPROVED"})
    with pytest.raises(AuditWriteError, match="faulted"):
        logger.log_event("DECISION", {"verdict": "DENIED"})


def test_idempotency_store_rejects_duplicate_request(tmp_path):
    decision_id, key = decision_identifiers("patient-1", "policy-hash", "request-1")
    store = IdempotencyStore(tmp_path / "idempotency.jsonl")
    store.reserve(decision_id, key)
    with pytest.raises(DuplicateDecisionError):
        store.reserve(decision_id, key)


def test_override_requires_authorized_actor_and_audits_change(tmp_path):
    logger = AuditLogger(str(tmp_path / "audit.jsonl"), checkpoint_dir=tmp_path / "checkpoints")
    with pytest.raises(AuthorizationError):
        apply_human_override(
            decision_id="d1",
            patient_id="p1",
            original_verdict="DENIED",
            replacement_verdict="APPROVED",
            reason="Reviewed additional evidence",
            actor=Actor("user:viewer", ("viewer",), "oidc"),
            audit_logger=logger,
        )

    override = apply_human_override(
        decision_id="d1",
        patient_id="p1",
        original_verdict="DENIED",
        replacement_verdict="APPROVED",
        reason="Reviewed additional evidence",
        actor=Actor("user:reviewer", ("clinical_reviewer",), "oidc"),
        audit_logger=logger,
    )
    assert override["actor"] == "user:reviewer"
    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert entry["event_type"] == "HUMAN_OVERRIDE"
    assert entry["details"]["original_verdict"] == "DENIED"


def test_actor_identity_is_authenticated_by_signed_token():
    payload = {
        "sub": "user:reviewer",
        "roles": ["clinical_reviewer"],
        "iss": "test-idp",
        "exp": datetime.now(timezone.utc).timestamp() + 60,
    }
    encoded = urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(b"verification-key", encoded.encode(), hashlib.sha256).hexdigest()

    actor = Actor.from_signed_token(f"{encoded}.{signature}", "verification-key")

    assert actor.subject == "user:reviewer"
    assert actor.roles == ("clinical_reviewer",)
    with pytest.raises(AuthorizationError):
        Actor.from_signed_token(f"{encoded}.invalid", "verification-key")

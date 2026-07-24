# Security

## Threat model
- Prevent outbound egress of sensitive inputs.
- Ensure deterministic eligibility logic cannot be altered by external prompts.
- Preserve audit log integrity (tamper-evident records).
- Reduce supply-chain drift with locked dependencies.

## Offline mode (opt-in, localhost-safe)
Offline mode is a process-level guardrail that patches network primitives when enabled.

- Enable with `PA_OFFLINE_MODE=true` (or `OFFLINE_MODE=true`).
- Localhost is allowed by default; override with `PA_OFFLINE_ALLOW_LOCALHOST`.
- Allowlist extra hosts with `PA_OFFLINE_ALLOWLIST` (comma-separated).
- Unknown hosts are blocked by default (`PA_OFFLINE_STRICT_UNKNOWN_HOST=true`).

Patched callables (when enabled):
- `socket.socket.connect` / `connect_ex`
- `socket.socket.sendto` / `sendmsg`
- `socket.getaddrinfo`
- `socket.create_connection`
- `urllib.request.urlopen`
- `requests.Session.request`

Behavior:
- External DNS and outbound connections raise standard network exceptions.
- Loopback aliases (e.g., `localhost`, `127.0.0.1`, `::1`) remain allowed by default so local Ollama calls keep working.

## Audit logging
- `audit_log.jsonl` uses SHA-256 hash chaining and durable `fsync` writes.
- Set `PA_AUDIT_SIGNING_KEY` (or the secret-file path `PA_AUDIT_SIGNING_KEY_FILE`) and `PA_AUDIT_REQUIRE_SIGNATURE=true` to require HMAC-SHA256 checkpoint signatures.
- Mount `PA_AUDIT_ANCHOR_DIR` from independently administered object-locked or append-only storage and set `PA_AUDIT_REQUIRE_EXTERNAL_ANCHOR=true`. Checkpoint collisions fail closed.
- Automated decisions are withheld when the event, signed checkpoint, required external anchor, or idempotency reservation cannot be durably persisted.
- `PA_ACTOR_TOKEN` carries an expiring HMAC-authenticated subject, issuer, and roles and is verified with `PA_AUTH_VERIFICATION_KEY`. Trusted service runtimes may use `PA_ACTOR_ID`, `PA_ACTOR_ROLES`, and `PA_AUTH_SOURCE`. Overrides require `clinical_reviewer` or `medical_director` and record the reason, actor, time, original verdict, and replacement verdict.
- Verify with `python -m priorauth.tools.verify_audit`.

The local checkpoint directory demonstrates the append-only contract but is not an independent trust domain. Production anchors must use WORM/object-lock retention, separate administrator credentials, restrictive deletion permissions, and external monitoring.

## Infrastructure network isolation
- The black-box Docker Compose profile uses `network_mode: none`, drops all capabilities, and enables `no-new-privileges`.
- `deploy/kubernetes/network-policy.yaml` supplies default-deny ingress/egress and an explicit local-model exception on port 11434.
- Python offline interception remains defense in depth rather than the network security boundary.

## Clinical validation boundary
Synthetic and regression evidence does not establish clinical validity. `docs/CLINICAL_VALIDATION.md` defines the prospective, independently adjudicated validation and stop-ship protocol required before production use.

## Data handling
- The repository ships synthetic data only; no PHI is included.
- Use `.env` for local secrets and keep them out of git (ignored by `.gitignore`).
- Generated artifacts (e.g., `output/`, `chroma_db/`) are ignored by default.

## Limitations
- Offline mode is process-level patching, not a full firewall.
- Native extensions or external processes can bypass Python-level patches.
- Model downloads should be pre-cached; offline mode will block external pulls.

## Verification
- `pytest -q tests/test_offline_mode.py` validates outbound blocking and localhost allowances.

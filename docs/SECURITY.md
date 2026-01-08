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
- [output/audit_log.jsonl](../output/audit_log.jsonl) uses SHA-256 hash chaining for tamper evidence (runtime artifact; gitignored).
- Verify with `python -m priorauth.tools.verify_audit` ([src/priorauth/tools/verify_audit.py](../src/priorauth/tools/verify_audit.py)).

## Data handling
- The repository ships synthetic data only; no PHI is included.
- Use a local `.env` file for secrets and keep them out of git (ignored by [.gitignore](../.gitignore)).
- Generated artifacts (e.g., [output/](../output/), [chroma_db/](../chroma_db/)) are ignored by default.

## Limitations
- Offline mode is process-level patching, not a full firewall.
- Native extensions or external processes can bypass Python-level patches.
- Model downloads should be pre-cached; offline mode will block external pulls.

## Verification
- `pytest -q tests/test_offline_mode.py` validates outbound blocking and localhost allowances (see [tests/test_offline_mode.py](../tests/test_offline_mode.py)).

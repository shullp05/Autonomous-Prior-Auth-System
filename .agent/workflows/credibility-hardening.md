---
description: Credibility Hardening - Make the system survive a hostile reviewer (Phases 9-14)
---

# Credibility Hardening Workflow

This workflow covers Phases 9 through 14 of the project roadmap, focused on making the system survive a hostile reviewer who is looking for unsafe clinical calls, tamperable logs, or outbound network fetches.

**Execution Order:** Phase 9 → 13 → 10 → 11 → 12 → 14

---

## Phase 9 — Clinical safety correctness (stabilize outputs before anything else)

### 9.0 Freeze a baseline (so you can prove you improved)
- [ ] Tag current state: `v0.8-metrics-contract-stable`
- [ ] Save a small “before” run output folder as `tests/fixtures/baselines/v0_8/`
  - **Done when:** you can run “before vs after” diffs and prove fewer safety false positives.

### 9.1 Add a conservative safety state (the key ripple you *want* early)
- [ ] Add `SAFETY_SIGNAL_NEEDS_REVIEW` to your status taxonomy.
- [ ] Rule: **Hard-stop only when evidence is current/active.** Everything else downgrades to Safety Signal.
  - **Files touched:** `status_taxonomy.*` (source of truth), backend decision emitter, UI pill mapping + filters + Sankey rollups.
  - **Done when:** “history of miscarriage” can *never* be labeled as pregnancy hard-stop.

### 9.2 Implement deterministic context rules (tiny, strict, tested)
- [ ] Build a small module that classifies a matched safety term as: `active` vs `historical` vs `negated` vs `hypothetical` vs `family_history`.
- [ ] **Do this with boring rules first (not LLM).**
- [ ] **Minimum rule coverage:**
  - Temporality: “history of”, “prior”, “remote”, “postpartum”, “s/p”
  - Negation: “denies”, “negative for”, “ruled out”
  - Subject: “family history”, “mother had”
  - Uncertainty: “possible”, “concern for”
  - **Done when:** you have **30–50 unit tests** that lock these classifications.

### 9.3 Add explicit safety evidence fields (explainability)
- [ ] For every safety-triggered case, store:
  - `safety_term`
  - `evidence_snippet` (±N chars around match)
  - `context_classification` (active/historical/negated…)
  - `safety_confidence_tier` (hard-stop vs signal)
  - **Done when:** the UI can show “why we downgraded” without hand-waving.

---

## Phase 13 — Evaluation harness (lock safety + correctness before UI churn)

### 13.1 Create a gold fixture pack (synthetic, versioned)
- [ ] Create 50–200 synthetic cases with **expected**:
  - `status_code`
  - key reasons (at least 1–2 strings)
  - safety classification (hard-stop vs signal)
  - **Done when:** you can compute regression metrics locally with one command.

### 13.2 Add regression gates (tests that fail on safety backsliding)
- [ ] Add thresholds like:
  - pregnancy hard-stop false positives must be **0** on fixtures
  - overall deterministic eligibility accuracy must not decrease
  - ambiguity triggers must remain stable (or changes must be explained)
  - **Done when:** you can’t “accidentally” reintroduce the miscarriage bug without CI screaming.

---

## Phase 10 — Workflow truth in UI (now safe to touch UX)

### 10.1 Normalize row actions into one primary button everywhere
- [ ] Stop mixing icon-only, blocked icons, and different paradigms.
- [ ] **Mapping:**
  - Meets Criteria → `Open Packet` / `Review & Sign`
  - Needs Clarification → `Resolve Clarification`
  - Missing Required Data → `Add Required Data`
  - Safety Contraindication → `View Hard Stop`
  - Safety Signal → `Verify Safety Signal`
  - Not Eligible → `View Rationale`
  - **Done when:** every status has a clear next move with consistent UI mechanics.

### 10.2 Fix the “minutes per case” confusion
- [ ] Keep **Processing Velocity** = system compute KPI (move to Diagnostics)
- [ ] Keep **Staff Minutes per PA** = configurable governance assumption (used for Hours Saved)
  - **Done when:** Hours Saved never references compute speed; only staff baseline assumptions.

### 10.3 Make Sankey reflect your taxonomy (simple node, rich hover)
- [ ] Sankey node: `Needs Action`
- [ ] Tooltip breakdown: Needs Clarification / Missing Required Data / Safety Signal
  - **Done when:** table counts, KPI counts, and Sankey counts can’t disagree.

### 10.4 Accessibility + clinical usability pass
- [ ] Tooltips: click-to-pin + ESC close + keyboard focus
- [ ] Table: filters + search + sort
- [ ] Large table perf: virtualization if needed
  - **Done when:** it works on trackpads/touch and doesn’t rely on hover-only.

---

## Phase 11 — Tamper-evident audit + local security (this is what makes “black box” real)

### 11.1 Centralize audit events behind one interface
- [ ] Create `audit/log_event(...)` and ban direct writes elsewhere.
- [ ] Events to log: case status transitions, letter generation, letter edits (with diff summary), sign/confirm actions, policy_pack_hash + model_pack_hash + run_id.
  - **Done when:** grep shows no other module writes audit JSON directly.

### 11.2 Hash-chain the audit log (tamper evidence)
- [ ] Each record includes: `prev_hash` and `hash = sha256(prev_hash + canonical_json(payload))`
  - **Done when:** editing one past file breaks the chain.

### 11.3 PHI minimization + retention policy
- [ ] Default: store patient *reference ID only* (no names/DOB) in logs
- [ ] Add retention config + `purge` command (and document it)
  - **Done when:** repo answers “what PHI exists where” clearly and defensibly.

### 11.4 Local-only auth + RBAC
- [ ] Min viable: local users file (admin/staff/clinician), role gates (only clinician can “Sign”, staff can “Resolve Missing Data”), session timeout / lock screen.
  - **Done when:** your UI isn’t a “toy with a sign button.”

---

## Phase 12 — Offline enforcement + reproducibility (Windows + Linux)

### 12.1 Add build-time artifact scripts (online, controlled)
- [ ] Create `/artifacts`: `wheels/windows/`, `wheels/linux/`, `models/`, `manifests/sha256sums.txt`
- [ ] Add scripts: `scripts/build_artifacts_linux.sh`, `scripts/build_artifacts_windows.ps1`
- [ ] **Core rule:** runtime installs must use wheelhouse only: `pip install --no-index --find-links=...`
  - **Done when:** a clean offline machine can install + run using only artifacts.

### 12.2 Force Hugging Face + telemetry offline behavior
- [ ] At runtime set: `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `TRANSFORMERS_OFFLINE=1`
- [ ] Ensure all model loads use local paths / caches.
  - **Done when:** missing models fail fast (no “helpful download”).

### 12.3 Docker hard lockdown
- [ ] Agent container: `network_mode: "none"`, run as non-root, mount `/models` read-only, write only to `/output` and `/chroma_db`.
  - **Done when:** even malicious code can’t phone home because there’s no network.

### 12.4 “Network dead-man switch” test
- [ ] Add a pytest that blocks outbound sockets during a representative run.
  - **Done when:** any accidental outbound call fails tests immediately.

### 12.5 Windows + Linux dependency strategy
- [ ] Deliver two lockfiles + wheelhouses: `requirements.in`, `requirements.lock.linux.txt`, `requirements.lock.windows.txt`
- [ ] Handle Torch separately (CPU vs CUDA).
  - **Done when:** a reviewer can reproduce on both OS’s without “well on my machine…”

---

## Phase 14 — README + repo polish

### 14.1 README as a security + product spec
- [ ] Threat model + air-gap enforcement
- [ ] Architecture diagram (containers, volumes, trust boundaries)
- [ ] Clinical safety policy (hard-stop vs signal, context handling)
- [ ] Audit integrity (hash chain)
- [ ] Reproducibility (Win/Linux)
- [ ] Evaluation harness (fixtures + regression gates)

### 14.2 Public GitHub hygiene checklist
- [ ] Ensure **zero PHI** in repo (fixtures must be synthetic)
- [ ] Add `SECURITY.md`, license clarity, and “one-command demo” script.
  - **Done when:** your repo reads like something a regulated org could take seriously.

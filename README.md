# 🏥 Autonomous Prior Authorization Agent

Deterministic policy engine + local LLM/RAG fallback for Wegovy prior auth, producing auditable decision traces and fairness reporting.  
**Synthetic data only. Local-first design to reduce PHI exposure; not a certified HIPAA implementation.**

<p align="left">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-2b9348?style=for-the-badge"></a>
  <a href="https://github.com/shullp05/Autonomous-Prior-Auth-System/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/shullp05/Autonomous-Prior-Auth-System/ci.yml?branch=main&style=for-the-badge&label=CI"></a>
  <a href="https://github.com/shullp05/Autonomous-Prior-Auth-System/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/shullp05/Autonomous-Prior-Auth-System/tests.yml?branch=main&style=for-the-badge&label=Tests"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Workflow_Orchestration-6c757d?style=for-the-badge">
  <img alt="RAG" src="https://img.shields.io/badge/RAG-Policy_Lookup-0a9396?style=for-the-badge">
  <a href="docs/ARCHITECTURE.md"><img alt="Deterministic" src="https://img.shields.io/badge/Deterministic-Source_of_Truth-5fa8d3?style=for-the-badge"></a>
  <a href="docs/SECURITY.md"><img alt="Local-first" src="https://img.shields.io/badge/Local--First-No_PHI_Required-1b4965?style=for-the-badge"></a>
</p>

## Hiring Manager Quick Scan (30 seconds)

- **Does:** Evaluates Wegovy PA eligibility (approve/deny/needs-review) from CSV/FHIR using deterministic rules + evidence-grounded LLM narrative.
- **Unique:** Deterministic engine is the **source of truth**; LLM is constrained to extraction + justification with citations.
- **Proves:** workflow orchestration (LangGraph), RAG (Chroma + embeddings/rerank), strict contracts (Pydantic), adversarial testing, reproducible audit artifacts, React+D3 dashboard.
- **Run (3–5 min):**
  ```bash
  pip install -r requirements.lock
  python -m priorauth.apps.agent.chaos_monkey
  python -m priorauth.apps.agent.setup_rag
  python -m priorauth.apps.agent.batch_runner
  cd apps/ui && npm i && npm run dev

## **Outputs:** `dashboard_data.json`, `governance_report.json`, `.last_model_trace.json` (+ `audit_log.jsonl` chained; gitignored).

## Demo (placeholders)

**90-second walkthrough (add link):**

* [End-to-end demo clip (placeholder)](docs/media/demo-clip.mp4)
* [Short demo clip (placeholder)](docs/media/demo-clip-short.mp4)

**Screenshots (add 3–5):**
![Dashboard UI (placeholder)](docs/media/screenshot-dashboard.svg)
![Governance UI (placeholder)](docs/media/screenshot-governance.svg)

---

## 🚀 Executive Summary

This repo demonstrates an end-to-end prior authorization evaluator for Wegovy (semaglutide) weight management prescriptions.

Instead of treating an LLM as an oracle, the system uses a split-responsibility design:

* **Deterministic policy engine:** authoritative eligibility + safety rules (reproducible, testable)
* **LLM + RAG:** constrained to evidence-grounded extraction + narrative justification and optional letter artifacts

The default inference path is local-only (Ollama). Optional offline mode can block outbound egress while still permitting localhost calls. The repo ships synthetic data only; no PHI is included.

---

## 🛠️ Architecture & Workflow

The system uses a stateful workflow graph (LangGraph) to orchestrate retrieval, audit, and verification.

### Workflow Diagram

```mermaid
graph TD
    User([User / API Request]) -->|Patient ID + Drug| GraphStart

    subgraph Core["Core Agent Workflow (LangGraph)"]
        GraphStart --> NodeRetrieval["Policy Retrieval (RAG)"]
        NodeRetrieval -->|Docs + Scores| NodeAudit["Clinical Audit (LLM)"]
        NodeAudit -->|Extraction + Reasoning| NodeDecision["Decision Engine"]

        subgraph Verify["Split-Brain Verification"]
            NodeDecision -->|Check 1| Guardrails{Safety Checks?}
            Guardrails -->|Fail| VerdictDenySafety["Deny: Safety"]
            Guardrails -->|Pass| DeterministicLogic{Rule Engine}
            DeterministicLogic -->|Match| VerdictApprove["Approve"]
            DeterministicLogic -->|Mismatch| VerdictDenyClinical["Deny: Clinical"]
        end
    end

    VerdictApprove --> OutputGen["Generate Report"]
    VerdictDenySafety --> OutputGen
    VerdictDenyClinical --> OutputGen

    OutputGen -->|JSON + Dashboard| Dashboard((Analytics Dashboard))
```

### Data Flow

1. **Ingestion:** Patient observations (BMI, conditions, meds) loaded from CSV/FHIR.
2. **Retrieval:** ChromaDB retrieves policy atoms via MedEmbed embeddings (k=25 by default), optional BCE reranking; top 8 docs feed the LLM by default.
3. **Audit:** Local LLM analyzes clinical data against retrieved policy evidence (override model via `PA_AUDIT_MODEL_FLAVOR`).
4. **Verification:** Deterministic Python layer cross-verifies the LLM findings against safety + eligibility rules.
5. **Output:** Structured JSON decisions + optional letter artifacts + audit/fairness reports.

---

## 🔬 Technical Deep Dive

### Core Stack

* **Language:** Python 3.11+
* **Orchestration:** LangGraph, LangChain
* **Vector Query:** ChromaDB, BCEmbedding (reranker)
* **LLM Serving:** Ollama (Local) — default flavor `nemo8b` (`pa-audit-nemotron-cascade8b:latest`)
* **Validation:** Pydantic (strict schema enforcement)
* **Testing:** Pytest (unit, integration, adversarial, and safety tests)

### Key Features

* **Deterministic guardrails:** Safety exclusions (e.g., pregnancy, MTC, concurrent GLP-1) override any LLM output.
* **Policy snapshotting:** Parses policy guidelines into versioned JSON (`RX-WEG-2025.json`) with SHA-256 hashing.
* **RAG + reranking:** Two-stage retrieval (vector search → cross-encoder rerank) narrows context to relevant clauses.
* **Governance audit:** FNR parity analysis using Wilson CI + two-proportion z-tests across demographic slices.

### Multi-Layer Safety Model

| Layer | Component              | Description                                         |
| ----: | ---------------------- | --------------------------------------------------- |
|     1 | LLM Output             | Pydantic `AuditResult` schema validation            |
|     2 | JSON Parsing           | Robust `_extract_json_object()` with fallbacks      |
|     3 | Python Guardrails      | `_apply_policy_guardrails()` enforces hard rules    |
|     4 | Deterministic Override | `evaluate_eligibility()` always runs as cross-check |
|     5 | Governance Audit       | `run_governance_audit()` for FNR parity             |

---

## 📊 Dashboard Metrics & Definitions

The analytics dashboard computes metrics deterministically using the Metrics Contract (`apps/ui/src/metricsEngine.js`).

### Status Taxonomy

| Display Label               | Definition                                                      | Action Required      |
| :-------------------------- | :-------------------------------------------------------------- | :------------------- |
| **Meets Criteria**          | Clinical data satisfies policy requirements (clinical + admin). | None (Auto-Approved) |
| **Needs Clarification**     | Ambiguous terms found (e.g., “elevated BP”).                    | Manual Review        |
| **Missing Required Data**   | Essential observation data absent.                              | Provider Outreach    |
| **CDI Required**            | Clinically eligible but missing anchor codes (e.g., E66.x).     | Physician Query      |
| **Safety Signal**           | Potential safety risk detected; requires confirmation.          | Safety Verification  |
| **Safety Contraindication** | Active hard stop detected (e.g., pregnancy).                    | None (Auto-Denied)   |
| **Not Eligible**            | Violates policy criteria.                                       | None (Auto-Denied)   |

---

## 🔒 Security & Reproducibility Notes

This repo is local-first by design and supports offline enforcement options.

### Offline Enforcement & Reproducibility

* **Offline enforcement:** runtime patching of `socket`, `getaddrinfo`, `urllib`, and `requests` blocks outbound egress while allowing localhost (Ollama).

  * Enabled via `PA_OFFLINE_MODE=true` (opt-in).
* **Offline env guardrails:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `LANGSMITH_DISABLED=true`, etc. set in offline scripts/compose.
* **Dependency locking:** `requirements.lock` generated via `scripts/freeze_dependencies.sh`.

### Tamper-Evident Audit (runtime artifact)

* **Hash chaining:** decisions logged to `audit_log.jsonl` with SHA-256 chaining (gitignored).
* **Verification:** `verify_audit.py` detects modification/deletion/reordering.

---

## 📂 Project Structure

```text
PriorAuth/
├── src/priorauth/          # Core package (agent, policy, governance)
│   ├── agent_logic.py      # LangGraph orchestration
│   ├── policy_engine.py    # Deterministic rules
│   ├── governance_audit.py # Fairness audit + parity checks
│   └── apps/               # CLI entrypoints (agent + api)
├── apps/ui/                # React/Vite analytics dashboard
├── docker/                 # Dockerfile + compose
├── docs/                   # Architecture, security, runbook, modeling
├── policies/               # JSON Policy Snapshots (version controlled)
├── tests/                  # Pytest suite (unit + safety coverage)
└── output/                 # Generated artifacts (CSVs, logs, reports)
```

---

## 🚦 Getting Started

### Prerequisites

* Python 3.11+
* Conda/Micromamba (optional)
* [Ollama](https://ollama.ai/) (local LLM inference)

### Installation

```bash
git clone https://github.com/your-username/autonomous-prior-auth.git
cd autonomous-prior-auth
pip install -r requirements.lock
```

### Environment

Create a `.env` (or rely on defaults in `src/priorauth/config.py`):

```ini
PA_AUDIT_MODEL_FLAVOR=nemo8b
PA_EMBED_MODEL=kronos483/MedEmbed-large-v0.1:latest
```

### Execution

```bash
# 1) Generate synthetic data
python -m priorauth.apps.agent.chaos_monkey

# 2) Setup vector store
python -m priorauth.apps.agent.setup_rag

# 3) Run batch
python -m priorauth.apps.agent.batch_runner
```

### Runtime Modes (Letters + Offline)

```bash
# Deterministic letters (default, zero LLM calls)
PA_LETTER_MODE=deterministic python -m priorauth.apps.agent.batch_runner

# Ollama letters (requires local Ollama/model)
PA_LETTER_MODE=ollama python -m priorauth.apps.agent.batch_runner

# Allow explicit fallback if Ollama is unavailable
PA_ALLOW_LETTER_FALLBACK=1 PA_LETTER_MODE=ollama python -m priorauth.apps.agent.batch_runner

# Offline deterministic (no external egress)
scripts/run_offline_deterministic.sh

# Offline Ollama (loopback allowed, external blocked)
scripts/run_offline_ollama.sh
```

Offline enforcement note: **offline ≠ “no sockets allowed.”** Offline mode blocks external egress while optionally allowing loopback for local Ollama. Set `PA_OFFLINE_ALLOW_LOCALHOST=false` to block all sockets.

### Tests

```bash
pytest -q
```

Local evidence from this repo run: `reports/pytest.txt`.

---

## 🧪 Test Coverage

The test suite includes:

* Adversarial tests (edge cases, boundary conditions, ambiguous terms)
* Safety tests (zero false approvals for MTC, pregnancy, concurrent GLP-1)
* Policy integration (comorbidity detection, BMI thresholds)
* Statistical tests (Wilson CI, two-proportion z-test edge cases)
* JSON extraction (robust parsing from LLM output)

---

## 📜 Licensing & Attribution

* **Repository:** MIT — see [LICENSE](LICENSE)
* **Third-party licenses:** documented in [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md)
* **Model licensing:** see [docs/THIRD_PARTY_LICENSES.md#models](docs/THIRD_PARTY_LICENSES.md#models)

<a href="docs/THIRD_PARTY_LICENSES.md"><img alt="Third-party licenses" src="https://img.shields.io/badge/Third--Party_Licenses-Documented-2d6a4f?style=for-the-badge"></a> <a href="docs/THIRD_PARTY_LICENSES.md#models"><img alt="Model licensing" src="https://img.shields.io/badge/Model_Licensing-Documented-005f73?style=for-the-badge"></a>

---


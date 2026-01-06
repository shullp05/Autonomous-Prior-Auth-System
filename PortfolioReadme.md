# Peter Shull, PharmD
## Clinical AI Engineer | Healthcare Systems Architect | Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/peter-shull)
[![Email](https://img.shields.io/badge/Email-Contact_Me-teal?style=for-the-badge&logo=gmail)](mailto:shullp05@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Demo-orange?style=for-the-badge)](https://github.com/shullp05)

---

## 🎯 Executive Summary: Bridging Medicine & Machine Learning

I am a **PharmD with clinical and data/AI engineering experience**, focused on building **safe, auditable clinical AI systems**.

This repository demonstrates a **deterministic-first prior authorization system** where the policy engine is the source of truth and LLMs are constrained to narrative and extraction. The repo ships synthetic data only, includes tamper-evident audit logs, and supports offline enforcement for data sovereignty.

**My Value Proposition:**
> I design governance-first clinical AI systems that separate deterministic eligibility from probabilistic narration, so high-stakes decisions remain auditable and reproducible.

---

## 💡 Core Competencies Demonstrated

| **Clinical Strategy & Compliance** | **AI Engineering & Architecture** | **Risk Mitigation & Governance** |
|-----------------------------------|-----------------------------------|----------------------------------|
| ✅ Medical Policy Interpretation | 🐍 Python 3.11+ (Type-Safe) | 🛡️ Zero-Trust Architecture |
| ✅ FHIR Data Interoperability | 🤖 LangGraph Agentic Workflows | ⚖️ Algorithmic Fairness Auditing |
| ✅ Clinical Decision Support | 🔍 RAG (ChromaDB + MedEmbed) | 🧪 Adversarial & Chaos Testing |
| ✅ HIPAA/HITECH Regulatory Knowledge | ⚡ Deterministic Rule Engines | 🔒 PII/PHI Privacy Preservation |

---

## 🏥 Project Showcase: Autonomous Prior Authorization System

**The Challenge:** Prior authorization is costly and delays access to care.
**The Solution:** A deterministic-first, human-in-the-loop agent that processes synthetic claims quickly and produces evidence-backed decisions (see `reports/benchmark.txt` for the latest recorded timing evidence).

### 🚀 Key Technical & Strategic Features

#### 1. Safety-First "Neuro-Symbolic" Architecture
*Demonstrates: Architectural Design, Safety Engineering, Strategic Planning*

I explicitly rejected a "pure LLM" approach due to hallucination risks. Instead, I architected a **Hybrid Engine**:
*   **Deterministic Policy Engine (`src/priorauth/policy_engine.py`)**: Hard-coded policy logic handles binary clinical rules (e.g., *BMI > 30*, *Contraindication = MTC*). This ensures reproducible outcomes for audit and governance.
*   **LLM Agent (`src/priorauth/agent_logic.py`)**: Handles nuance (e.g., *Is "elevated A1c" equivalent to T2DM diagnosis?*) using **LangGraph** for state management and **RAG** for policy grounding.

#### 2. Healthcare-Specific Guardrails & Risk Mitigation
*Demonstrates: Clinical Knowledge, Risk Management, Patient Safety*

I implemented a multi-layer safety net to prevent AI errors from harming patients:
*   **Python Guardrails**: Post-processing logic that *overrules* the LLM if it hallucinates an approval despite a safety contraindication (e.g., Pregnancy, Thyroid Cancer).
*   **Adversarial Testing (`tests/test_adversarial.py`)**: Parameterized cases covering BMI boundaries, safety exclusions, ambiguous terms, and GLP-1 contraindications to stress-test system integrity.

#### 3. Production-Ready RAG (Retrieval Augmented Generation)
*Demonstrates: NLP, Vector Search, Information Retrieval*

*   **Medical Embeddings**: Default embedding model `kronos483/MedEmbed-large-v0.1:latest` (configurable via `PA_EMBED_MODEL`).
*   **Re-ranking**: Integrated `bce-reranker` to optimize context precision, reducing token costs and improving answer quality.
*   **Evidence Support**: Outputs can include policy excerpts and context identifiers; see `reports/rag_rerank_sanity.txt` for retrieval checks.

#### 4. Algorithmic Fairness & Ethical AI
*Demonstrates: Ethics, Statistics, Governance*

*   **Bias Auditing (`governance_audit.py`)**: Automated statistical tests (Wilson Score Interval, Z-tests) run on every batch to detect **False Negative Rate (FNR) Disparity** across demographics.
*   **Compliance**: The system proactively flags potential bias *before* deployment, aligning with emerging FDA and NIST AI frameworks.

#### 5. Credibility Hardening
*Demonstrates: Advanced Verification, Clinical UX, Offline Security*

*   **Offline Enforcement**: Implemented a strict outbound-blocking mode (`src/priorauth/offline_mode.py`) that forbids external network access at runtime while allowing localhost when enabled.
*   **Tamper-Evident Audit**: Architected a cryptographic log (`src/priorauth/audit_logger.py`) using **SHA-256 hash chaining** to ensure decision history is immutable and verifiable.
*   **Coding Integrity Overlay**: Differentiates between "Clinically Eligible" and "Administratively Ready," automating the generation of precise **Physician Queries** for missing codes (e.g., E66.9) rather than issuing flat denials.


---

## 🛠 Tech Stack & Implementation Details

*   **Languages:** Python 3.11 (Strict Typing), JavaScript (React)
*   **AI Frameworks:** LangChain, LangGraph, Ollama (local LLMs; see `docs/MODELING.md` for supported flavors)
*   **Vector Database:** ChromaDB (Local, Zero-Trust)
*   **Testing:** Pytest suite (latest run recorded in `reports/pytest.txt`)
*   **Infrastructure:** Docker, GitHub Actions (CI/CD), Makefiles
*   **Frontend:** React + Vite + D3.js (Interactive Decision Dashboard)

---

## 💼 Why Hire Me?

I bridge the gap that paralyzes most healthcare AI projects: **The gap between "Technical Possibility" and "Clinical Reality."**

*   **For Tech Teams:** I write clean, tested, production-ready Python code and understand modern AI stacks.
*   **For Clinical Teams:** I speak your language (ICD-10, FHIR, Guidelines) and prioritize patient safety above all.
*   **For Leadership:** I build systems that reduce administrative overhead while mitigating legal and reputational risk.

**I am ready to lead your Clinical AI initiatives from concept to compliant production.**

---
*Check out the full code and documentation in the main [README.md](./README.md).*

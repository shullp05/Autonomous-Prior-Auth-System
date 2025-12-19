# Peter Shull, PharmD
## Clinical AI Engineer | Healthcare Systems Architect | Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/petershull/) 
[![Email](https://img.shields.io/badge/Email-Contact_Me-teal?style=for-the-badge&logo=gmail)](mailto:peter@example.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Demo-orange?style=for-the-badge)](https://github.com/petershull)

---

## 🎯 Executive Summary: Bridging Medicine & Machine Learning

I am a **PharmD with 10+ years of clinical experience** and **6+ years in Data/AI Engineering**, specializing in building **safe, compliant, and scalable clinical AI systems**.

This repository serves as a **technical masterwork** demonstrating my ability to solve one of healthcare's most expensive problems—**Prior Authorization (PA)**—using a production-grade **Neuro-Symbolic AI architecture**. unlike standard "AI demos," this system is engineered for **HIPAA compliance**, **auditability**, and **patient safety**.

**My Value Proposition:**
> I don't just "write code" or "prompt LLMs." I design **clinical governance architectures** that allow healthcare organizations to deploy AI safely. I translate complex medical guidelines into deterministic logic and leverage LLMs for reasoning, ensuring technology serves patient care without compromising safety.

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

**The Challenge:** Prior Authorization costs the US healthcare system **$31B annually** and causes **34% of physicians** to report patient harm due to delays.
**The Solution:** An autonomous, human-in-the-loop agent that processes claims in milliseconds with **94%+ accuracy** compared to manual review.

### 🚀 Key Technical & Strategic Features

#### 1. Safety-First "Neuro-Symbolic" Architecture
*Demonstrates: Architectural Design, Safety Engineering, Strategic Planning*

I explicitly rejected a "pure LLM" approach due to hallucination risks. Instead, I architected a **Hybrid Engine**:
*   **Deterministic Policy Engine (`policy_engine.py`)**: Hard-coded Python logic handles binary clinical rules (e.g., *BMI > 30*, *Contraindication = MTC*). This ensures **100% reproducibility** for regulatory audits.
*   **LLM Agent (`agent_logic.py`)**: Handles nuance (e.g., *Is "elevated A1c" equivalent to T2DM diagnosis?*) using **LangGraph** for state management and **RAG** for policy grounding.

#### 2. Healthcare-Specific Guardrails & Risk Mitigation
*Demonstrates: Clinical Knowledge, Risk Management, Patient Safety*

I implemented a multi-layer safety net to prevent AI errors from harming patients:
*   **Python Guardrails**: Post-processing logic that *overrules* the LLM if it hallucinates an approval despite a safety contraindication (e.g., Pregnancy, Thyroid Cancer).
*   **Adversarial Testing (`tests/test_adversarial.py`)**: A "Chaos Monkey" suite that injects 148+ edge cases (borderline vitals, ambiguous terms, malicious inputs) to stress-test system integrity.

#### 3. Production-Ready RAG (Retrieval Augmented Generation)
*Demonstrates: NLP, Vector Search, Information Retrieval*

*   **Medical Embeddings**: Utilized `MedEmbed-large` for domain-specific semantic understanding, outperforming generic OpenAI embeddings on clinical text.
*   **Re-ranking**: Integrated `bce-reranker` to optimize context precision, reducing token costs and improving answer quality.
*   **Citation Tracking**: The system doesn't just decide; it **cites evidence** (e.g., *"Page 12, Section 4.1"*) for every decision, enabling human verification.

#### 4. Algorithmic Fairness & Ethical AI
*Demonstrates: Ethics, Statistics, Governance*

*   **Bias Auditing (`governance_audit.py`)**: Automated statistical tests (Wilson Score Interval, Z-tests) run on every batch to detect **False Negative Rate (FNR) Disparity** across demographics.
*   **Compliance**: The system proactively flags potential bias *before* deployment, aligning with emerging FDA and NIST AI frameworks.

#### 5. "Credibility Hardening" (Phases 10-14)
*Demonstrates: Advanced Verification, Clinical UX, Offline Security*

*   **Offline Enforcement**: Implemented a **"Dead-Man Switch"** (`offline_mode.py`) that strictly forbids network access at runtime, ensuring data sovereignty.
*   **Tamper-Evident Audit**: Architected a cryptographic log (`audit_logger.py`) using **SHA-256 hash chaining** to ensure decision history is immutable and verifiable.
*   **Coding Integrity Overlay**: Differentiates between "Clinically Eligible" and "Administratively Ready," automating the generation of precise **Physician Queries** for missing codes (e.g., E66.9) rather than issuing flat denials.


---

## 🛠 Tech Stack & Implementation Details

*   **Languages:** Python 3.11 (Strict Typing), JavaScript (React)
*   **AI Frameworks:** LangChain, LangGraph, Ollama (Local LLMs: Qwen2.5:14b, Llama 3)
*   **Vector Database:** ChromaDB (Local, Zero-Trust)
*   **Testing:** Pytest (148+ tests), Chaos Engineering principles
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

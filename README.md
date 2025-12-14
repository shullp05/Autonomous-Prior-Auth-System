# 🏥 Autonomous Clinical Prior Authorization Agent (AI-Pa)

> **"A Deterministic-Guardrailed AI Architect for High-Stakes Clinical Decision Making."**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-DD0031?style=for-the-badge&logo=langchain&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange?style=for-the-badge)
![Ollama](https://img.shields.io/badge/LLM-Local%20Inference-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 🚀 Executive Summary

The **Autonomous Clinical Prior Authorization Agent (AI-Pa)** is a production-grade system designed to automate the complex, high-liability process of medical insurance prior authorization. Unlike standard "chatbots," AI-Pa leverages a novel **"Split-Brain" Architecture** that separates **Deterministic Governance** (hard rules, safety exclusions) from **Probabilistic Reasoning** (clinical nuance, unstructured data extraction).

This system solves a critical healthcare inefficiency: manual review of prior auth requests often leads to delays in patient care. AI-Pa ingest patient data (CSV/FHIR), retrieves policy guidelines via **RAG (Retrieval Augmented Generation)**, and executes a multi-step audit to render a verdict (Approve/Deny) with full clinical evidence and appeal generation capabilities.

---

## 🛠️ Architecture & Workflow

The system utilizes a **Stateful Graph Workflow (LangGraph)** to orchestrate the decision process, ensuring auditability and retrievability at every step.

```mermaid
graph TD
    User([User / API Request]) -->|Patient ID + Drug| GraphStart
    
    subgraph "Core Agent Workflow (LangGraph)"
        GraphStart --> NodeRetrieval[🔍 Policy Retrieval (RAG)];
        NodeRetrieval -->|Docs + Scores| NodeAudit[🧠 Clinical Audit (LLM)];
        NodeAudit -->|Extraction + Reasoning| NodeDecision[⚖️ Decision Engine];
        
        subgraph "Split-Brain Verification"
            NodeDecision -->|Check 1| Guardrails{Safety Checks?};
            Guardrails -->|Fail| VerdictDenySafety[❌ Deny: Safety];
            Guardrails -->|Pass| DeterministicLogic{Rule Engine};
            DeterministicLogic -->|Match| VerdictApprove[✅ Approve];
            DeterministicLogic -->|Mismatch| VeridctDenyClinical[❌ Deny: Clinical];
        end
    end

    VerdictApprove --> OutputGen[📝 Generate Report];
    VerdictDenySafety --> OutputGen;
    VeridctDenyClinical --> OutputGen;
    
    OutputGen -->|JSON + PDF| Dashboard((Analytics Dashboard));
```

### Data Flow
1.  **Ingestion**: Patient observations (BMI, Conditions, Meds) are loaded.
2.  **Retrieval**: `ChromaDB` retrieves relevant policy sections using `MedEmbed-large` embeddings + `BCEmbedding` reranking.
3.  **Audit**: A local Large Language Model (e.g., Llama 3, Mistral) analyzes unstructured notes against retrieved policy.
4.  **Governance**: A deterministic Python layer (`policy_engine.py`) cross-verifies the LLM's findings against hardcoded safety rules to prevent hallucinations.
5.  **Output**: Structured JSON decisions + automated Appeal Letters if denied.

---

## 🔬 Technical Deep Dive

### Core Stack
*   **Language**: Python 3.11+
*   **Orchestration**: LangGraph, LangChain
*   **Vector Query**: ChromaDB, BCEmbedding (Reranker)
*   **LLM Serving**: Ollama (Local), HuggingFace Transformers
*   **Validation**: Pydantic (Strict Schema Enforcement)
*   **Testing**: Pytest (Unit & Integration)

### Key Features
*   **🛡️ Deterministic Guardrails**: Prevents the "black box" problem. Safety exclusions (e.g., Pregnancy, MTC) are hard-coded checks that override any AI hallucination.
*   **📚 Dynamic Policy Parsing**: Automatically parses raw text guidelines (`UpdatedPAGuidelines.txt`) into structured JSON snapshots using SHA-256 hashing for version control.
*   **📉 RAG with Reranking**: Implements a two-stage retrieval process (Vector Search -> Cross-Encoder Reranking) to ensure the AI sees only the most relevant policy clauses.
*   **🕵️ Governance Audit**: A dedicated module that "audits the auditor," comparing the AI's probabilistic output against a ground-truth deterministic evaluation.

### Advanced Techniques
*   **Lazy Loading & Singleton Patterns**: Heavy models (e.g., Rerankers) are lazy-loaded with thread-safe locks (`threading.Lock`) to optimize resource usage in concurrent batch processing.
*   **Policy "Bucket" Sort**: Custom sorting logic (`_policy_aware_sort_docs`) prioritizes document chunks based on semantic sections (e.g., prioritizing "Contraindications" over "Documentation" during safety checks).
*   **Chaos Engineering**: Includes a `chaos_monkey.py` module to generate synthetic, adversarial patient data to stress-test the agent's logic.

---

## 📂 Project Structure

```text
/root/projects/PriorAuth
├── agent_logic.py         # 🧠 Core Agent orchestration (LangGraph state machine)
├── policy_engine.py       # 🛡️ Deterministic Rule Engine (The "Safety Brain")
├── governance_audit.py    # ⚖️ Audit module to verify AI vs. Rule Logic
├── batch_runner.py        # 🚀 High-throughput batch processing entry point
├── policy_snapshot.py     # 📄 Policy parsing and versioning (Text -> JSON)
├── policy_utils.py        # 🛠️ Shared utilities for normalization/matching
├── config.py              # ⚙️ Centralized configuration management
├── policies/              # 📂 JSON Policy Snapshots (Version Controlled)
├── tests/                 # 🧪 Pytest suite (Unit, Integration, Safety)
└── output/                # 📊 Generated artifacts (CSVs, Logs, Reports)
```

---

## 🚦 Getting Started

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.ai/) (for local LLM inference)

### Installation
1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/autonomous-prior-auth.git
    cd autonomous-prior-auth
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment**
    Create a `.env` file (or rely on defaults in `config.py`):
    ```ini
    PA_AUDIT_MODEL=mistral
    PA_EMBED_MODEL=kronos483/MedEmbed-large-v0.1:latest
    ```

### Execution
**Run a Batch Simulation:**
```bash
# 1. Generate Synthetic Data
python chaos_monkey.py

# 2. Setup Vector Store
python setup_rag.py

# 3. Run the Agent
python batch_runner.py
```

**Run Verification Tests:**
```bash
pytest tests/
```

---

*Engineered with precision. Designed for trust.*

# Architecture & Design Document

> **Status:** Active
> **Version:** 2.1
> **Author:** Principal System Architect

## 1. High-Level Architecture

The **Autonomous Prior Authorization Agent (AI-Pa)** implements a **Hybrid Neuro-Symbolic Architecture**. It combines the probabilistic reasoning capabilities of Large Language Models (LLMs) with the safety guarantees of deterministic rule engines.

The system is designed as a **pipeline-driven processing graph**, utilizing `LangGraph` to manage state transitions between retrieval, reasoning, and decision-making nodes. This approach decouples the "Brain" (LLM) from the "Law" (Policy Engine), mitigating the risk of AI hallucination in high-liability clinical settings.

### Design Pattern: The "Split-Brain" Controller
The architecture explicitly divides responsibility:
1.  **System 1 (Deterministic):** Fast, rule-based execution. Parses structured text, regex matches medical codes (ICD-10/LOINC), and enforces "Hard Stop" safety exclusions.
    *   *Implemented in:* `src/priorauth/policy_engine.py`
2.  **System 2 (Probabilistic):** Slow, deliberative reasoning. Synthesizes unstructured clinical notes, extracts intent, and maps complex patient histories to policy criteria.
    *   *Implemented in:* `src/priorauth/agent_logic.py` (Audit/Reasoning Nodes)

---

## 2. Component Design (C4 Model)

### **Context Level**
*   **User/API**: Submits a `PatientID` and `DrugRequest`.
*   **AI-Pa System**: Returns a `Decision` (Approve/Deny) and `EvidenceBundle`.
*   **External Systems**: Electronic Health Record (EHR) Database (CSV/FHIR), Policy Knowledge Base.

### **Container Level**

#### 1. Data Ingestion Layer (`src/priorauth/etl_pipeline.py`, `src/priorauth/policy_snapshot.py`)
*   **Responsibility**: Normalizing heterogeneous input data (CSV, Text) into canonical internal representations.
*   **Key Interaction**: Parses `UpdatedPAGuidelines.txt` into a hashed JSON snapshot (`policies/RX-WEG-2025.json`) to serve as the **Single Source of Truth**.

#### 2. Retrieval Engine (`src/priorauth/apps/agent/setup_rag.py`, `chroma_db/`)
*   **Responsibility**: Semantic search.
*   **Component**: **Hybrid RAG**. Uses MedEmbed vector similarity (k=25 by default) with optional BCE reranking (`BCEmbedding`) before selecting the top-k policy atoms (k=8 by default).
*   **Why**: standard embeddings can miss nuanced medical negation (e.g., "History of depression" vs. "No history of depression"). Reranking reduces distractors when enabled.

#### 3. The Agentic Core (`src/priorauth/agent_logic.py`)
*   **Responsibility**: Orchestration.
*   **Pattern**: State Machine (`LangGraph`).
*   **Nodes**:
    *   `retrieve_policy`: Fetches context.
    *   `clinical_audit`: Invokes the LLM to extract clinical parameters (BMI, Comorbidities).
    *   `make_decision`: Merges LLM findings with the Policy Engine.

#### 4. The Safety Layer (`src/priorauth/policy_engine.py`, `src/priorauth/governance_audit.py`)
*   **Responsibility**: Compliance and Guardrails.
*   **Logic**: Pure Python functions that operate on normalized strings. Zero AI involvement.
*   **Role**: The "Veto Power". If the LLM says "Approve" but the Python layer detects "Pregnancy" in the active condition list, the Safety Layer forces a denial.

---

## 3. Architectural Decision Records (ADR)

### ADR-001: Hybrid Neuro-Symbolic Verification ("Split-Brain")
*   **Context**: Clinical policies typically contain hard exclusions (e.g., "Contraindicated in Pregnancy"). LLMs are non-deterministic and can miss these or be "convinced" to overlook them via prompt injection.
*   **Decision**: Implement a parallel, deterministic verification layer (`src/priorauth/policy_engine.py`) that runs *after* the LLM reasoning to validate the verdict.
*   **Trade-off**:
    *   *Pro*: Mathematically guaranteed safety for known exclusions.
    *   *Con*: Code duplication. Logic for "Pregnancy" exists in both the Policy text (for LLM) and Python constants (for Engine). mitigated via `src/priorauth/policy_utils.py` centralization.

### ADR-002: Two-Stage RAG (Retrieval + Optional Reranking)
*   **Context**: Medical policies are dense. A simple top-k vector search often retrieves irrelevant sections that share keywords but not semantic relevance.
*   **Decision**: Use `MedEmbed` for initial retrieval (k=25), then (optionally) apply a BCE cross-encoder (`BCEmbedding`) to rerank and filter down to top-k (k=8) with a score floor and minimum-doc guardrail.
*   **Trade-off**:
    *   *Pro*: Higher precision; fewer distractors passed to the LLM.
    *   *Con*: Added latency and memory footprint when the reranker is enabled.

### ADR-003: Dynamic Policy Parsing vs. Static Configuration
*   **Context**: Insurance guidelines change frequently (quarterly/annually). Hardcoding rules into Python classes makes updates slow and requires developer intervention.
*   **Decision**: Implement a text-to-JSON parser (`src/priorauth/policy_snapshot.py`) that reads a human-readable text file and compiles it into a machine-readable schema at runtime/build time.
*   **Trade-off**:
    *   *Pro*: Agility. Clinical Ops can update `UpdatedPAGuidelines.txt` without touching code.
    *   *Con*: Parser fragility. If the text formatting changes significantly, the parser breaks (mitigated via Hash consistency checks).

---

## 4. Data Flow & State Management

### State Synchronization
State is managed via `TypedDict` (`AgentState`) passed immutably between graph nodes.
```python
class AgentState(TypedDict):
    patient_id: str
    patient_data: dict      # Raw EHR data
    policy_docs: list       # RAG Context
    audit_findings: dict    # LLM Structured Output
    decision: str           # Final Verdict
```

### Flow Sequence
1.  **Load**: `src/priorauth/apps/agent/batch_runner.py` lazy-loads Global Models (Embed, Rerank, LLM).
2.  **Input**: User -> `AgentState` initialized.
3.  **RAG**: `retrieve_policy` queries ChromaDB -> Context injected into `AgentState`.
4.  **Audit**: LLM accepts (Context + Patient Data) -> Outputs JSON (`audit_findings`).
5.  **Guardrail**: `make_decision` executes `policy_engine.evaluate(audit_findings)`.
    *   *Sync*: If LLM & Engine disagree, the Engine's verdict takes precedence (Safety First).
6.  **Persistence**: Results written to `output/` CSVs.

---

## 5. Security & Performance Strategy

### Security (Safety & Compliance)
*   **Input Sanitization**: All terms are normalized via `policy_utils.normalize` (lower-case, strip, strict word boundary regex) before matching.
*   **Model Isolation**: The safety layer can run with offline enforcement enabled, blocking outbound network access at the process level.
*   **Type Safety**: Use of `Pydantic` models (`AuditResult`) ensures LLM outputs strictly conform to required schemas (BMI as float, Enums for verdicts).

### Performance
*   **Lazy Loading**: Expensive models (e.g., BCE Reranker) are initialized only on the first request (`_get_bce_model`).
*   **Thread Locking**: `_BCE_LOCK` prevents race conditions during lazy initialization in multithreaded environments.
*   **Caching**: `ChromaDB` functions as a semantic cache for policy documents.

---

## 6. Future Scalability

To scale this system 100x (from Batch CSV to Real-time API):

1.  **Database Migration**:
    *   Move from `pandas` CSV reads to a proper SQL Database (PostgreSQL) for Patient Data.
    *   Migrate `ChromaDB` (local) to a distributed Vector Store (e.g., Qdrant Cloud or Pinecone).
2.  **Async Worker Queues**:
    *   Decouple ingestion from processing. Use `Celery` + `Redis` to handle Authorizations asynchronously. The `src/priorauth/apps/agent/batch_runner.py` loop becomes a Producer; the Agent becomes a Consumer.
3.  **Model Serving**:
    *   Move local `Ollama` inference to a dedicated Inference Server (e.g., vLLM or TGI) to handle concurrent LLM requests with higher throughput.
    *   The `src/priorauth/agent_logic.py` would call an HTTP endpoint for generation rather than invoking a local library.

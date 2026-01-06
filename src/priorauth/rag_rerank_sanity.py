#!/usr/bin/env python
"""
rag_rerank_sanity.py

Sanity suite for Wegovy (RX-WEG-2025) RAG + BCE reranker behavior.

This focuses on three questions per scenario:
1) Coverage: Are the expected policy atoms present anywhere in the filtered set?
2) Priority: Are they actually in the top-K docs that would be shown to Qwen?
3) Performance: How long do vector search and BCE rerank take?
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from priorauth import paths
from priorauth.agent_logic import _build_policy_query, _format_policy_evidence
from priorauth.bce_reranker import rerank_bce
from priorauth.config import EMBED_MODEL_NAME
from priorauth.policy_engine import evaluate_eligibility
from priorauth.policy_snapshot import POLICY_ID, SNAPSHOT_PATH, load_policy_snapshot
from priorauth.schema_validation import validate_policy_snapshot

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PERSIST_DIR = paths.CHROMA_DIR
COLLECTION_NAME = "priorauth_policies"

K_VECTOR = 25           # initial vector retrieval
TOP_K_DOCS = 8          # how many docs Qwen actually sees
SCORE_FLOOR = 0.35      # BCE score floor
MIN_DOCS = 3            # minimum docs after filtering

logger = logging.getLogger("rag_rerank_sanity")


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    patient: dict
    expected_sections: list[str]
    priority_n: int


@dataclass
class ScenarioResult:
    name: str
    bmi: float | None
    verdict: str
    policy_path: str | None

    vector_sections: list[str]
    rerank_sections_top: list[str]
    filtered_sections_all: list[str]
    llm_sections: list[str]

    scores_all: list[float]

    expected_sections: list[str]
    hits_vector: dict[str, bool]
    hits_llm: dict[str, bool]
    coverage_hits: dict[str, bool]

    coverage_ok: bool
    priority_ok: bool
    pass_flag: bool

    vector_ms: float
    rerank_ms: float


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _ensure_vectorstore() -> Chroma:
    if not os.path.isdir(PERSIST_DIR):
        raise RuntimeError(
            f"ChromaDB directory '{PERSIST_DIR}' not found. "
            "Run `python -m priorauth.apps.agent.setup_rag` to build the Wegovy policy index first."
        )

    logger.info("Using embedding model for sanity tests: %s", EMBED_MODEL_NAME)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    return Chroma(
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def _sections_from_docs(docs: Sequence[Document]) -> list[str]:
    sections: list[str] = []
    for d in docs:
        sec = (d.metadata or {}).get("section") or "unknown"
        sections.append(str(sec))
    return sections


def _apply_score_floor(scored_docs, floor: float, min_docs: int):
    """
    scored_docs: List[(Document, score)]
    Returns: (filtered_docs, filtered_scores)
    """
    if not scored_docs:
        return [], []

    docs, scores = zip(*scored_docs)
    docs = list(docs)
    scores = list(scores)

    # Filter by score
    filtered_docs: list[Document] = []
    filtered_scores: list[float] = []
    for d, s in zip(docs, scores):
        if s >= floor:
            filtered_docs.append(d)
            filtered_scores.append(float(s))

    # Enforce minimum
    if len(filtered_docs) < min_docs:
        filtered_docs = docs[:min_docs]
        filtered_scores = scores[:min_docs]

    return filtered_docs, filtered_scores


def _check_expected_sections(sections: Sequence[str],
                             expected: Sequence[str]) -> dict[str, bool]:
    """
    For each expected snippet, check if it appears in any section string.
    """
    joined = " || ".join(sections).lower()
    return {e: (e.lower() in joined) for e in expected}


def _is_ollama_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if isinstance(exc, ConnectionError):
        return True
    if "failed to connect to ollama" in msg:
        return True
    if "http://127.0.0.1:11434" in msg and ("connect" in msg or "connection" in msg):
        return True
    if "socket: operation not permitted" in msg:
        return True
    if "connection refused" in msg:
        return True
    if "ollama" in msg and "not found" in msg and "model" in msg:
        return True
    return False


def _preflight_ollama_embeddings() -> tuple[bool, str | None]:
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
        embeddings.embed_query("healthcheck")
        return True, None
    except Exception as exc:
        if _is_ollama_unavailable(exc):
            return False, str(exc)
        raise


def _cond(
    name: str,
    icd10_dx: str | None = None,
    icd10_bmi: str | None = None,
) -> dict[str, str | None]:
    return {
        "condition_name": name,
        "icd10_dx": icd10_dx,
        "icd10_bmi": icd10_bmi,
    }


def _filter_docs_for_policy_path(
    docs: Sequence[Document],
    policy_path: str | None,
) -> list[Document]:
    """
    Policy-aware filter that restricts AND reorders sections based on the
    deterministic policy_path.

    - BMI30_OBESITY      → eligibility:pathway1 + obesity dx + documentation
    - BMI27_COMORBIDITY  → eligibility:pathway2 + overweight dx + comorbidities + documentation
    - SAFETY_EXCLUSION   → safety_exclusions:* + drug_conflicts:glp1_glp1_gip,
                           with critical atoms forced to the front.
    """
    if not docs or not policy_path:
        return list(docs)

    def section(d: Document) -> str:
        return str((d.metadata or {}).get("section") or "")

    # Simple allow-lists per policy_path
    if policy_path == "BMI30_OBESITY":
        allowed_prefixes = (
            "documentation:requirements",
            "eligibility:pathway1",
            "diagnosis:obesity_strings",
        )
        priority_order: list[str] = [
            "eligibility:pathway1",
            "diagnosis:obesity_strings",
            "documentation:requirements",
        ]
    elif policy_path == "BMI27_COMORBIDITY":
        allowed_prefixes = (
            "documentation:requirements",
            "eligibility:pathway2",
            "diagnosis:overweight_strings",
            "comorbidity:",
        )
        priority_order = [
            "eligibility:pathway2",
            "comorbidity:hypertension",
            "diagnosis:overweight_strings",
            "documentation:requirements",
        ]
    elif policy_path == "SAFETY_EXCLUSION":
        allowed_prefixes = (
            "safety_exclusions:",
            "drug_conflicts:glp1_glp1_gip",
        )
        priority_order = [
            "safety_exclusions:mtc_men2",
            "safety_exclusions:pregnancy_nursing",
            "safety_exclusions:concurrent_glp1",
            "drug_conflicts:glp1_glp1_gip",
        ]
    elif policy_path == "AMBIGUITY_MANUAL_REVIEW":
        allowed_prefixes = (
            "ambiguity:",
            "eligibility:",
            "diagnosis:obesity_strings",
            "diagnosis:overweight_strings",
            "documentation:requirements",
        )
        priority_order = [
            "ambiguity:",
            "eligibility:pathway1",
            "eligibility:pathway2",
            "diagnosis:obesity_strings",
            "diagnosis:overweight_strings",
            "documentation:requirements",
        ]
    else:
        # Unknown or non-Wegovy path → do not restrict further
        return list(docs)

    # First, apply the allow-list filter
    allowed: list[Document] = []
    for d in docs:
        sec = section(d)
        if any(sec.startswith(prefix) for prefix in allowed_prefixes):
            allowed.append(d)

    if not allowed:
        # If the filter nukes everything, fall back to original docs
        return list(docs)

    # Then, apply deterministic priority within the allowed set:
    #   - anything whose section starts with a priority entry gets bubbled up,
    #   - rest keep the BCE order behind them.
    priority_docs: list[Document] = []
    other_docs: list[Document] = []

    for d in allowed:
        sec = section(d)
        if any(sec.startswith(p) for p in priority_order):
            priority_docs.append(d)
        else:
            other_docs.append(d)

    # Maintain original BCE ordering inside each bucket
    return priority_docs + other_docs


# ---------------------------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------------------------

def build_scenarios() -> list[ScenarioConfig]:
    """
    Synthetic test patients designed to exercise key policy pathways.

    NOTE: expected_sections MUST match your actual section IDs in snapshot /
          _section_documents, not the names you wish existed.
    """
    return [
        # 1) Obesity pathway: BMI >= 30 + obesity dx
        ScenarioConfig(
            name="BMI_35_OBESITY_PATHWAY",
            patient={
                "latest_bmi": "35.2",
                "conditions": [
                    _cond("Obesity due to excess calories (E66.0)", icd10_dx="E66.0", icd10_bmi="Z68.35"),
                ],
                "meds": [],
            },
            expected_sections=[
                "eligibility:pathway1",
                "diagnosis:obesity_strings",
            ],
            priority_n=5,
        ),
        # 2) Overweight + HTN comorbidity
        ScenarioConfig(
            name="BMI_28_HTN_OVERWEIGHT_PATHWAY",
            patient={
                "latest_bmi": "28.4",
                "conditions": [
                    _cond("Overweight (E66.3)", icd10_dx="E66.3", icd10_bmi="Z68.28"),
                    _cond("Hypertension"),
                ],
                "meds": [],
            },
            expected_sections=[
                "eligibility:pathway2",
                "comorbidity:hypertension",
            ],
            priority_n=5,
        ),
        # 3) Safety exclusion: MTC + concurrent GLP-1 (Ozempic)
        ScenarioConfig(
            name="SAFETY_MTC_OZEMPIC_CONTRAINDICATED",
            patient={
                "latest_bmi": "33.0",
                "conditions": [
                    _cond("Personal or family history of Medullary Thyroid Carcinoma"),
                    _cond("Type 2 diabetes mellitus"),
                ],
                "meds": ["Ozempic"],
            },
            expected_sections=[
                "safety_exclusions:mtc_men2",
                "safety_exclusions:concurrent_glp1",
                "drug_conflicts:glp1_glp1_gip",
            ],
            priority_n=8,
        ),
        # 4) Ambiguous thyroid malignancy – must not auto-safety deny
        ScenarioConfig(
            name="AMBIG_THYROID_NEEDS_REVIEW",
            patient={
                "latest_bmi": "31.0",
                "conditions": [
                    _cond("Thyroid cancer"),
                    _cond("Obesity due to excess calories (E66.0)", icd10_dx="E66.0", icd10_bmi="Z68.31"),
                ],
                "meds": [],
            },
            expected_sections=[
                "eligibility:pathway1",
                "diagnosis:obesity_strings",
                "ambiguity:thyroid",
            ],
            priority_n=8,
        ),
        # 5) Borderline blood pressure (no true HTN)
        ScenarioConfig(
            name="AMBIG_BP_BORDERLINE",
            patient={
                "latest_bmi": "28.0",
                "conditions": [
                    _cond("Overweight (E66.3)", icd10_dx="E66.3", icd10_bmi="Z68.28"),
                    _cond("Elevated blood pressure"),
                ],
                "meds": [],
            },
            expected_sections=[
                "eligibility:pathway2",
                "diagnosis:overweight_strings",
                "ambiguity:bp_borderline",
            ],
            priority_n=8,
        ),
        # 6) Pregnancy safety exclusion
        ScenarioConfig(
            name="SAFETY_PREGNANCY_EXCLUSION",
            patient={
                "latest_bmi": "32.0",
                "conditions": [
                    _cond("Pregnancy, unspecified"),
                    _cond("Obesity due to excess calories (E66.0)", icd10_dx="E66.0", icd10_bmi="Z68.32"),
                ],
                "meds": [],
            },
            expected_sections=[
                "safety_exclusions:pregnancy_nursing",
            ],
            priority_n=8,
        ),
    ]


# ---------------------------------------------------------------------------
# CORE TEST LOGIC
# ---------------------------------------------------------------------------

def run_scenario(cfg: ScenarioConfig, vectordb: Chroma) -> ScenarioResult:
    # Deterministic engine
    det = evaluate_eligibility(cfg.patient)
    bmi = det.bmi_numeric
    verdict = det.verdict
    policy_path = getattr(det, "policy_path", None)

    logger.info("=== Scenario: %s ===", cfg.name)
    logger.info(
        "Deterministic verdict=%s | BMI=%.2f | policy_path=%s",
        verdict,
        bmi if bmi is not None else -1.0,
        policy_path,
    )

    # Build query the same way the agent does
    query = _build_policy_query(det, cfg.patient, "Wegovy")
    logger.info("Query: %s", query)

    # 1) Vector search
    t0 = time.time()
    vector_docs: list[Document] = vectordb.similarity_search(
        query,
        k=K_VECTOR,
        filter={"policy_id": POLICY_ID},
    )
    t1 = time.time()
    vector_ms = (t1 - t0) * 1000.0
    vector_sections = _sections_from_docs(vector_docs[:TOP_K_DOCS])
    logger.info("[Vector] %d docs (%.1f ms), top sections: %s",
                len(vector_docs), vector_ms, vector_sections[:5])

    if not vector_docs:
        logger.warning("[Vector] No docs returned for scenario %s", cfg.name)
        return ScenarioResult(
            name=cfg.name,
            bmi=bmi,
            verdict=verdict,
            policy_path=policy_path,
            vector_sections=[],
            rerank_sections_top=[],
            filtered_sections_all=[],
            llm_sections=[],
            scores_all=[],
            expected_sections=cfg.expected_sections,
            hits_vector=dict.fromkeys(cfg.expected_sections, False),
            hits_llm=dict.fromkeys(cfg.expected_sections, False),
            coverage_hits=dict.fromkeys(cfg.expected_sections, False),
            coverage_ok=False,
            priority_ok=False,
            pass_flag=False,
            vector_ms=vector_ms,
            rerank_ms=0.0,
        )

    # 2) BCE re-rank
    t2 = time.time()
    scored = rerank_bce(query, vector_docs)
    t3 = time.time()
    rerank_ms = (t3 - t2) * 1000.0

    if not scored:
        logger.warning("[Rerank] No scores returned, falling back to vector order")
        scored = [(d, 0.0) for d in vector_docs]

    docs_all = [d for d, _ in scored]
    scores_all = [float(s) for _, s in scored]
    rerank_sections_top = _sections_from_docs(docs_all[:TOP_K_DOCS])
    logger.info(
        "[Rerank] %d docs (%.1f ms), top sections: %s",
        len(docs_all),
        rerank_ms,
        rerank_sections_top[:5],
    )
    logger.info("[Rerank] Top 10 scores: %s",
                [f"{s:.3f}" for s in scores_all[:10]])

    # 3) Apply score floor
    filtered_docs, filtered_scores = _apply_score_floor(scored, SCORE_FLOOR, MIN_DOCS)

    # 4) Policy-aware filter based on deterministic policy_path
    filtered_docs = _filter_docs_for_policy_path(filtered_docs, policy_path)
    filtered_sections_all = _sections_from_docs(filtered_docs)
    logger.info(
        "[Filter] Kept %d docs after score floor + policy filter; sections: %s",
        len(filtered_docs),
        filtered_sections_all,
    )

    # 5) Take top-K for the LLM
    llm_docs = filtered_docs[:TOP_K_DOCS]
    llm_sections = _sections_from_docs(llm_docs)
    logger.info("[LLM] Will pass %d docs; sections: %s",
                len(llm_docs), llm_sections)

    # Touch policy_text to ensure formatting is valid (not used further)
    _ = _format_policy_evidence(llm_docs)

    # 6) Expectations
    hits_vector = _check_expected_sections(vector_sections, cfg.expected_sections)
    hits_llm = _check_expected_sections(llm_sections, cfg.expected_sections)
    coverage_hits = _check_expected_sections(filtered_sections_all, cfg.expected_sections)

    coverage_ok = all(coverage_hits.values())
    priority_ok = all(hits_llm.values())
    pass_flag = coverage_ok and priority_ok

    return ScenarioResult(
        name=cfg.name,
        bmi=bmi,
        verdict=verdict,
        policy_path=policy_path,
        vector_sections=vector_sections,
        rerank_sections_top=rerank_sections_top,
        filtered_sections_all=filtered_sections_all,
        llm_sections=llm_sections,
        scores_all=scores_all,
        expected_sections=cfg.expected_sections,
        hits_vector=hits_vector,
        hits_llm=hits_llm,
        coverage_hits=coverage_hits,
        coverage_ok=coverage_ok,
        priority_ok=priority_ok,
        pass_flag=pass_flag,
        vector_ms=vector_ms,
        rerank_ms=rerank_ms,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Sanity: snapshot exists and matches schema
    snapshot = load_policy_snapshot(SNAPSHOT_PATH, POLICY_ID)
    validate_policy_snapshot(snapshot)
    logger.info(
        "Loaded policy snapshot %s (effective %s, source_hash=%s)",
        snapshot["policy_id"],
        snapshot["effective_date"],
        snapshot["source_hash"],
    )

    vectordb = _ensure_vectorstore()
    ok, err = _preflight_ollama_embeddings()
    if not ok:
        logger.warning("Ollama unavailable; skipping RAG/rerank sanity run: %s", err)
        summary = {
            "config": {
                "policy_id": POLICY_ID,
                "embed_model": EMBED_MODEL_NAME,
                "collection": COLLECTION_NAME,
                "k_vector": K_VECTOR,
                "top_k_docs": TOP_K_DOCS,
                "score_floor": SCORE_FLOOR,
                "min_docs": MIN_DOCS,
            },
            "status": "OLLAMA_UNAVAILABLE",
            "error": err,
            "scenarios": [],
        }
        print("\n=== RAG / RERANK SANITY SUMMARY (JSON) ===")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    scenarios = build_scenarios()

    results: list[ScenarioResult] = []
    try:
        for cfg in scenarios:
            res = run_scenario(cfg, vectordb)
            results.append(res)
    except Exception as exc:
        if _is_ollama_unavailable(exc):
            logger.warning(
                "Ollama unavailable; short-circuiting after %d/%d scenarios: %s",
                len(results),
                len(scenarios),
                exc,
            )
            summary = {
                "config": {
                    "policy_id": POLICY_ID,
                    "embed_model": EMBED_MODEL_NAME,
                    "collection": COLLECTION_NAME,
                    "k_vector": K_VECTOR,
                    "top_k_docs": TOP_K_DOCS,
                    "score_floor": SCORE_FLOOR,
                    "min_docs": MIN_DOCS,
                },
                "status": "OLLAMA_UNAVAILABLE",
                "error": str(exc),
                "scenarios_completed": len(results),
                "scenarios_total": len(scenarios),
                "scenarios": [asdict(r) for r in results],
            }
            print("\n=== RAG / RERANK SANITY SUMMARY (JSON) ===")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        raise

    summary = {
        "config": {
            "policy_id": POLICY_ID,
            "embed_model": EMBED_MODEL_NAME,
            "collection": COLLECTION_NAME,
            "k_vector": K_VECTOR,
            "top_k_docs": TOP_K_DOCS,
            "score_floor": SCORE_FLOOR,
            "min_docs": MIN_DOCS,
        },
        "scenarios": [asdict(r) for r in results],
    }

    print("\n=== RAG / RERANK SANITY SUMMARY (JSON) ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

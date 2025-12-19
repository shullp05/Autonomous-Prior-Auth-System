# bce_reranker.py
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, List, Tuple

# IMPORTANT:
# - Keep this module safe to import even when BCEmbedding / LangChain are not installed,
#   so deterministic runs (no LLM deps) don't explode at import-time.
# - Import heavy deps lazily inside functions.

if TYPE_CHECKING:
    from langchain_core.documents import Document  # pragma: no cover


# Model lifecycle locks
_BCE_INIT_LOCK = threading.Lock()
_BCE_INFER_LOCK = threading.Lock()

_BCE_MODEL: Any = None
_BCE_IMPORT_ERROR: Exception | None = None


def _load_config() -> tuple[bool, str, str]:
    """
    Load rerank config from central config if available; fall back safely to env.
    Returns: (enabled, model_name, device)
    """
    try:
        # Prefer centralized config
        from config import PA_ENABLE_RERANK, PA_RERANK_MODEL, PA_RERANK_DEVICE  # type: ignore

        enabled = bool(PA_ENABLE_RERANK)
        model_name = str(PA_RERANK_MODEL)
        device = str(PA_RERANK_DEVICE)
        return enabled, model_name, device
    except Exception:
        # Safe fallback (do NOT raise just because config import failed)
        import os

        enabled = os.getenv("PA_ENABLE_RERANK", "true").lower() == "true"
        model_name = os.getenv("PA_RERANK_MODEL", "maidalun1020/bce-reranker-base_v1")
        device = os.getenv("PA_RERANK_DEVICE", "cuda")
        return enabled, model_name, device


def _normalize_device(device: str) -> str:
    """
    Normalize device strings for BCEmbedding.
    Accepts: cpu, cuda, cuda:0, gpu, etc.
    """
    d = (device or "").strip().lower()
    if not d:
        return "cuda"

    # Common aliases
    if d in {"cpu", "host"}:
        return "cpu"
    if d in {"gpu", "cuda"}:
        return "cuda"

    # Allow explicit cuda index (cuda:0, cuda:1)
    if d.startswith("cuda:"):
        return d

    # Conservative fallback: if it contains 'cpu' => cpu, else cuda
    return "cpu" if "cpu" in d else "cuda"


def _lazy_imports() -> tuple[Any, Any]:
    """
    Lazily import BCEmbedding + Document.
    Raises a clear error only when reranking is actually invoked.
    """
    global _BCE_IMPORT_ERROR
    if _BCE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "BCEmbedding dependency is not available, but reranking was requested. "
            "Install 'BCEmbedding' (and its torch deps) or disable reranking with PA_ENABLE_RERANK=false."
        ) from _BCE_IMPORT_ERROR

    try:
        from BCEmbedding import RerankerModel  # type: ignore
        from langchain_core.documents import Document  # type: ignore

        return RerankerModel, Document
    except Exception as e:  # pragma: no cover
        _BCE_IMPORT_ERROR = e
        raise RuntimeError(
            "Failed to import BCEmbedding and/or langchain_core. "
            "Install missing deps or disable reranking with PA_ENABLE_RERANK=false."
        ) from e


def get_bce_reranker() -> Any:
    """
    Lazily load BCE reranker model.
    Thread-safe init, cached singleton.
    """
    global _BCE_MODEL
    enabled, model_name, device = _load_config()

    if not enabled:
        return None

    with _BCE_INIT_LOCK:
        if _BCE_MODEL is None:
            RerankerModel, _ = _lazy_imports()
            _BCE_MODEL = RerankerModel(
                model_name_or_path=model_name,
                device=_normalize_device(device),
            )
        return _BCE_MODEL


def rerank_bce(query: str, docs: List["Document"]) -> List[Tuple["Document", float]]:
    """
    Return docs with BCE relevance scores in descending order.
    If reranking is disabled, returns docs in original order with score 0.0.
    """
    if not docs:
        return []

    enabled, _, _ = _load_config()
    if not enabled:
        return [(d, 0.0) for d in docs]

    model = get_bce_reranker()
    if model is None:
        return [(d, 0.0) for d in docs]

    # Defensive: docs should have page_content, but don't trust callers.
    passages: List[str] = []
    for d in docs:
        try:
            passages.append(getattr(d, "page_content", "") or "")
        except Exception:
            passages.append("")

    pairs = [[query or "", p] for p in passages]

    # Some model implementations aren't thread-safe during inference
    with _BCE_INFER_LOCK:
        scores = model.compute_score(pairs)

    # Normalize score container (could be list, numpy array, torch tensor, etc.)
    try:
        scores_list = list(scores)
    except Exception:
        scores_list = [0.0] * len(docs)

    # Guard against mismatched lengths
    if len(scores_list) != len(docs):
        scores_list = [0.0] * len(docs)

    scored_docs: List[Tuple["Document", float]] = []
    for d, s in zip(docs, scores_list):
        try:
            scored_docs.append((d, float(s)))
        except Exception:
            scored_docs.append((d, 0.0))

    scored_docs.sort(key=lambda t: t[1], reverse=True)
    return scored_docs

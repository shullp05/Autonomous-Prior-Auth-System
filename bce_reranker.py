# bce_reranker.py
from __future__ import annotations

import os
import threading
from typing import List, Tuple

from BCEmbedding import RerankerModel  # type: ignore
from langchain_core.documents import Document

_BCE_LOCK = threading.Lock()
_BCE_MODEL: RerankerModel | None = None


def _get_device() -> str:
    """
    Device for the reranker. Default = cuda
    Set PA_RERANK_DEVICE=CPU if you have limited VRAM and need to run on CPU.
    """
    return os.getenv("PA_RERANK_DEVICE", "cuda").lower()


def get_bce_reranker() -> RerankerModel:
    """
    Lazily load maidalun1020/bce-reranker-base_v1 as a BCEmbedding RerankerModel.
    """
    global _BCE_MODEL
    with _BCE_LOCK:
        if _BCE_MODEL is None:
            model_name = os.getenv(
                "PA_RERANK_MODEL",
                "maidalun1020/bce-reranker-base_v1",  # canonical HF id
            )
            device = _get_device()
            _BCE_MODEL = RerankerModel(
                model_name_or_path=model_name,
                device=device,
            )
        return _BCE_MODEL


def rerank_bce(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Return docs with real BCE scores in descending order.
    """
    if not docs:
        return []

    model = get_bce_reranker()
    passages = [d.page_content for d in docs]
    pairs = [[query, p] for p in passages]

    # BCEmbedding returns higher = more relevant
    scores = model.compute_score(pairs)  # List[float]

    scored_docs: List[Tuple[Document, float]] = list(zip(docs, scores))
    scored_docs.sort(key=lambda t: t[1], reverse=True)
    return scored_docs

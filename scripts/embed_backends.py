from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class EmbedderInfo:
    backend: str
    model: str
    dim: int


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    # vectors: (n, d)
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return vectors / denom


def embed_texts(
    texts: List[str],
    *,
    backend: str,
    model: str,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True,
) -> tuple[np.ndarray, EmbedderInfo]:
    backend = (backend or "").strip().lower()

    if backend in ("fastembed", "onnx"):
        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "fastembed backend selected but not installed. Install with: pip install -r scripts/requirements-fastembed.txt"
            ) from e

        # fastembed returns an iterator of vectors
        # Model defaults are handled by fastembed; we keep ours explicit
        emb = TextEmbedding(model_name=model)
        vecs_list = list(emb.embed(texts))
        vecs = np.asarray(vecs_list, dtype=np.float32)
        if normalize:
            vecs = _l2_normalize(vecs)
        return vecs, EmbedderInfo(backend="fastembed", model=model, dim=int(vecs.shape[1]))

    if backend in ("sentence-transformers", "st", "torch"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers backend selected but not installed. Install with: pip install -r scripts/requirements-sentence-transformers.txt"
            ) from e

        model_obj = SentenceTransformer(model)
        vecs = model_obj.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        return vecs, EmbedderInfo(backend="sentence-transformers", model=model, dim=int(vecs.shape[1]))

    raise ValueError(
        f"Unknown backend: {backend}. Use 'fastembed' or 'sentence-transformers'."
    )

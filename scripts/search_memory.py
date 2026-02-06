#!/usr/bin/env python
"""Offline semantic search over OpenClaw memory.

Requires an index built by index_memory.py.

Usage:
  python search_memory.py --query "o365 timezone" --top 5

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from embed_backends import embed_texts


console = Console()


def default_index_dir() -> Path:
    base = Path.home() / ".openclaw" / "credentials" / "local-memory-search"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_index(index_dir: Path):
    import faiss

    idx_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"
    if not idx_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Index not found in {index_dir}. Run index_memory.py first.")

    index = faiss.read_index(str(idx_path))
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    chunks = payload.get("chunks", [])
    return index, meta, chunks


def embed_query(query: str, *, backend: str, model_name: str) -> np.ndarray:
    v, _info = embed_texts([query], backend=backend, model=model_name, batch_size=1, normalize=True, show_progress=False)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--index-dir", default=str(default_index_dir()))
    ap.add_argument(
        "--backend",
        default=None,
        choices=["lsa", "sentence-transformers"],
        help="Override backend (default: index meta backend)",
    )
    ap.add_argument("--model", default=None, help="Override model used for query embedding (default: index meta model)")
    ap.add_argument("--json", action="store_true", help="Emit JSON results")

    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    index, meta, chunks = load_index(index_dir)

    backend = args.backend or meta.get("backend") or "lsa"
    model_name = args.model or meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2"

    if backend == "lsa":
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vec_path = index_dir / "lsa-vectorizer.joblib"
        svd_path = index_dir / "lsa-svd.joblib"
        if not vec_path.exists() or not svd_path.exists():
            raise FileNotFoundError("LSA artifacts not found. Rebuild index with --backend lsa")
        vectorizer: TfidfVectorizer = joblib.load(vec_path)
        svd: TruncatedSVD = joblib.load(svd_path)
        Xq = vectorizer.transform([args.query])
        Vq = svd.transform(Xq).astype(np.float32)
        Vq /= (np.linalg.norm(Vq, axis=1, keepdims=True) + 1e-12)
        qv = Vq
    else:
        qv = embed_query(args.query, backend=backend, model_name=model_name)

    scores, ids = index.search(qv, args.top)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict[str, Any]] = []
    for score, i in zip(scores, ids):
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        results.append(
            {
                "score": float(score),
                "path": c.get("doc_path"),
                "chunk_index": c.get("chunk_index"),
                "start_line": c.get("start_line"),
                "end_line": c.get("end_line"),
                "text": c.get("text"),
            }
        )

    if args.json:
        console.print_json(json.dumps({"query": args.query, "results": results}, ensure_ascii=False))
        return 0

    table = Table(title=f"Local Memory Search: {args.query}")
    table.add_column("Score", justify="right")
    table.add_column("File")
    table.add_column("Chunk", justify="right")
    table.add_column("Lines", justify="right")
    table.add_column("Snippet")

    for r in results:
        snippet = (r["text"] or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "…"
        lines = "?"
        if r.get("start_line") and r.get("end_line"):
            lines = f"{r['start_line']}-{r['end_line']}"
        table.add_row(f"{r['score']:.3f}", str(r["path"]), str(r["chunk_index"]), lines, snippet)

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

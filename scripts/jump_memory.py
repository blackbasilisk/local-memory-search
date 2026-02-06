#!/usr/bin/env python
"""Search memory index, then open the source file and print exact lines.

This is the "default behavior" workflow:
  1) semantic jump to best chunk
  2) open the file
  3) quote exact lines for correctness

Usage:
  python jump_memory.py --query "teams link" --top 1 --context 2

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rich.console import Console
# (no Panel output; keep output minimal)


console = Console()


def default_index_dir() -> Path:
    base = Path.home() / ".openclaw" / "credentials" / "local-memory-search"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_payload(index_dir: Path):
    import json

    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.json in {index_dir}. Run index_memory.py first.")
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return payload.get("meta", {}), payload.get("chunks", [])


def search(index_dir: Path, query: str, top: int):
    # reuse search_memory's logic in a minimal, dependency-safe way
    import faiss
    import json
    import numpy as np

    idx_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"
    if not idx_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Index not found in {index_dir}. Run index_memory.py first.")

    index = faiss.read_index(str(idx_path))
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    chunks = payload.get("chunks", [])

    backend = meta.get("backend") or "lsa"

    if backend == "lsa":
        import joblib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec_path = index_dir / "lsa-vectorizer.joblib"
        svd_path = index_dir / "lsa-svd.joblib"
        vectorizer: TfidfVectorizer = joblib.load(vec_path)
        svd: TruncatedSVD = joblib.load(svd_path)
        Xq = vectorizer.transform([query])
        Vq = svd.transform(Xq).astype("float32")
        Vq /= (np.linalg.norm(Vq, axis=1, keepdims=True) + 1e-12)
        qv = Vq
    else:
        from embed_backends import embed_texts

        model = meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
        qv, _info = embed_texts([query], backend=backend, model=model, batch_size=1, normalize=True, show_progress=False)

    scores, ids = index.search(qv, top)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results = []
    for score, i in zip(scores, ids):
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        results.append({"score": float(score), **c})

    return meta, results


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--context", type=int, default=0, help="Extra lines above/below")
    ap.add_argument(
        "--show-source",
        action="store_true",
        help="Include file + line range header (off by default for minimal output)",
    )
    ap.add_argument(
        "--show-line-numbers",
        action="store_true",
        help="Include line numbers (implies --show-source)",
    )
    ap.add_argument("--index-dir", default=str(default_index_dir()))
    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    meta, results = search(index_dir, args.query, args.top)

    if not results:
        console.print("[yellow]No matches.[/yellow]")
        return 2

    for r in results:
        src_path = r.get("doc_abspath")
        if not src_path:
            # backward compat: try workspace + doc_path
            workspace = Path(meta.get("workspace") or "").expanduser()
            rel = r.get("doc_path")
            if not rel or not str(workspace).strip():
                continue
            src = (workspace / rel).resolve()
        else:
            src = Path(str(src_path)).expanduser().resolve()
        start = int(r.get("start_line") or 1)
        end = int(r.get("end_line") or start)
        ctx = max(0, int(args.context))

        lines = read_lines(src)
        lo = max(1, start - ctx)
        hi = min(len(lines), end + ctx)

        # Minimal output by default: just the matched lines (optionally with context)
        show_source = bool(args.show_source or args.show_line_numbers)

        selected = lines[lo - 1 : hi]

        if show_source:
            rel = r.get("doc_path") or str(src)
            header = f"{rel}:{start}-{end}"
            console.print(f"[{header}]")

        if args.show_line_numbers:
            for ln in range(lo, hi + 1):
                console.print(f"{ln:4d}: {lines[ln-1]}")
        else:
            # Print plain text block (no fluff)
            for s in selected:
                console.print(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

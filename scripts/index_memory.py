#!/usr/bin/env python
"""Build an offline semantic index for OpenClaw memory files.

- Reads MEMORY.md and memory/*.md from the given workspace.
- Chunks text, embeds with sentence-transformers, stores a FAISS index + metadata.

Index location (by default): ~/.openclaw/credentials/local-memory-search/

Usage:
  python index_memory.py --workspace C:\\Users\\dave\\.openclaw\\workspace

"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from rich.console import Console

from embed_backends import embed_texts


console = Console()


def default_index_dir() -> Path:
    base = Path.home() / ".openclaw" / "credentials" / "local-memory-search"
    base.mkdir(parents=True, exist_ok=True)
    return base


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Chunk:
    doc_path: str
    doc_abspath: str
    doc_sha256: str
    chunk_index: int
    start_line: int
    end_line: int
    text: str


def iter_memory_files(workspace: Path) -> List[Path]:
    """Default OpenClaw memory layout: MEMORY.md + memory/*.md"""
    files: List[Path] = []
    mem = workspace / "MEMORY.md"
    if mem.exists():
        files.append(mem)
    daily = workspace / "memory"
    if daily.exists():
        for p in sorted(daily.glob("*.md")):
            files.append(p)
    return files


def resolve_files_from_args(*, workspace: Path | None, paths: list[str], globs: list[str], root: Path | None) -> tuple[list[Path], list[Path]]:
    """Return (roots, files) based on CLI args.

    - If --paths/--glob are provided, index those (relative to --root if set).
    - Else, fall back to iter_memory_files(workspace).

    roots are used only for pretty doc_path rendering.
    """
    roots: list[Path] = []
    files: list[Path] = []

    if (paths and len(paths) > 0) or (globs and len(globs) > 0):
        base = (root or workspace or Path.cwd()).expanduser().resolve()
        roots = [base]

        for p in paths:
            pp = (Path(p).expanduser())
            if not pp.is_absolute():
                pp = (base / pp)
            pp = pp.resolve()
            if pp.is_dir():
                # directory means: include markdown-ish files recursively
                for m in sorted(pp.rglob("*.md")):
                    files.append(m)
            else:
                files.append(pp)

        for g in globs:
            gg = g
            if not Path(g).is_absolute():
                gg = str((base / g))
            for hit in sorted(glob.glob(gg, recursive=True)):
                hp = Path(hit).expanduser().resolve()
                if hp.is_file():
                    files.append(hp)

        # de-dupe, keep order
        seen = set()
        uniq: list[Path] = []
        for f in files:
            if str(f) in seen:
                continue
            seen.add(str(f))
            if f.exists() and f.is_file():
                uniq.append(f)
        return roots, uniq

    if workspace is None:
        raise ValueError("--workspace is required unless --paths/--glob are provided")

    ws = workspace.expanduser().resolve()
    roots = [ws]
    return roots, iter_memory_files(ws)


def render_doc_path(p: Path, roots: list[Path]) -> str:
    for r in roots:
        try:
            return str(p.relative_to(r)).replace("\\", "/")
        except Exception:
            continue
    return str(p).replace("\\", "/")


def _pick_overlap_lines(prev_lines: List[str], overlap_chars: int) -> int:
    """Return how many trailing lines to carry forward for overlap."""
    if overlap_chars <= 0:
        return 0
    total = 0
    count = 0
    for line in reversed(prev_lines):
        total += len(line) + 1
        count += 1
        if total >= overlap_chars:
            break
    return min(count, len(prev_lines))


def chunk_text_with_lines(text: str, chunk_chars: int, overlap_chars: int) -> List[tuple[str, int, int]]:
    """Chunk markdown-ish text into (chunk_text, start_line, end_line), 1-indexed.

    Strategy:
    - First group lines into "blocks" (headings, bullet items + their wrapped continuation lines, paragraphs).
    - Then pack blocks into chunks up to chunk_chars.

    This yields smaller, more quotable chunks than raw line-window chunking.
    """
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")

    # 1) Build blocks: (start_line, end_line, block_text)
    blocks: List[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        ln = i + 1
        line = lines[i]
        stripped = line.lstrip()

        if stripped == "":
            i += 1
            continue

        # Heading block
        if stripped.startswith("#"):
            blocks.append((ln, ln, line))
            i += 1
            continue

        # Bullet/list item block (include wrapped continuation lines until next bullet/heading/blank)
        if stripped.startswith("- ") or stripped.startswith("* "):
            start = ln
            buf = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nxts = nxt.lstrip()
                if nxts == "":
                    break
                if nxts.startswith("#") or nxts.startswith("- ") or nxts.startswith("* "):
                    break
                buf.append(nxt)
                i += 1
            end = start + len(buf) - 1
            blocks.append((start, end, "\n".join(buf)))
            continue

        # Paragraph-ish block: consume until blank line
        start = ln
        buf = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() == "":
                break
            buf.append(nxt)
            i += 1
        end = start + len(buf) - 1
        blocks.append((start, end, "\n".join(buf)))

    # 2) Pack blocks into chunks
    chunks: List[tuple[str, int, int]] = []
    cur_texts: List[str] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def flush():
        nonlocal cur_texts, cur_start, cur_end
        if not cur_texts or cur_start is None or cur_end is None:
            cur_texts = []
            cur_start = None
            cur_end = None
            return
        t = "\n\n".join(cur_texts).strip("\n")
        if t.strip():
            chunks.append((t, int(cur_start), int(cur_end)))
        cur_texts = []
        cur_start = None
        cur_end = None

    for (b_start, b_end, b_text) in blocks:
        if cur_start is None:
            cur_start = b_start
            cur_end = b_end
            cur_texts = [b_text]
            continue

        projected = (len("\n\n".join(cur_texts)) + 2 + len(b_text))
        if projected > chunk_chars:
            # flush current chunk
            prev_lines = "\n".join(cur_texts).splitlines()
            flush()

            # overlap by trailing lines from previous chunk
            carry = _pick_overlap_lines(prev_lines, overlap_chars)
            if carry > 0:
                tail = prev_lines[-carry:]
                cur_texts = ["\n".join(tail), b_text]
                # best-effort line range: overlap start unknown precisely; approximate to keep provenance useful
                cur_start = max(1, b_start - carry)
                cur_end = b_end
            else:
                cur_texts = [b_text]
                cur_start = b_start
                cur_end = b_end
        else:
            cur_texts.append(b_text)
            cur_end = b_end

    flush()
    return chunks


def build_chunks(files: list[Path], roots: list[Path], chunk_chars: int, overlap_chars: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in files:
        raw = doc.read_bytes()
        doc_sha = sha256_bytes(raw)
        text = raw.decode("utf-8", errors="replace")
        for idx, (c, start_line, end_line) in enumerate(
            chunk_text_with_lines(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        ):
            c2 = normalize_ws(c)
            if not c2:
                continue
            chunks.append(
                Chunk(
                    doc_path=render_doc_path(doc, roots),
                    doc_abspath=str(doc.resolve()),
                    doc_sha256=doc_sha,
                    chunk_index=idx,
                    start_line=int(start_line),
                    end_line=int(end_line),
                    text=c2,
                )
            )
    return chunks


# embedding implemented in scripts/embed_backends.py


def write_index(index_dir: Path, vectors: np.ndarray, chunks: List[Chunk], meta: dict) -> None:
    import faiss

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized embeddings
    index.add(vectors)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Store chunk metadata (aligned to vector rows)
    payload = {
        "meta": meta,
        "chunks": [c.__dict__ for c in chunks],
    }
    (index_dir / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workspace",
        required=False,
        help="Path to OpenClaw workspace (contains MEMORY.md + memory/). Optional if --paths/--glob are provided.",
    )
    ap.add_argument(
        "--root",
        required=False,
        help="Base folder for relative --paths/--glob (defaults to --workspace, else cwd).",
    )
    ap.add_argument(
        "--paths",
        action="append",
        default=[],
        help="File or folder to index (repeatable). Folders are scanned recursively for *.md.",
    )
    ap.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob to index (repeatable). Example: --glob 'imports/openai-export-derived/**/*.md'",
    )
    ap.add_argument("--index-dir", default=str(default_index_dir()), help="Where to store the FAISS index")
    ap.add_argument(
        "--backend",
        default="fastembed",
        choices=["lsa", "sentence-transformers"],
        help="Backend. lsa=lightweight TF-IDF+SVD (no neural model). sentence-transformers=neural embeddings (torch).",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for sentence-transformers backend (ignored for lsa).",
    )
    ap.add_argument("--chunk-chars", type=int, default=120, help="Chunk size in characters (smaller = more precise quotes)")
    ap.add_argument("--overlap-chars", type=int, default=0, help="Overlap between chunks; default 0 for clean quoting")
    ap.add_argument("--batch-size", type=int, default=64)

    args = ap.parse_args()

    ws = Path(args.workspace).expanduser().resolve() if args.workspace else None
    root = Path(args.root).expanduser().resolve() if args.root else None

    idx = Path(args.index_dir).expanduser().resolve()
    idx.mkdir(parents=True, exist_ok=True)

    roots, files = resolve_files_from_args(workspace=ws, paths=args.paths, globs=args.glob, root=root)
    if not files:
        console.print("[yellow]No files found to index.[/yellow]")
        return 2

    console.print("Indexing files:")
    for f in files:
        console.print(f"- {f}")

    chunks = build_chunks(files, roots, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars)
    if not chunks:
        console.print("[yellow]No chunks produced.[/yellow]")
        return 2

    if args.backend == "lsa":
        console.print("Backend: [bold]lsa[/bold] (TF-IDF + SVD)")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        texts = [c.text for c in chunks]
        # Fit TF-IDF then reduce dimensionality with SVD to get dense vectors
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(256, max(2, X.shape[1] - 1)), random_state=42)
        V = svd.fit_transform(X).astype(np.float32)
        # L2 normalize so we can use inner product for cosine-ish similarity
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

        # Persist artifacts alongside index
        joblib.dump(vectorizer, idx / "lsa-vectorizer.joblib")
        joblib.dump(svd, idx / "lsa-svd.joblib")

        vectors = V
        info = type("Info", (), {"backend": "lsa", "model": "tfidf+svd", "dim": int(vectors.shape[1])})()
    else:
        console.print(f"Backend: [bold]{args.backend}[/bold]")
        console.print(f"Model: [bold]{args.model}[/bold]")
        vectors, info = embed_texts(
            [c.text for c in chunks],
            backend=args.backend,
            model=args.model,
            batch_size=args.batch_size,
            normalize=True,
            show_progress=True,
        )

    meta = {
        "workspace": str(ws) if ws else None,
        "roots": [str(r) for r in roots],
        "backend": info.backend,
        "model": info.model,
        "dim": info.dim,
        "chunk_chars": args.chunk_chars,
        "overlap_chars": args.overlap_chars,
        "count": len(chunks),
        "files_count": len(files),
    }

    write_index(idx, vectors, chunks, meta)

    console.print(f"[green]OK[/green] wrote index to: {idx}")
    console.print(f"Chunks: {len(chunks)} · Dim: {vectors.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

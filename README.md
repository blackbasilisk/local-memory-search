# local-memory-search (OpenClaw skill)

Offline semantic search over OpenClaw memory files:

- `MEMORY.md`
- `memory/*.md`

No online LLM/API required.

## Backends (user choice)

### 1) Light (default): LSA (TF‑IDF + SVD)

- No neural model
- No HuggingFace downloads
- Good “semantic-ish” matching (better than plain keyword search)

### 2) Heavy: sentence-transformers (Torch)

- True neural embeddings
- Larger install (torch) + model download

## Setup

```powershell
cd ~/.openclaw/workspace/skills/local-memory-search
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip

# Light backend
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-lsa.txt
```

## Index

```powershell
# Defaults are tuned for concise, quotable chunks (small chunk size, no overlap)
.\.venv\Scripts\python.exe .\scripts\index_memory.py --workspace "~/.openclaw/workspace" --backend lsa
```

## Default workflow (recommended): jump + quote (minimal output)

```powershell
# Minimal output (default): prints just the best-matching lines
.\.venv\Scripts\python.exe .\scripts\jump_memory.py --query "o365 timezone" --top 1

# Provenance (optional): include file + line numbers
.\.venv\Scripts\python.exe .\scripts\jump_memory.py --query "o365 timezone" --top 1 --show-source --show-line-numbers --context 2
```

## Search only (semantic)

```powershell
.\.venv\Scripts\python.exe .\scripts\search_memory.py --query "o365 timezone" --top 5
```

## Notes

- Index stored at: `~/.openclaw/credentials/local-memory-search/`
- Re-run indexing after editing memory files.

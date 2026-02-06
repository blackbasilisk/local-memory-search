# local-memory-search (OpenClaw skill)

**Search your OpenClaw notes locally** — even when you can’t remember the exact wording.

This skill creates a **local search index** for your OpenClaw memory files and lets an agent quickly jump to the right place in your notes **without sending any text to an online LLM**.

---

## What it does (non-technical)

If you’ve ever thought:

- “We discussed this before… where did we write it down?”
- “I don’t remember the exact phrase, but I remember the *idea*.”

…this skill helps.

It lets you search your memory by *meaning*, not just exact keywords, and then prints the relevant lines from your notes.

**Typical use:**
- ask a question like “where is o365 timezone stored?”
- the agent finds the best match in your local notes
- the agent opens the source file and quotes the exact lines (optional provenance)

---

## Why use this instead of OpenClaw’s built-in `memory_search`?

OpenClaw’s built-in `memory_search` is great, but it usually relies on **cloud embeddings** (e.g., OpenAI/Google). That means:

- it can require **API keys + billing**
- it can fail on quota/keys/network
- it consumes **tokens/spend** over time

**`local-memory-search` is for people who want semantic recall but want to keep it local**:

- **No cloud LLM calls** for search/indexing
- **No embedding token spend**
- Works **offline** (after installing dependencies)

Trade-off: you must **build/refresh the index locally** when your notes change.

---

## How it works (technical overview)

### Inputs
By default, it indexes the standard OpenClaw memory layout:

- `MEMORY.md`
- `memory/*.md`

It can also index **any folder/files** using `--paths` and `--glob` (useful for imported knowledge bases).

### Output style (default)
The recommended workflow is **jump + quote**:

1) semantic jump to best matching chunk
2) open the source file
3) print exact lines (minimal output by default; provenance is optional)

### Backends (choose what you need)
You can choose between two local backends:

- **Light (default): LSA (TF‑IDF + SVD)**
  - No neural model
  - No HuggingFace downloads
  - “Semantic-ish” matching (better than plain grep)
  - Very reliable on Windows

- **Heavy: sentence-transformers (Torch)**
  - True neural embeddings
  - Larger install (torch) + model download
  - Usually better semantic accuracy on messy/long corpora

### Storage
Indexes are written to:

- `~/.openclaw/credentials/local-memory-search/`

Files include:
- `index.faiss` (vector index)
- `chunks.json` (chunk metadata)
- `lsa-*.joblib` (LSA artifacts, when using LSA)

---

## Install

### Requirements
- Python 3.10+ available in PATH

### Light backend (recommended)

```powershell
cd ~/.openclaw/workspace/skills/local-memory-search
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-lsa.txt
```

### Heavy backend (optional)

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-sentence-transformers.txt
```

---

## Build / refresh the index

### Default OpenClaw memory layout

```powershell
# Defaults are tuned for concise, quotable chunks (small chunk size, no overlap)
.\.venv\Scripts\python.exe .\scripts\index_memory.py --workspace "~/.openclaw/workspace" --backend lsa
```

### Index an external folder (Option B)

```powershell
.\.venv\Scripts\python.exe .\scripts\index_memory.py `
  --root "~/.openclaw/workspace" `
  --glob "imports/openai-export-derived/conversations/**/*.md" `
  --glob "imports/openai-export-derived/projects/*.md" `
  --backend lsa `
  --index-dir "~/.openclaw/credentials/local-memory-search-openai-kb"
```

---

## Use

### Default workflow (recommended): jump + quote (minimal output)

```powershell
.\.venv\Scripts\python.exe .\scripts\jump_memory.py --query "o365 timezone" --top 1
```

### Provenance (optional)

```powershell
.\.venv\Scripts\python.exe .\scripts\jump_memory.py --query "o365 timezone" --top 1 --show-source --show-line-numbers --context 2
```

### Search only (semantic)

```powershell
.\.venv\Scripts\python.exe .\scripts\search_memory.py --query "o365 timezone" --top 5
```

---

## Notes / caveats

- If your notes change, **rebuild the index**.
- If you want the best semantic accuracy, use the heavy backend.
- This repository intentionally does **not** include `.venv/` or any index artifacts.

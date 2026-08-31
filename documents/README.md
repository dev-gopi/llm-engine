# Local RAG documents

Place the private or project documents that Gopi should search in this folder,
then rebuild `data/rag/index.sqlite`. Supported formats are TXT, Markdown, HTML,
CSV/TSV, JSON/JSONL, YAML, and PDF.

This file also acts as a small indexing example. Remove it if you do not want
these instructions included in retrieval results.

Index only files placed here:

```bash
.venv/bin/python scripts/build_rag_index.py documents/ --output data/rag/index.sqlite
```

Index this folder together with all project documentation:

```bash
.venv/bin/python scripts/build_rag_index.py README.md docs/ documents/ \
  --output data/rag/index.sqlite
```

"""Build a persistent local-document RAG index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from inference.rag import RagIndex, build_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path, help="files or directories to index")
    parser.add_argument("--output", type=Path, default=Path("data/rag/index.json"))
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()
    chunks = build_chunks(
        args.documents, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars
    )
    destination = RagIndex(chunks).save(args.output)
    print(json.dumps({"index": str(destination), "chunks": len(chunks)}, indent=2))


if __name__ == "__main__":
    main()

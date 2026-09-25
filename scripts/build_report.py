"""Build a unified system report: generation, serving, chat, and RAG.

Examples
--------
# Full report (CPU, all sections, save JSON + Markdown)
.venv/bin/python scripts/build_report.py --output reports/system_report.json

# Generation-only report on GPU
.venv/bin/python scripts/build_report.py --only generation --device cuda

# Serving + RAG only, check live server
.venv/bin/python scripts/build_report.py --only serving --only rag --endpoint http://localhost:8000

# Print Markdown summary to stdout
.venv/bin/python scripts/build_report.py --no-save
"""

from __future__ import annotations

import sys
from pathlib import Path

# Remove scripts/ from sys.path so stdlib modules are not shadowed.
_script_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [e for e in sys.path if str(Path(e or ".").resolve()) != _script_dir]

import argparse
import json
import os
import time
from datetime import datetime, timezone

from inference.reporting import build_system_inference_report


def _section_set(args_only: list[str] | None) -> set[str]:
    """Return the enabled section names, defaulting to all four."""
    all_sections = {"generation", "serving", "chat", "rag"}
    if not args_only:
        return all_sections
    requested = {s.strip().lower() for s in args_only}
    unknown = requested - all_sections
    if unknown:
        print(f"[warn] unknown section(s) ignored: {', '.join(sorted(unknown))}", file=sys.stderr)
    return requested & all_sections


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SECTION",
        help="Enable only specific section(s): generation, serving, chat, rag. "
             "Repeatable. Default: all sections.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.gpu.yaml"),
        help="Model architecture YAML.",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Inference / serving config YAML.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizer-finetuning"),
        help="Tokenizer directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/finetuning/best.pt"),
        help="Fine-tuned model checkpoint for generation probes.",
    )
    parser.add_argument(
        "--sessions-db",
        type=Path,
        default=Path("data/cache/sessions.sqlite"),
        help="SQLite session store for chat activity analysis.",
    )
    parser.add_argument(
        "--rag-db",
        type=Path,
        default=Path("data/rag/index.sqlite"),
        help="SQLite RAG index for retrieval benchmarks.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for generation probes: cpu, cuda, mps. Default: cpu.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Base URL of the running API server to probe for serving status.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/system_report.json"),
        help="Destination path for the JSON report. Parent directory is created.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write files; print Markdown summary to stdout only.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Optional generation probe prompts (space-separated, quoted strings).",
    )
    parser.add_argument(
        "--rag-queries",
        nargs="*",
        help="Optional RAG benchmark queries.",
    )
    args = parser.parse_args()

    sections = _section_set(args.only)

    print("=" * 62, flush=True)
    print("  Gopi Engine — System Inference & Operations Report")
    print("=" * 62, flush=True)
    t0 = time.perf_counter()

    # Build keyword kwargs for reporting
    report_kwargs: dict = dict(
        include_generation="generation" in sections,
        include_serving="serving" in sections,
        include_chat="chat" in sections,
        include_rag="rag" in sections,
        model_config_path=args.model_config,
        inference_config_path=args.inference_config,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        sessions_db_path=args.sessions_db,
        rag_db_path=args.rag_db,
        device=args.device,
        endpoint=args.endpoint,
    )

    for section in ["generation", "serving", "chat", "rag"]:
        if section in sections:
            icon = {"generation": "⚡", "serving": "🌐", "chat": "💬", "rag": "📚"}[section]
            print(f"\n{icon}  Collecting {section} report…", flush=True)

    report = build_system_inference_report(**report_kwargs)

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Report collected in {elapsed:.1f}s\n", flush=True)

    markdown = report.to_markdown()
    json_str = report.to_json()

    print(markdown)

    if not args.no_save:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")

        md_path = out_path.with_suffix(".md")
        md_path.write_text(markdown, encoding="utf-8")

        # Also write standalone offline HTML report with embedded data
        from serving.report_router import DASHBOARD_HTML
        html_content = DASHBOARD_HTML.replace(
            "let currentReportData = null;",
            f"let currentReportData = {json_str};\n    window.addEventListener('DOMContentLoaded', () => {{ renderReport(currentReportData); }});"
        )
        html_path = out_path.with_suffix(".html")
        html_path.write_text(html_content, encoding="utf-8")

        print(f"\n💾 JSON  saved → {out_path}")
        print(f"💾 Report saved → {md_path}")
        print(f"💾 HTML   saved → {html_path}")


if __name__ == "__main__":
    main()

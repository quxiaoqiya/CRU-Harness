"""Offline command line entry point for creating a :class:`DocumentIR`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse a local document into DocumentIR")
    parser.add_argument("source", help="local PDF, HTML, XML, LaTeX, Markdown, TXT, or image")
    parser.add_argument("--source-type", help="override source type inference")
    parser.add_argument("--json-out", type=Path, help="write the IR JSON to this path")
    args = parser.parse_args(argv)
    document = parse(Path(args.source), source_type=args.source_type)
    payload = document.to_json()
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the shell
    raise SystemExit(main())

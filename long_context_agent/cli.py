"""Offline command line entry point for the long-context reader."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from independent_ingestion import parse
from reading_memory import MemoryStateStore
from .reader_agent import ReaderAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a document with the offline long-context Agent")
    parser.add_argument("source", help="local document path")
    parser.add_argument("--deterministic-only", action="store_true", help="disable model-assisted providers")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint directory or JSON file")
    parser.add_argument("--required-field", action="append", default=[], dest="required_fields")
    parser.add_argument("--json-out", type=Path, help="write a compact result JSON")
    args = parser.parse_args(argv)
    # The default reader is deterministic and provider-free.  The flag is
    # accepted explicitly so scripts can document that property.
    _ = args.deterministic_only
    document = parse(Path(args.source))
    checkpoint = args.checkpoint
    root = checkpoint.parent if checkpoint and checkpoint.suffix == ".json" else checkpoint
    store = MemoryStateStore(root or ".memory_checkpoints")
    expected = store.path(document.document_id)
    if checkpoint and checkpoint.suffix == ".json" and checkpoint.exists() and not expected.exists():
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, expected)
    _, memory, state = ReaderAgent(state_store=store).read(
        document, required_fields=args.required_fields
    )
    result = {
        "document_id": document.document_id,
        "status": state.status.value,
        "state": state.to_dict(),
        "memory": memory.to_dict(),
    }
    if checkpoint and checkpoint.suffix == ".json" and expected.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(expected, checkpoint)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the shell
    raise SystemExit(main())

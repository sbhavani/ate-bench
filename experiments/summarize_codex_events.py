#!/usr/bin/env python3
"""Summarize Codex JSONL attempt artifacts.

The Codex event stream is primarily useful here as an effort trace: turn starts,
tool-like events, errors, and final status. This parser keeps the schema
handling intentionally defensive so older snapshots remain readable if Codex
adds or renames event fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def event_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("codex-events.jsonl")))
        elif path.name == "codex-events.jsonl":
            files.append(path)
    return files


def walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def is_tool_like(mapping: dict[str, Any]) -> bool:
    fields = " ".join(
        text_field(mapping.get(key)).lower()
        for key in ("type", "kind", "item_type", "role", "name")
        if key in mapping
    )
    return "tool" in fields or "function" in fields or "exec" in fields or "command" in fields


def summarize(path: Path) -> dict[str, Any]:
    event_types: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    lines = 0
    invalid_lines = 0
    last_error = ""

    with path.open() as f:
        for raw in f:
            lines += 1
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue

            event_type = text_field(event.get("type") or "unknown")
            event_types[event_type] += 1

            if event_type in {"error", "turn.failed"}:
                last_error = text_field(event.get("message") or event.get("error") or event)

            seen_tool_payloads: set[int] = set()
            for mapping in walk_dicts(event):
                if not is_tool_like(mapping):
                    continue
                marker = id(mapping)
                if marker in seen_tool_payloads:
                    continue
                seen_tool_payloads.add(marker)
                name = text_field(
                    mapping.get("name")
                    or mapping.get("tool_name")
                    or mapping.get("kind")
                    or mapping.get("type")
                    or "tool_like"
                )
                tool_names[name] += 1

    status = "unknown"
    if event_types["turn.failed"]:
        status = "failed"
    elif event_types["turn.completed"]:
        status = "completed"
    elif event_types["turn.started"]:
        status = "started"

    metadata = {}
    metadata_path = path.parent / "run-metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}

    return {
        "path": path.as_posix(),
        "status": status,
        "run_label": metadata.get("run_label"),
        "challenge": metadata.get("challenge"),
        "model": metadata.get("model"),
        "lines": lines,
        "invalid_lines": invalid_lines,
        "turns_started": event_types["turn.started"],
        "turns_completed": event_types["turn.completed"],
        "turns_failed": event_types["turn.failed"],
        "errors": event_types["error"],
        "tool_like_events": sum(tool_names.values()),
        "top_tool_like_names": tool_names.most_common(8),
        "event_types": dict(event_types),
        "last_error": last_error,
    }


def print_text(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No codex-events.jsonl files found.")
        return

    headers = [
        "status",
        "label",
        "turns",
        "tool_like",
        "errors",
        "events",
        "path",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row["status"],
                    text_field(row.get("run_label") or "-"),
                    str(row["turns_started"]),
                    str(row["tool_like_events"]),
                    str(row["errors"] + row["turns_failed"]),
                    str(row["lines"]),
                    row["path"],
                ]
            )
        )
        if row["last_error"]:
            print(f"  last_error: {row['last_error'][:400]}")
        if row["top_tool_like_names"]:
            tools = ", ".join(f"{name}={count}" for name, count in row["top_tool_like_names"])
            print(f"  tool_like_names: {tools}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Snapshot dirs or codex-events.jsonl files")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON")
    args = parser.parse_args()

    rows = [summarize(path) for path in event_files(args.paths)]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_text(rows)


if __name__ == "__main__":
    main()

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
import re
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


def numeric_field(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def token_usage_total(token_usage: dict[str, int]) -> int | None:
    for key in ("total_tokens", "tokens_total"):
        if key in token_usage:
            return token_usage[key]
    if "prompt_tokens" in token_usage and "completion_tokens" in token_usage:
        return token_usage["prompt_tokens"] + token_usage["completion_tokens"]
    if "input_tokens" in token_usage and "output_tokens" in token_usage:
        return token_usage["input_tokens"] + token_usage["output_tokens"]
    return None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def phase_elapsed_summary(phase_metrics: dict[str, Any]) -> str:
    if not phase_metrics.get("available"):
        return ""
    phases = phase_metrics.get("phases")
    if not isinstance(phases, dict):
        return ""
    ordered = [
        "inspect",
        "torch-bootstrap",
        "megatron+te-install",
        "other",
    ]
    parts = []
    for phase in ordered:
        metrics = phases.get(phase)
        if not isinstance(metrics, dict):
            continue
        elapsed = metrics.get("elapsed_sec")
        commands = metrics.get("commands")
        elapsed_text = format_seconds(elapsed)
        if commands is None:
            parts.append(f"{phase}={elapsed_text}s")
        else:
            parts.append(f"{phase}={elapsed_text}s/{commands}cmd")
    return ", ".join(parts)


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
    token_usage_max: dict[str, int] = {}
    token_usage_observations = 0

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
                for key, value in mapping.items():
                    token_count = numeric_field(value)
                    normalized_key = snake_case(str(key))
                    if token_count is None or "token" not in normalized_key:
                        continue
                    token_usage_observations += 1
                    token_usage_max[normalized_key] = max(
                        token_count, token_usage_max.get(normalized_key, 0)
                    )
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

    metadata = load_json(path.parent / "run-metadata.json")
    metrics = load_json(path.parent / "run-metrics.json")
    phase_metrics = load_json(path.parent / "install-phase-metrics.json")
    install_smoke = metrics.get("install_smoke") if isinstance(metrics.get("install_smoke"), dict) else {}
    success_artifact = (
        metrics.get("success_artifact") if isinstance(metrics.get("success_artifact"), dict) else {}
    )
    if (
        status == "started"
        and metrics.get("stopped_after_success_artifact")
        and success_artifact.get("ready")
    ):
        status = "completed"

    smoke_elapsed_sec = install_smoke.get("since_agent_start_sec") or install_smoke.get(
        "since_attempt_start_sec"
    )
    success_elapsed_sec = success_artifact.get("since_agent_start_sec") or success_artifact.get(
        "since_attempt_start_sec"
    )

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
        "token_usage_max": token_usage_max,
        "token_usage_observations": token_usage_observations,
        "token_usage_total": token_usage_total(token_usage_max),
        "agent_elapsed_sec": metrics.get("agent_elapsed_sec"),
        "attempt_elapsed_sec": metrics.get("attempt_elapsed_sec"),
        "benchmark_elapsed_sec": metrics.get("benchmark_elapsed_sec"),
        "smoke_elapsed_sec": smoke_elapsed_sec,
        "success_elapsed_sec": success_elapsed_sec,
        "stopped_after_success_artifact": bool(metrics.get("stopped_after_success_artifact")),
        "install_smoke": install_smoke,
        "success_artifact": success_artifact,
        "install_phase_metrics": phase_metrics,
        "phase_elapsed_summary": phase_elapsed_summary(phase_metrics),
    }


def format_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "-"


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
        "agent_s",
        "success_s",
        "tokens",
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
                    format_seconds(row.get("agent_elapsed_sec")),
                    format_seconds(row.get("success_elapsed_sec") or row.get("smoke_elapsed_sec")),
                    text_field(row.get("token_usage_total") or "-"),
                    row["path"],
                ]
            )
        )
        if row.get("stopped_after_success_artifact"):
            print("  stopped_after_success_artifact: true")
        if row["last_error"]:
            print(f"  last_error: {row['last_error'][:400]}")
        if row.get("token_usage_max") and row.get("token_usage_total") is None:
            token_keys = ", ".join(f"{name}={count}" for name, count in row["token_usage_max"].items())
            print(f"  token_usage_max: {token_keys}")
        if row["top_tool_like_names"]:
            tools = ", ".join(f"{name}={count}" for name, count in row["top_tool_like_names"])
            print(f"  tool_like_names: {tools}")
        if row.get("phase_elapsed_summary"):
            print(f"  phase_elapsed_s: {row['phase_elapsed_summary']}")


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

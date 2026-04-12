#!/usr/bin/env python
"""Write a deduplicated result JSONL file.

The input file is left untouched. By default, duplicate
method/dataset/protocol/model/pretrained/shots/seed rows keep the latest
created_at timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.evaluation.results import RunResult, read_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", help="Path to the source result JSONL file.")
    parser.add_argument("output_jsonl", help="Path to write deduplicated rows.")
    parser.add_argument(
        "--keep",
        choices=("latest", "first"),
        default="latest",
        help="Which row to keep when duplicate run keys are found.",
    )
    return parser.parse_args()


def run_key(result: RunResult) -> tuple[str, str, str, str, str, int, int]:
    return (
        result.method,
        result.dataset,
        result.protocol,
        result.model_name,
        result.pretrained,
        result.shots,
        result.seed,
    )


def created_at_value(result: RunResult) -> datetime:
    return datetime.fromisoformat(result.created_at)


def dedupe_results(results: list[RunResult], keep: str) -> list[RunResult]:
    selected: dict[tuple[str, str, str, str, str, int, int], RunResult] = {}
    for result in results:
        key = run_key(result)
        existing = selected.get(key)
        if existing is None:
            selected[key] = result
            continue
        if keep == "latest" and created_at_value(result) >= created_at_value(existing):
            selected[key] = result
        elif keep == "first" and created_at_value(result) < created_at_value(existing):
            selected[key] = result
    return sorted(selected.values(), key=lambda item: (*run_key(item), item.created_at))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    results = read_results(input_path)
    deduped = dedupe_results(results, keep=args.keep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in deduped:
            handle.write(json.dumps(result.to_dict(), sort_keys=True))
            handle.write("\n")

    removed = len(results) - len(deduped)
    print(f"read={len(results)} wrote={len(deduped)} removed_duplicates={removed} output={output_path}")


if __name__ == "__main__":
    main()

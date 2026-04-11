#!/usr/bin/env python
"""Summarize JSONL run results across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.evaluation.results import read_results, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_jsonl", help="Path to a JSONL file containing RunResult records.")
    parser.add_argument("--metric", default="test/top1_accuracy")
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_results(read_results(args.result_jsonl), metric_name=args.metric)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if args.format == "csv":
        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return

    if not rows:
        print(f"No rows found for metric {args.metric!r}.")
        return
    print("method\tdataset\tshots\tmetric\tmean\tstd\tnum_seeds")
    for row in rows:
        print(
            f"{row['method']}\t{row['dataset']}\t{row['shots']}\t{row['metric']}\t"
            f"{row['mean']:.6f}\t{row['std']:.6f}\t{row['num_seeds']}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Inspect raw datasets and print normalized manifest summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.manifest import summarize_records
from common.datasets.paths import resolve_raw_root
from common.datasets.registry import build_manifest
from common.datasets.sources import DATASET_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SOURCES), default=sorted(DATASET_SOURCES))
    parser.add_argument("--data-root", default=None, help="Defaults to ./data or FSL_CLIP_DATA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        raw_root = resolve_raw_root(dataset, data_root=args.data_root)
        records = build_manifest(dataset, raw_root)
        payload = summarize_records(records)
        payload["raw_root"] = str(raw_root)
        print(f"== {dataset} ==")
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

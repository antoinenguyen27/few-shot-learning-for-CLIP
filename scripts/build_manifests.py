#!/usr/bin/env python
"""Build normalized JSONL manifests from raw datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.manifest import summarize_records, write_manifest
from common.datasets.paths import manifest_path, resolve_raw_root
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
        output = write_manifest(records, manifest_path(dataset, data_root=args.data_root))
        summary = summarize_records(records)
        print(
            f"{dataset}: wrote {summary['num_records']} records across "
            f"{summary['num_classes']} classes to {output}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download selected datasets through kagglehub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.download import download_many
from common.datasets.sources import DATASET_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SOURCES), default=sorted(DATASET_SOURCES))
    parser.add_argument("--data-root", default=None, help="Defaults to ./data or FSL_CLIP_DATA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_many(args.datasets, data_root=args.data_root)
    for dataset, path in sorted(paths.items()):
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build deterministic few-shot split JSON files from manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.manifest import read_manifest
from common.datasets.paths import manifest_path, split_path
from common.datasets.sources import DATASET_SOURCES
from common.datasets.splits import make_few_shot_split, write_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SOURCES), default=sorted(DATASET_SOURCES))
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--protocol", default="few_shot_all_classes")
    parser.add_argument("--base-split-seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--data-root", default=None, help="Defaults to ./data or FSL_CLIP_DATA.")
    parser.add_argument("--allow-fewer", action="store_true", help="Allow classes with fewer than k shots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        records = read_manifest(manifest_path(dataset, data_root=args.data_root))
        for shots in args.shots:
            for seed in args.seeds:
                split = make_few_shot_split(
                    records,
                    dataset=dataset,
                    shots=shots,
                    seed=seed,
                    protocol=args.protocol,
                    base_split_seed=args.base_split_seed,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    allow_fewer=args.allow_fewer,
                )
                output = split_path(dataset, args.protocol, shots, seed, data_root=args.data_root)
                write_split(split, output)
                print(
                    f"{dataset} {args.protocol} shots={shots} seed={seed}: "
                    f"train={len(split.train_ids)} val={len(split.val_ids)} test={len(split.test_ids)} -> {output}"
                )


if __name__ == "__main__":
    main()

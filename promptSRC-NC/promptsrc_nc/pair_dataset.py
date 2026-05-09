"""Neighbor-pair dataset for Stage 2 PromptSRC-NC training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence


class NeighborPairDataset:
    def __init__(
        self,
        pairs_path: str | Path,
        items_path: str | Path,
        transform: Callable,
        max_pairs: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for NeighborPairDataset") from exc

        self.torch = torch
        self.pairs_payload = torch.load(pairs_path, map_location="cpu")
        self.pairs = self.pairs_payload["pairs"].long()
        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]
        self.items = _read_jsonl(items_path)
        self.transform = transform
        if len(self.items) == 0:
            raise ValueError(f"No unlabeled items found in {items_path}")
        if self.pairs.numel() == 0:
            raise ValueError(f"No pairs found in {pairs_path}")

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        i, j = self.pairs[index].tolist()
        item_i = self.items[int(i)]
        item_j = self.items[int(j)]
        img_i = Image.open(item_i["impath"]).convert("RGB")
        img_j = Image.open(item_j["impath"]).convert("RGB")
        return {
            "img_i": self.transform(img_i),
            "img_j": self.transform(img_j),
            "uid_i": int(i),
            "uid_j": int(j),
            "impath_i": item_i["impath"],
            "impath_j": item_j["impath"],
        }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def build_pair_loader(
    pair_dir: str | Path,
    pair_mode: str,
    transform: Callable,
    pair_batch_size: int,
    num_workers: int,
    max_pairs: int | None = None,
):
    from torch.utils.data import DataLoader

    pair_dir = Path(pair_dir)
    if pair_mode not in {"real", "shuffled"}:
        raise ValueError("pair_mode must be real or shuffled")
    dataset = NeighborPairDataset(
        pairs_path=pair_dir / f"{pair_mode}_pairs.pt",
        items_path=pair_dir / "unlabeled_items.jsonl",
        transform=transform,
        max_pairs=max_pairs,
    )
    return DataLoader(
        dataset,
        batch_size=pair_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )


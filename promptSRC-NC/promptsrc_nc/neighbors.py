"""Stage 0 frozen-CLIP neighbor construction."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import PromptSRCNCConfig, neighbor_dir
from .data import PromptSRCNCDataset, load_split_records
from .model import build_openclip_bundle, _normalize
from .structured_logging import append_jsonl, write_json


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_frozen_features(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    batch_size: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader

    _train, _val, _test, unlabeled, split, _classnames = load_split_records(
        data_root,
        config.dataset,
        config.shots,
        config.seed,
        config.protocol,
    )
    if config.max_unlabeled_images is not None:
        unlabeled = unlabeled[: config.max_unlabeled_images]
    if any(uid in set(split.test_ids) for uid in split.unlabeled_ids):
        raise RuntimeError("Unsafe split: test IDs appeared in unlabeled pool")

    device = _device()
    bundle = build_openclip_bundle(config.backbone, config.pretrained, device=device, precision="fp32")
    bundle.model.eval()
    loader = DataLoader(
        PromptSRCNCDataset(unlabeled, bundle.eval_preprocess),
        batch_size=batch_size or config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    features = []
    items: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            try:
                feats = bundle.model.encode_image(images, normalize=True)
            except TypeError:
                feats = _normalize(bundle.model.encode_image(images))
            features.append(feats.detach().cpu().float())
            for uid, impath, label, classname in zip(
                batch["uid"],
                batch["impath"],
                batch["label"],
                batch["classname"],
                strict=True,
            ):
                items.append(
                    {
                        "uid": str(uid),
                        "impath": str(impath),
                        "label": int(label),
                        "classname": str(classname),
                    }
                )
    if not features:
        raise ValueError("No unlabeled features were extracted")
    return torch.cat(features, dim=0), items


def mutual_knn_pairs(features: Any, k: int) -> tuple[Any, Any]:
    import torch

    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    n = int(features.shape[0])
    if n < 2:
        raise ValueError("At least two features are required")
    k = min(k, n - 1)
    sims = features @ features.t()
    sims.fill_diagonal_(-float("inf"))
    topk = torch.topk(sims, k=k, dim=1).indices
    neighbor_sets = [set(row.tolist()) for row in topk]
    pairs: list[tuple[int, int]] = []
    cosines: list[float] = []
    for i in range(n):
        for j in neighbor_sets[i]:
            if i < j and i in neighbor_sets[j]:
                pairs.append((i, j))
                cosines.append(float((features[i] * features[j]).sum().item()))
    if not pairs:
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(cosines, dtype=torch.float32)


def degree_preserving_shuffle(
    pairs: Any,
    num_nodes: int,
    seed: int,
    swaps_per_edge: int = 10,
) -> Any:
    import torch

    rng = random.Random(seed)
    edges = {tuple(sorted(map(int, pair))) for pair in pairs.tolist()}
    if len(edges) < 2:
        return pairs.clone()
    edge_list = list(edges)
    num_swaps = swaps_per_edge * len(edge_list)
    for _ in range(num_swaps):
        e1, e2 = rng.sample(edge_list, 2)
        a, b = e1
        c, d = e2
        if rng.random() < 0.5:
            proposed = (tuple(sorted((a, d))), tuple(sorted((c, b))))
        else:
            proposed = (tuple(sorted((a, c))), tuple(sorted((b, d))))
        p1, p2 = proposed
        if p1[0] == p1[1] or p2[0] == p2[1] or p1 == p2:
            continue
        if p1 in edges or p2 in edges:
            continue
        edges.remove(e1)
        edges.remove(e2)
        edges.add(p1)
        edges.add(p2)
        edge_list = list(edges)
    shuffled = sorted(edges)
    return torch.tensor(shuffled, dtype=torch.long)


def build_neighbor_artifacts(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
    log_path: str | Path | None = None,
) -> Path:
    import torch

    output_dir = neighbor_dir(run_root, config.run_id, config.dataset, config.shots, config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, items = extract_frozen_features(config, data_root)
    num_unlabeled = int(features.shape[0])
    requested_pairs, requested_cosines = mutual_knn_pairs(features, config.neighbor_k)
    neighbor_k_used = config.neighbor_k
    real_pairs, real_cosines = requested_pairs, requested_cosines
    if int(real_pairs.shape[0]) < config.min_pairs_fraction * num_unlabeled and config.fallback_k != config.neighbor_k:
        real_pairs, real_cosines = mutual_knn_pairs(features, config.fallback_k)
        neighbor_k_used = config.fallback_k
    if int(real_pairs.shape[0]) == 0:
        raise ValueError("Mutual-neighbor construction produced zero pairs")

    shuffled_pairs = degree_preserving_shuffle(real_pairs, num_unlabeled, seed=config.seed + 10_000)
    shuffled_cosines = torch.tensor(
        [float((features[int(i)] * features[int(j)]).sum().item()) for i, j in shuffled_pairs.tolist()],
        dtype=torch.float32,
    )
    degree_counts = Counter()
    for i, j in real_pairs.tolist():
        degree_counts[int(i)] += 1
        degree_counts[int(j)] += 1

    with (output_dir / "unlabeled_items.jsonl").open("w", encoding="utf-8") as handle:
        for index, item in enumerate(items):
            handle.write(json.dumps({"index": index, **item}, sort_keys=True))
            handle.write("\n")
    torch.save(
        {
            "features": features,
            "uids": [item["uid"] for item in items],
            "impaths": [item["impath"] for item in items],
            "backbone": config.backbone,
            "pretrained": config.pretrained,
        },
        output_dir / "features.pt",
    )
    torch.save(
        {
            "pairs": real_pairs,
            "cosine": real_cosines,
            "neighbor_k": neighbor_k_used,
            "mutual": True,
        },
        output_dir / "real_pairs.pt",
    )
    torch.save(
        {
            "pairs": shuffled_pairs,
            "cosine": shuffled_cosines,
            "source": "degree_preserving_edge_swap",
            "num_swaps": 10 * int(real_pairs.shape[0]),
        },
        output_dir / "shuffled_pairs.pt",
    )
    metadata = {
        "dataset": config.dataset,
        "shots": config.shots,
        "seed": config.seed,
        "unlabeled_policy": "train_remain",
        "clip_backbone": config.backbone,
        "pretrained": config.pretrained,
        "feature_source": "frozen_unprompted_openclip_before_promptsrc",
        "num_unlabeled": num_unlabeled,
        "neighbor_k_requested": config.neighbor_k,
        "neighbor_k_used": neighbor_k_used,
        "fallback_used": neighbor_k_used != config.neighbor_k,
        "num_real_pairs": int(real_pairs.shape[0]),
        "num_shuffled_pairs": int(shuffled_pairs.shape[0]),
        "mean_real_cosine": float(real_cosines.mean().item()),
        "mean_shuffled_cosine": float(shuffled_cosines.mean().item()),
        "degree_min": min(degree_counts.values()) if degree_counts else 0,
        "degree_max": max(degree_counts.values()) if degree_counts else 0,
        "degree_mean_over_connected": float(sum(degree_counts.values()) / max(len(degree_counts), 1)),
        "num_connected_nodes": len(degree_counts),
    }
    write_json(output_dir / "metadata.json", metadata)
    if log_path is not None:
        append_jsonl(log_path, {"event": "neighbors_built", **metadata, "neighbor_dir": str(output_dir)})
    return output_dir


def build_config_from_args(args: Any) -> PromptSRCNCConfig:
    return PromptSRCNCConfig(
        dataset=args.dataset,
        shots=args.shots,
        seed=args.seed,
        backbone=args.backbone,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        neighbor_k=args.neighbor_k,
        fallback_k=args.fallback_k,
        min_pairs_fraction=args.min_pairs_fraction,
        max_unlabeled_images=args.max_unlabeled_images,
        run_id=args.run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build frozen-CLIP PromptSRC-NC neighbor artifacts.")
    parser.add_argument("--data-root", default="/vol/data")
    parser.add_argument("--run-root", default="/vol/runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--neighbor-k", type=int, default=1)
    parser.add_argument("--fallback-k", type=int, default=5)
    parser.add_argument("--min-pairs-fraction", type=float, default=0.25)
    parser.add_argument("--max-unlabeled-images", type=int, default=None)
    parser.add_argument("--log-path", default="")
    args = parser.parse_args(argv)
    out = build_neighbor_artifacts(
        build_config_from_args(args),
        data_root=args.data_root,
        run_root=args.run_root,
        log_path=args.log_path or None,
    )
    print(out)


if __name__ == "__main__":
    main()


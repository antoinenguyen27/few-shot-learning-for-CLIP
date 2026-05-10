"""Stage 0 frozen-CLIP neighbor construction."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import PromptSRCNCConfig, neighbor_dir
from .data import ManifestRecord, PromptSRCNCDataset, load_split_records
from .model import build_openclip_bundle, _normalize
from .provenance import (
    UNLABELED_POLICY_TRAIN_REMAIN,
    ordered_ids_hash,
    split_hash,
    validate_neighbor_metadata,
)
from .structured_logging import append_jsonl, emit_status, write_json


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_frozen_features(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    batch_size: int | None = None,
) -> tuple[Any, list[dict[str, Any]], Any, str]:
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
    emit_status(
        "neighbors_model_load_start",
        run_id=config.run_id,
        stage="neighbors",
        dataset=config.dataset,
        shots=config.shots,
        seed=config.seed,
        backbone=config.backbone,
        num_unlabeled=len(unlabeled),
        device=device,
    )
    bundle = build_openclip_bundle(config.backbone, config.pretrained, device=device, precision="fp32")
    emit_status(
        "neighbors_model_ready",
        run_id=config.run_id,
        stage="neighbors",
        dataset=config.dataset,
        shots=config.shots,
        seed=config.seed,
        backbone=config.backbone,
        openclip_model_name=bundle.model_name,
        device=device,
    )
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
    num_batches = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
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
            if batch_index % max(1, config.log_interval) == 0 or batch_index == num_batches:
                emit_status(
                    "neighbors_feature_batch",
                    run_id=config.run_id,
                    stage="neighbors",
                    dataset=config.dataset,
                    shots=config.shots,
                    seed=config.seed,
                    batch=batch_index,
                    batches=num_batches,
                    encoded_items=len(items),
                    num_unlabeled=len(unlabeled),
                )
    if not features:
        raise ValueError("No unlabeled features were extracted")
    return torch.cat(features, dim=0), items, split, bundle.model_name


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
    return_audit: bool = False,
) -> Any:
    import torch

    rng = random.Random(seed)
    edges = {tuple(sorted(map(int, pair))) for pair in pairs.tolist()}
    original_edges = set(edges)
    if len(edges) < 2:
        audit = {
            "attempted_swaps": 0,
            "accepted_swaps": 0,
            "edge_overlap_fraction": 1.0 if len(edges) else 0.0,
            "degree_preserved": True,
            "weak_shuffled_control": True,
        }
        shuffled = pairs.clone()
        return (shuffled, audit) if return_audit else shuffled
    edge_list = list(edges)
    num_swaps = swaps_per_edge * len(edge_list)
    accepted_swaps = 0
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
        accepted_swaps += 1
    shuffled = sorted(edges)
    shuffled_tensor = torch.tensor(shuffled, dtype=torch.long)
    overlap = len(original_edges & edges) / max(len(original_edges), 1)
    audit = {
        "attempted_swaps": num_swaps,
        "accepted_swaps": accepted_swaps,
        "edge_overlap_fraction": float(overlap),
        "degree_preserved": _degree_sequence(original_edges) == _degree_sequence(edges),
        "weak_shuffled_control": bool(len(original_edges) < 4 or overlap > 0.5),
    }
    return (shuffled_tensor, audit) if return_audit else shuffled_tensor


def _degree_sequence(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    counts = Counter()
    for i, j in edges:
        counts[int(i)] += 1
        counts[int(j)] += 1
    return sorted(counts.items())


def _normalized_path_string(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _validate_pair_tensor(pairs: Any, *, name: str, expected_num_unlabeled: int) -> None:
    if pairs.ndim != 2 or int(pairs.shape[1]) != 2:
        raise ValueError(f"Neighbor artifact {name} pairs must have shape [N, 2]")
    if pairs.numel() == 0:
        return
    if int(pairs.min()) < 0 or int(pairs.max()) >= expected_num_unlabeled:
        raise ValueError(f"Neighbor artifact {name} contains pair indices outside unlabeled_items")
    if bool((pairs[:, 0] == pairs[:, 1]).any().item()):
        raise ValueError(f"Neighbor artifact {name} pairs contain self-loops")
    if bool((pairs[:, 0] >= pairs[:, 1]).any().item()):
        raise ValueError(f"Neighbor artifact {name} pairs must be canonical undirected pairs with i < j")
    canonical = [tuple(map(int, pair)) for pair in pairs.tolist()]
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Neighbor artifact {name} pairs contain duplicate undirected edges")


def validate_neighbor_artifacts(
    pair_dir: str | Path,
    config: PromptSRCNCConfig,
    split: Any,
    *,
    unlabeled_records: Sequence[ManifestRecord] | None = None,
) -> dict[str, Any]:
    import torch

    pair_dir = Path(pair_dir)
    required = ("metadata.json", "unlabeled_items.jsonl", "features.pt", "real_pairs.pt", "shuffled_pairs.pt")
    missing = [name for name in required if not (pair_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Neighbor artifact directory {pair_dir} is missing: {missing}")
    with (pair_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_neighbor_metadata(metadata, config, split)
    if metadata.get("feature_source") != "frozen_unprompted_openclip_before_promptsrc":
        raise ValueError(f"Neighbor artifact {pair_dir} has invalid feature_source={metadata.get('feature_source')!r}")
    if int(metadata.get("neighbor_k_requested", -1)) != int(config.neighbor_k):
        raise ValueError(f"Neighbor artifact {pair_dir} neighbor_k_requested does not match config")
    neighbor_k_used = int(metadata.get("neighbor_k_used", -1))
    if neighbor_k_used not in {int(config.neighbor_k), int(config.fallback_k)}:
        raise ValueError(f"Neighbor artifact {pair_dir} neighbor_k_used must be neighbor_k or fallback_k")
    if bool(metadata.get("fallback_used")) != (neighbor_k_used != int(config.neighbor_k)):
        raise ValueError(f"Neighbor artifact {pair_dir} fallback_used does not match neighbor_k_used")
    items = []
    with (pair_dir / "unlabeled_items.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("index", -1)) != len(items):
                raise ValueError(f"Neighbor artifact {pair_dir} has non-contiguous unlabeled index at line {line_number}")
            items.append(item)
    expected_num_unlabeled = int(metadata["num_unlabeled"])
    if len(items) != expected_num_unlabeled:
        raise ValueError(f"Neighbor artifact {pair_dir} has {len(items)} items; expected {expected_num_unlabeled}")
    item_uids = [str(item.get("uid")) for item in items]
    item_paths = [_normalized_path_string(str(item.get("impath"))) for item in items]
    if len(set(item_uids)) != len(item_uids):
        raise ValueError(f"Neighbor artifact {pair_dir} contains duplicate unlabeled item IDs")
    if ordered_ids_hash(item_uids) != metadata["unlabeled_ids_hash"]:
        raise ValueError(f"Neighbor artifact {pair_dir} item IDs do not match metadata unlabeled_ids_hash")
    if metadata.get("unlabeled_paths_hash") is not None and ordered_ids_hash(item_paths) != metadata["unlabeled_paths_hash"]:
        raise ValueError(f"Neighbor artifact {pair_dir} item paths do not match metadata unlabeled_paths_hash")
    if unlabeled_records is not None:
        records_by_uid = {record.uid: record for record in unlabeled_records}
        for uid, artifact_path in zip(item_uids, item_paths, strict=True):
            record = records_by_uid.get(uid)
            if record is None:
                raise ValueError(f"Neighbor artifact {pair_dir} UID {uid} is missing from active unlabeled records")
            manifest_path = _normalized_path_string(record.image_path)
            if artifact_path != manifest_path:
                raise ValueError(
                    f"Neighbor artifact {pair_dir} image_path for UID {uid} is {artifact_path!r}; "
                    f"expected active manifest path {manifest_path!r}"
                )
    features_payload = torch.load(pair_dir / "features.pt", map_location="cpu")
    features = features_payload.get("features")
    if features is None or features.ndim != 2:
        raise ValueError(f"Neighbor artifact {pair_dir / 'features.pt'} must contain features with shape [N, D]")
    if int(features.shape[0]) != expected_num_unlabeled:
        raise ValueError(f"Neighbor artifact {pair_dir / 'features.pt'} feature count does not match metadata")
    feature_uids = [str(uid) for uid in features_payload.get("uids", [])]
    feature_paths = [_normalized_path_string(str(path)) for path in features_payload.get("impaths", [])]
    if feature_uids != item_uids:
        raise ValueError(f"Neighbor artifact {pair_dir / 'features.pt'} UIDs do not match unlabeled_items")
    if feature_paths != item_paths:
        raise ValueError(f"Neighbor artifact {pair_dir / 'features.pt'} image paths do not match unlabeled_items")
    if features_payload.get("feature_source") not in {None, "frozen_unprompted_openclip_before_promptsrc"}:
        raise ValueError(f"Neighbor artifact {pair_dir / 'features.pt'} has invalid feature_source")
    if features_payload.get("openclip_model_name") != metadata.get("openclip_model_name"):
        raise ValueError(f"Neighbor artifact {pair_dir / 'features.pt'} OpenCLIP model name does not match metadata")
    pair_payloads = {
        "real_pairs.pt": torch.load(pair_dir / "real_pairs.pt", map_location="cpu"),
        "shuffled_pairs.pt": torch.load(pair_dir / "shuffled_pairs.pt", map_location="cpu"),
    }
    pair_tensors = {}
    for name, payload in pair_payloads.items():
        pairs = payload.get("pairs")
        if pairs is None:
            raise ValueError(f"Neighbor artifact {pair_dir / name} is missing pairs")
        pairs = pairs.long()
        _validate_pair_tensor(pairs, name=str(pair_dir / name), expected_num_unlabeled=expected_num_unlabeled)
        pair_tensors[name] = pairs
    real_payload = pair_payloads["real_pairs.pt"]
    if real_payload.get("mutual") is not True:
        raise ValueError(f"Neighbor artifact {pair_dir / 'real_pairs.pt'} must record mutual=True")
    if int(real_payload.get("neighbor_k", -1)) != neighbor_k_used:
        raise ValueError(f"Neighbor artifact {pair_dir / 'real_pairs.pt'} neighbor_k does not match metadata")
    shuffled_payload = pair_payloads["shuffled_pairs.pt"]
    if shuffled_payload.get("source") != "degree_preserving_edge_swap":
        raise ValueError(f"Neighbor artifact {pair_dir / 'shuffled_pairs.pt'} must record degree-preserving source")
    if int(pair_tensors["real_pairs.pt"].shape[0]) != int(metadata["num_real_pairs"]):
        raise ValueError(f"Neighbor artifact {pair_dir} real pair count does not match metadata")
    if int(pair_tensors["shuffled_pairs.pt"].shape[0]) != int(metadata["num_shuffled_pairs"]):
        raise ValueError(f"Neighbor artifact {pair_dir} shuffled pair count does not match metadata")
    if _degree_sequence(tuple(sorted(map(int, pair))) for pair in pair_tensors["real_pairs.pt"].tolist()) != _degree_sequence(
        tuple(sorted(map(int, pair))) for pair in pair_tensors["shuffled_pairs.pt"].tolist()
    ):
        raise ValueError(f"Neighbor artifact {pair_dir} shuffled pairs do not preserve the real-pair degree sequence")
    return metadata


def build_neighbor_artifacts(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
    log_path: str | Path | None = None,
) -> Path:
    import torch

    output_dir = neighbor_dir(run_root, config.run_id, config.dataset, config.shots, config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_status(
        "neighbors_start",
        run_id=config.run_id,
        stage="neighbors",
        dataset=config.dataset,
        shots=config.shots,
        seed=config.seed,
        backbone=config.backbone,
        neighbor_k=config.neighbor_k,
        fallback_k=config.fallback_k,
        max_unlabeled_images=config.max_unlabeled_images,
    )
    features, items, split, openclip_model_name = extract_frozen_features(config, data_root)
    num_unlabeled = int(features.shape[0])
    requested_pairs, requested_cosines = mutual_knn_pairs(features, config.neighbor_k)
    neighbor_k_used = config.neighbor_k
    real_pairs, real_cosines = requested_pairs, requested_cosines
    if int(real_pairs.shape[0]) < config.min_pairs_fraction * num_unlabeled and config.fallback_k != config.neighbor_k:
        real_pairs, real_cosines = mutual_knn_pairs(features, config.fallback_k)
        neighbor_k_used = config.fallback_k
    if int(real_pairs.shape[0]) == 0:
        raise ValueError("Mutual-neighbor construction produced zero pairs")

    shuffled_pairs, shuffle_audit = degree_preserving_shuffle(
        real_pairs,
        num_unlabeled,
        seed=config.seed + 10_000,
        return_audit=True,
    )
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
            "impaths": [_normalized_path_string(item["impath"]) for item in items],
            "backbone": config.backbone,
            "openclip_model_name": openclip_model_name,
            "pretrained": config.pretrained,
            "feature_source": "frozen_unprompted_openclip_before_promptsrc",
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
        "protocol": config.protocol,
        "unlabeled_policy": UNLABELED_POLICY_TRAIN_REMAIN,
        "uses_test_images_for_unlabeled": False,
        "clip_backbone": config.backbone,
        "openclip_model_name": openclip_model_name,
        "pretrained": config.pretrained,
        "feature_source": "frozen_unprompted_openclip_before_promptsrc",
        "num_unlabeled": num_unlabeled,
        "max_unlabeled_images": config.max_unlabeled_images,
        "unlabeled_pool_variant": "capped_train_remain_prefix" if config.max_unlabeled_images else "full_train_remain",
        "unlabeled_ids_hash": ordered_ids_hash(item["uid"] for item in items),
        "unlabeled_paths_hash": ordered_ids_hash(_normalized_path_string(item["impath"]) for item in items),
        "split_hash": split_hash(split),
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
        **{f"shuffle_{key}": value for key, value in shuffle_audit.items()},
    }
    write_json(output_dir / "metadata.json", metadata)
    if log_path is not None:
        append_jsonl(log_path, {"event": "neighbors_built", **metadata, "neighbor_dir": str(output_dir)})
    emit_status("neighbors_built", run_id=config.run_id, **metadata, neighbor_dir=str(output_dir))
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

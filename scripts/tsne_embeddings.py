#!/usr/bin/env python
"""Generate OpenCLIP image-embedding t-SNE plots for local datasets."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.manifest import class_names_from_records, read_manifest
from common.datasets.paths import manifest_path, resolve_raw_root, split_path
from common.datasets.splits import read_split
from common.datasets.torch_dataset import ManifestImageDataset, records_by_ids
from common.datasets.sources import DATASET_SOURCES
from common.models.openclip import build_openclip_bundle, encode_image_features


DATASET_LABELS = {
    "eurosat": "EuroSAT",
    "flowers102": "Flowers102",
    "stanford_cars": "Stanford Cars",
}
DATASET_COLORS = {
    "eurosat": "#3f7f5f",
    "flowers102": "#c64f7a",
    "stanford_cars": "#496fb3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SOURCES), default=["eurosat", "flowers102", "stanford_cars"])
    parser.add_argument("--source", choices=("split-test", "split-val", "manifest"), default="split-test")
    parser.add_argument("--protocol", default="few_shot_all_classes")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-dir", default="data/cache/embeddings")
    parser.add_argument("--out-dir", default="docs/figures/tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Recompute features and t-SNE even when cached arrays exist.")
    return parser.parse_args()


def load_records_for_source(dataset: str, args: argparse.Namespace):
    records = read_manifest(manifest_path(dataset, data_root=args.data_root))
    if args.source == "manifest":
        selected = records
    else:
        split = read_split(split_path(dataset, args.protocol, args.shots, args.seed, data_root=args.data_root))
        ids = split.test_ids if args.source == "split-test" else split.val_ids
        selected = records_by_ids(records, ids)
    classnames = class_names_from_records(records)
    return balanced_sample(selected, args.max_per_class, args.random_state), classnames


def balanced_sample(records: list[Any], max_per_class: int, random_state: int) -> list[Any]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.label_id].append(record)
    rng = random.Random(random_state)
    sampled = []
    for label in sorted(grouped):
        items = sorted(grouped[label], key=lambda record: record.sample_id)
        rng.shuffle(items)
        sampled.extend(sorted(items[:max_per_class], key=lambda record: record.sample_id))
    return sorted(sampled, key=lambda record: (record.label_id, record.sample_id))


def cache_key(args: argparse.Namespace) -> str:
    dataset_part = "-".join(args.datasets)
    return slug(
        f"{dataset_part}_{args.source}_{args.protocol}_shots{args.shots}_seed{args.seed}_"
        f"max{args.max_per_class}_perp{args.perplexity}_rs{args.random_state}"
    )


def extract_or_load_features(args: argparse.Namespace):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"{cache_key(args)}_features.npz"
    metadata_path = cache_dir / f"{cache_key(args)}_metadata.json"

    if feature_path.exists() and metadata_path.exists() and not args.force:
        payload = np.load(feature_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return payload["features"], payload["labels"], payload["dataset_ids"], metadata

    bundle = build_openclip_bundle(device=args.device, precision="fp32")
    all_features = []
    all_labels = []
    all_dataset_ids = []
    metadata: dict[str, Any] = {
        "source": args.source,
        "protocol": args.protocol,
        "shots": args.shots,
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "datasets": {},
        "model_name": bundle.model_name,
        "pretrained": bundle.pretrained,
    }

    for dataset_index, dataset in enumerate(args.datasets):
        records, classnames = load_records_for_source(dataset, args)
        metadata["datasets"][dataset] = {
            "num_samples": len(records),
            "num_classes": len({record.label_id for record in records}),
            "class_names": classnames,
        }
        torch_dataset = ManifestImageDataset(
            records=records,
            raw_root=resolve_raw_root(dataset, data_root=args.data_root),
            transform=bundle.preprocess_eval,
            output_format="dict",
        )
        loader = DataLoader(torch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dataset_features = []
        dataset_labels = []
        for batch in loader:
            images = batch["image"]
            labels = batch["label"]
            features = encode_image_features(bundle, images, normalize=True)
            dataset_features.append(features.detach().cpu())
            dataset_labels.append(labels.detach().cpu())
        features_tensor = torch.cat(dataset_features, dim=0)
        labels_tensor = torch.cat(dataset_labels, dim=0)
        all_features.append(features_tensor.numpy())
        all_labels.append(labels_tensor.numpy())
        all_dataset_ids.append(np.full((features_tensor.shape[0],), dataset_index, dtype=np.int64))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    dataset_ids = np.concatenate(all_dataset_ids, axis=0)
    np.savez_compressed(feature_path, features=features, labels=labels, dataset_ids=dataset_ids)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return features, labels, dataset_ids, metadata


def compute_or_load_tsne(args: argparse.Namespace, features):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cache_dir = Path(args.cache_dir)
    tsne_path = cache_dir / f"{cache_key(args)}_tsne.npz"
    if tsne_path.exists() and not args.force:
        payload = np.load(tsne_path, allow_pickle=False)
        return payload["xy"]

    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for t-SNE.")
    pca_dim = min(50, features.shape[1], n_samples - 1)
    reduced = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(features)
    perplexity = min(args.perplexity, max(2.0, (n_samples - 1) / 3))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": args.random_state,
    }
    import inspect

    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    xy = TSNE(**tsne_kwargs).fit_transform(reduced)
    np.savez_compressed(tsne_path, xy=xy)
    return xy


def write_charts(args: argparse.Namespace, xy, labels, dataset_ids, metadata: dict[str, Any]) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(args.datasets)
    write_scatter_svg(
        out_dir / "combined_dataset_tsne.svg",
        "OpenCLIP t-SNE by Dataset",
        "Balanced sampled image embeddings, colored by dataset",
        xy,
        [datasets[index] for index in dataset_ids],
        {dataset: DATASET_COLORS.get(dataset, palette(index)) for index, dataset in enumerate(datasets)},
        {dataset: DATASET_LABELS.get(dataset, dataset) for dataset in datasets},
    )

    for dataset_index, dataset in enumerate(datasets):
        mask = dataset_ids == dataset_index
        dataset_xy = xy[mask]
        dataset_labels = labels[mask]
        class_names = metadata["datasets"][dataset]["class_names"]
        color_keys = [int(label) for label in dataset_labels]
        colors = {label: palette(label) for label in sorted(set(color_keys))}
        names = {label: class_names[label] if label < len(class_names) else str(label) for label in sorted(set(color_keys))}
        write_scatter_svg(
            out_dir / f"{dataset}_class_tsne.svg",
            f"{DATASET_LABELS.get(dataset, dataset)} t-SNE by Class",
            "OpenCLIP image embeddings, sampled from the selected split source",
            dataset_xy,
            color_keys,
            colors,
            names,
            max_legend_items=16,
        )

    write_report(Path(args.out_dir), args, metadata)


def write_scatter_svg(
    path: Path,
    title: str,
    subtitle: str,
    xy,
    color_keys: list[Any],
    colors: dict[Any, str],
    names: dict[Any, str],
    max_legend_items: int = 32,
) -> None:
    width, height = 900, 640
    margin_left, margin_right, margin_top, margin_bottom = 56, 228, 78, 52
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [float(item[0]) for item in xy]
    ys = [float(item[1]) for item in xy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * 0.06 or 1.0
    pad_y = (max_y - min_y) * 0.06 or 1.0
    min_x, max_x = min_x - pad_x, max_x + pad_x
    min_y, max_y = min_y - pad_y, max_y + pad_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2528}.title{font-size:22px;font-weight:700}.subtitle{font-size:12px;fill:#526066}.small{font-size:11px;fill:#526066}.axis{stroke:#d8dee2;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#fbfcfd"/>',
        f'<text x="32" y="36" class="title">{esc(title)}</text>',
        f'<text x="32" y="56" class="subtitle">{esc(subtitle)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d8dee2" rx="8"/>',
    ]

    for x_value, y_value, key in zip(xs, ys, color_keys):
        x = margin_left + (x_value - min_x) / (max_x - min_x) * plot_w
        y = margin_top + plot_h - (y_value - min_y) / (max_y - min_y) * plot_h
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{colors[key]}" fill-opacity="0.72"/>')

    parts.append(f'<text x="{margin_left}" y="{height - 22}" class="small">t-SNE axes are unitless; distances are qualitative local-neighborhood structure.</text>')

    legend_x, legend_y = width - margin_right + 24, margin_top
    unique_keys = list(dict.fromkeys(color_keys))
    for index, key in enumerate(unique_keys[:max_legend_items]):
        y = legend_y + index * 22
        parts.append(f'<rect x="{legend_x}" y="{y}" width="13" height="13" rx="3" fill="{colors[key]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{y + 11}" class="small">{esc(names.get(key, key))}</text>')
    if len(unique_keys) > max_legend_items:
        parts.append(f'<text x="{legend_x}" y="{legend_y + max_legend_items * 22 + 12}" class="small">+ {len(unique_keys) - max_legend_items} more classes</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")
    print(path)


def write_report(out_dir: Path, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    report = out_dir.parent.parent / "embedding-analysis.md"
    lines = [
        "# Embedding t-SNE Analysis",
        "",
        "Generated from OpenCLIP image embeddings using local manifests and split files.",
        "",
        "## Settings",
        "",
        f"- Source: `{args.source}`",
        f"- Protocol: `{args.protocol}`",
        f"- Shots/seed for split source: `{args.shots}` / `{args.seed}`",
        f"- Max samples per class: `{args.max_per_class}`",
        f"- Model: `{metadata['model_name']}`",
        f"- Pretrained: `{metadata['pretrained']}`",
        "",
        "## Charts",
        "",
        "![Combined dataset t-SNE](figures/tsne/combined_dataset_tsne.svg)",
        "",
    ]
    for dataset in args.datasets:
        label = DATASET_LABELS.get(dataset, dataset)
        lines.extend(
            [
                f"![{label} class t-SNE](figures/tsne/{dataset}_class_tsne.svg)",
                "",
            ]
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def palette(index: int) -> str:
    hue = (index * 137.508) % 360
    return hsl_to_hex(hue, 58, 47)


def hsl_to_hex(h: float, s: float, l: float) -> str:
    s /= 100
    l /= 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return f"#{round((r + m) * 255):02x}{round((g + m) * 255):02x}{round((b + m) * 255):02x}"


def main() -> None:
    args = parse_args()
    features, labels, dataset_ids, metadata = extract_or_load_features(args)
    xy = compute_or_load_tsne(args, features)
    write_charts(args, xy, labels, dataset_ids, metadata)


if __name__ == "__main__":
    main()

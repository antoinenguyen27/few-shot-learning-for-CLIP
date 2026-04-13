#!/usr/bin/env python
"""Generate SVG charts for dataset and split analysis.

This script intentionally uses only the Python standard library so chart
generation does not add a plotting dependency to the research environment.
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.manifest import read_manifest
from common.datasets.splits import read_split


DATASETS = ("eurosat", "flowers102", "stanford_cars")
DATASET_LABELS = {
    "eurosat": "EuroSAT",
    "flowers102": "Flowers102",
    "stanford_cars": "Stanford Cars",
}
SHOTS = (1, 2, 4, 8, 16)
COLORS = {
    "eurosat": "#3f7f5f",
    "flowers102": "#c64f7a",
    "stanford_cars": "#496fb3",
    "train_pool": "#3f7f5f",
    "val": "#e0a43a",
    "test": "#c64f7a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--protocol", default="few_shot_all_classes")
    parser.add_argument("--split-shot", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=1)
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_page(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "<style>\n"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2528} "
        ".title{font-size:22px;font-weight:700} .subtitle{font-size:12px;fill:#526066} "
        ".axis{stroke:#293134;stroke-width:1.2} .grid{stroke:#d8dee2;stroke-width:1} "
        ".label{font-size:12px} .small{font-size:11px;fill:#526066} .value{font-size:12px;font-weight:700}\n"
        "</style>\n"
        f"{body}\n</svg>\n"
    )


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(path)


def dataset_stats(data_root: Path) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        records = read_manifest(data_root / "manifests" / f"{dataset}.jsonl")
        class_counts = Counter(record.label_id for record in records)
        counts = list(class_counts.values())
        stats[dataset] = {
            "records": len(records),
            "classes": len(class_counts),
            "class_min": min(counts),
            "class_median": median(counts),
            "class_mean": mean(counts),
            "class_max": max(counts),
            "imbalance": max(counts) / min(counts),
            "source_splits": Counter(record.source_split for record in records),
        }
    return stats


def split_stats(data_root: Path, protocol: str, split_shot: int, split_seed: int) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        split = read_split(
            data_root
            / "splits"
            / dataset
            / protocol
            / f"shots_{split_shot}"
            / f"seed_{split_seed}.json"
        )
        stats[dataset] = {
            "train": len(split.train_ids),
            "val": len(split.val_ids),
            "test": len(split.test_ids),
            "train_pool": split.metadata.get("train_pool_size", len(split.train_ids)),
            "base_split": split.metadata.get("base_split", ""),
        }
    return stats


def train_sizes_by_shot(data_root: Path, protocol: str) -> dict[str, dict[int, int]]:
    output: dict[str, dict[int, int]] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for shot in SHOTS:
            split = read_split(data_root / "splits" / dataset / protocol / f"shots_{shot}" / "seed_1.json")
            output[dataset][shot] = len(split.train_ids)
    return output


def bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    output: Path,
    y_label: str = "",
) -> None:
    width, height = 920, 520
    margin_left, margin_right, margin_top, margin_bottom = 88, 36, 76, 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max(max(values) for _, values, _ in series)
    y_max = nice_max(max_value)
    group_w = plot_w / len(labels)
    bar_gap = 8
    bar_w = min(42, (group_w - 24 - bar_gap * (len(series) - 1)) / len(series))

    parts = [
        f'<rect width="{width}" height="{height}" fill="#fbfcfd"/>',
        f'<text x="32" y="36" class="title">{esc(title)}</text>',
        f'<text x="32" y="56" class="subtitle">{esc(subtitle)}</text>',
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = margin_top + plot_h - (value / y_max) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{format_number(value)}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis"/>')
    if y_label:
        parts.append(
            f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" '
            f'text-anchor="middle" class="small">{esc(y_label)}</text>'
        )

    for group_index, label in enumerate(labels):
        group_x = margin_left + group_index * group_w
        center = group_x + group_w / 2
        parts.append(f'<text x="{center:.1f}" y="{height - 44}" text-anchor="middle" class="label">{esc(label)}</text>')
        series_w = len(series) * bar_w + (len(series) - 1) * bar_gap
        start_x = center - series_w / 2
        for series_index, (_, values, color) in enumerate(series):
            value = values[group_index]
            bar_h = 0 if y_max == 0 else (value / y_max) * plot_h
            x = start_x + series_index * (bar_w + bar_gap)
            y = margin_top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="value">{format_number(value)}</text>')

    legend_x = width - margin_right - 220
    legend_y = 28
    for index, (name, _, color) in enumerate(series):
        y = legend_y + index * 22
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{y + 11}" class="small">{esc(name)}</text>')

    write_svg(output, svg_page(width, height, "\n".join(parts)))


def pie_chart(title: str, subtitle: str, slices: list[tuple[str, float, str]], output: Path) -> None:
    width, height = 720, 420
    cx, cy, radius = 210, 230, 130
    total = sum(value for _, value, _ in slices)
    parts = [
        f'<rect width="{width}" height="{height}" fill="#fbfcfd"/>',
        f'<text x="32" y="36" class="title">{esc(title)}</text>',
        f'<text x="32" y="56" class="subtitle">{esc(subtitle)}</text>',
    ]
    start = -math.pi / 2
    for name, value, color in slices:
        angle = 0 if total == 0 else (value / total) * 2 * math.pi
        end = start + angle
        parts.append(pie_path(cx, cy, radius, start, end, color))
        start = end

    legend_x, legend_y = 410, 132
    for index, (name, value, color) in enumerate(slices):
        y = legend_y + index * 46
        percent = 0 if total == 0 else value / total * 100
        parts.append(f'<rect x="{legend_x}" y="{y}" width="18" height="18" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{y + 14}" class="label">{esc(name)}</text>')
        parts.append(f'<text x="{legend_x + 28}" y="{y + 32}" class="small">{format_number(value)} images, {percent:.1f}%</text>')

    write_svg(output, svg_page(width, height, "\n".join(parts)))


def pie_path(cx: float, cy: float, radius: float, start: float, end: float, color: str) -> str:
    if end - start >= 2 * math.pi - 1e-9:
        return f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}"/>'
    x1 = cx + radius * math.cos(start)
    y1 = cy + radius * math.sin(start)
    x2 = cx + radius * math.cos(end)
    y2 = cy + radius * math.sin(end)
    large_arc = 1 if end - start > math.pi else 0
    return (
        f'<path d="M {cx:.2f} {cy:.2f} L {x1:.2f} {y1:.2f} '
        f'A {radius:.2f} {radius:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z" fill="{color}"/>'
    )


def nice_max(value: float) -> float:
    if value <= 0:
        return 1
    exponent = math.floor(math.log10(value))
    fraction = value / 10**exponent
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * 10**exponent


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        value = int(round(value))
    if isinstance(value, int):
        return f"{value:,}"
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def write_index(out_dir: Path, stats: dict[str, dict[str, object]], split_summary: dict[str, dict[str, object]]) -> None:
    rows = []
    for dataset in DATASETS:
        item = stats[dataset]
        split = split_summary[dataset]
        rows.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[dataset],
                    format_number(float(item["records"])),
                    format_number(float(item["classes"])),
                    format_number(float(item["class_min"])),
                    format_number(float(item["class_median"])),
                    format_number(float(item["class_max"])),
                    str(split["base_split"]),
                ]
            )
            + " |"
        )
    content = "\n".join(
        [
            "# Dataset Chart Pack",
            "",
            "Generated from local manifests and split JSON files.",
            "",
            "| Dataset | Images | Classes | Min/Class | Median/Class | Max/Class | Split Policy |",
            "|---|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "## Charts",
            "",
            "![Dataset images and classes](figures/dataset_size_classes.svg)",
            "",
            "![Class imbalance](figures/class_count_distribution.svg)",
            "",
            "![Few-shot train sizes](figures/few_shot_train_sizes.svg)",
            "",
            "![EuroSAT split composition](figures/eurosat_split_pie.svg)",
            "",
            "![Flowers102 split composition](figures/flowers102_split_pie.svg)",
            "",
            "![Stanford Cars split composition](figures/stanford_cars_split_pie.svg)",
            "",
        ]
    )
    output = out_dir.parent / "dataset-chart-pack.md"
    output.write_text(content, encoding="utf-8")
    print(output)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    stats = dataset_stats(data_root)
    split_summary = split_stats(data_root, args.protocol, args.split_shot, args.split_seed)
    train_sizes = train_sizes_by_shot(data_root, args.protocol)

    labels = [DATASET_LABELS[item] for item in DATASETS]
    bar_chart(
        "Dataset Scale",
        "Images and classes in the generated local manifests",
        labels,
        [
            ("Images", [float(stats[item]["records"]) for item in DATASETS], "#3f7f5f"),
            ("Classes", [float(stats[item]["classes"]) for item in DATASETS], "#496fb3"),
        ],
        out_dir / "dataset_size_classes.svg",
    )
    bar_chart(
        "Class Count Distribution",
        "Minimum, median, and maximum images per class",
        labels,
        [
            ("Min", [float(stats[item]["class_min"]) for item in DATASETS], "#c64f7a"),
            ("Median", [float(stats[item]["class_median"]) for item in DATASETS], "#e0a43a"),
            ("Max", [float(stats[item]["class_max"]) for item in DATASETS], "#3f7f5f"),
        ],
        out_dir / "class_count_distribution.svg",
        y_label="images per class",
    )
    bar_chart(
        "Few-Shot Train Set Size",
        "Number of supervised train images per dataset and shot count",
        [str(shot) for shot in SHOTS],
        [
            (DATASET_LABELS[dataset], [float(train_sizes[dataset][shot]) for shot in SHOTS], COLORS[dataset])
            for dataset in DATASETS
        ],
        out_dir / "few_shot_train_sizes.svg",
        y_label="train images",
    )

    for dataset in DATASETS:
        split = split_summary[dataset]
        pie_chart(
            f"{DATASET_LABELS[dataset]} Protocol Split",
            f"Train pool, validation, and test composition for {args.protocol}, shot={args.split_shot}, seed={args.split_seed}",
            [
                ("Train pool", float(split["train_pool"]), COLORS["train_pool"]),
                ("Validation", float(split["val"]), COLORS["val"]),
                ("Test", float(split["test"]), COLORS["test"]),
            ],
            out_dir / f"{dataset}_split_pie.svg",
        )

    write_index(out_dir, stats, split_summary)


if __name__ == "__main__":
    main()

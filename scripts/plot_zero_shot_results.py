#!/usr/bin/env python
"""Generate an SVG bar chart for zero-shot CLIP result metrics."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.evaluation.results import read_results, summarize_results


DATASET_LABELS = {
    "eurosat": "EuroSAT",
    "flowers102": "Flowers102",
    "stanford_cars": "Stanford Cars",
}
DATASET_ORDER = ("eurosat", "flowers102", "stanford_cars")
BAR_COLORS = {
    "eurosat": "#3f7f5f",
    "flowers102": "#c64f7a",
    "stanford_cars": "#496fb3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/zero_shot_clip.jsonl")
    parser.add_argument("--output", default="docs/figures/zero_shot_clip_top1_accuracy.svg")
    parser.add_argument("--metric", default="test/top1_accuracy")
    parser.add_argument("--method", default="Zero-shot CLIP")
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def svg_page(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        '<title id="title">Zero-shot CLIP top-1 accuracy</title>\n'
        '<desc id="desc">Bar chart of zero-shot CLIP test top-1 accuracy for EuroSAT, Flowers102, and Stanford Cars.</desc>\n'
        "<style>\n"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2528} "
        ".title{font-size:22px;font-weight:700} .subtitle{font-size:12px;fill:#526066} "
        ".axis{stroke:#293134;stroke-width:1.2} .grid{stroke:#d8dee2;stroke-width:1} "
        ".label{font-size:12px} .small{font-size:11px;fill:#526066} .value{font-size:13px;font-weight:700}\n"
        "</style>\n"
        f"{body}\n</svg>\n"
    )


def ordered_rows(result_path: Path, metric: str, method: str) -> list[dict[str, object]]:
    results = [result for result in read_results(result_path) if result.method == method]
    zero_shot_results = [result for result in results if result.shots == 0]
    if zero_shot_results:
        results = zero_shot_results
    rows = summarize_results(results, metric_name=metric)
    by_dataset = {str(row["dataset"]): row for row in rows}
    missing = [dataset for dataset in DATASET_ORDER if dataset not in by_dataset]
    if missing:
        raise ValueError(f"Missing {metric} rows for: {', '.join(missing)}")
    return [by_dataset[dataset] for dataset in DATASET_ORDER]


def write_chart(rows: list[dict[str, object]], metric: str, output: Path) -> None:
    width, height = 840, 500
    margin_left, margin_right, margin_top, margin_bottom = 86, 40, 82, 82
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_max = 1.0
    bar_w = 78
    group_w = plot_w / len(rows)

    parts = [
        f'<rect width="{width}" height="{height}" fill="#fbfcfd"/>',
        '<text x="32" y="36" class="title">Zero-shot CLIP Top-1 Accuracy</text>',
        f'<text x="32" y="56" class="subtitle">{esc(metric)} from results/zero_shot_clip.jsonl; shots=0</text>',
    ]

    for tick in range(6):
        value = tick / 5
        y = margin_top + plot_h - value * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{pct(value)}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis"/>')
    parts.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" '
        'text-anchor="middle" class="small">Test top-1 accuracy</text>'
    )

    for index, row in enumerate(rows):
        dataset = str(row["dataset"])
        value = float(row["mean"])
        center = margin_left + index * group_w + group_w / 2
        bar_h = value / y_max * plot_h
        x = center - bar_w / 2
        y = margin_top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="6" fill="{BAR_COLORS[dataset]}"/>')
        parts.append(f'<text x="{center:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{pct(value)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height - 46}" text-anchor="middle" class="label">{esc(DATASET_LABELS[dataset])}</text>')

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg_page(width, height, "\n".join(parts)), encoding="utf-8")
    print(output)


def main() -> None:
    args = parse_args()
    rows = ordered_rows(Path(args.results), args.metric, args.method)
    write_chart(rows, args.metric, Path(args.output))


if __name__ == "__main__":
    main()

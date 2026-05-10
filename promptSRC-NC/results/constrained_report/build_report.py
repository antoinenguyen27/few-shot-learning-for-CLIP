from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
RUNS = {
    "EuroSAT": ROOT.parent / "promptsrc-nc-main-20260510-192529" / "results" / "runs.jsonl",
    "Flowers102": ROOT.parent / "promptsrc-nc-main-20260510-192533" / "results" / "runs.jsonl",
    "Stanford Cars": ROOT.parent / "promptsrc-nc-main-20260510-192531" / "results" / "runs.jsonl",
}

ZERO_SHOT = {
    "EuroSAT": 47.5,
    "Flowers102": 71.4,
    "Stanford Cars": 65.3,
}

ZERO_SHOT_SOURCE = (
    "2SFS CVPR 2025 Table 2, ViT-B/16 Zero-Shot row "
    "(all-to-all, k=16 shots per class)"
)
ZERO_SHOT_URL = (
    "https://openaccess.thecvf.com/content/CVPR2025/papers/"
    "Farina_Rethinking_Few-Shot_Adaptation_of_Vision-Language_Models_in_Two_Stages_CVPR_2025_paper.pdf"
)

METHOD_ORDER = [
    "PromptSRC",
    "PromptSRC-NC real",
    "PromptSRC-NC shuffled",
]

METHOD_LABEL = {
    "PromptSRC": "PromptSRC",
    "PromptSRC-NC real": "PromptSRC-NC (real neighbors)",
    "PromptSRC-NC shuffled": "PromptSRC-NC (randomized neighbors)",
}


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def fmt_pct(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "-"
    if std is not None:
        return f"{value * 100:.2f} +/- {std * 100:.2f}"
    return f"{value * 100:.2f}"


def fmt_pp(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "-"
    if std is not None:
        return f"{value * 100:+.2f} +/- {std * 100:.2f}"
    return f"{value * 100:+.2f}"


def read_runs() -> list[dict]:
    rows: list[dict] = []
    for display_dataset, path in RUNS.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["display_dataset"] = display_dataset
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_accuracy(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("test_top1") is None:
            continue
        grouped[(row["display_dataset"], row["method"])].append(row)

    out: list[dict] = []
    for dataset in ["EuroSAT", "Flowers102", "Stanford Cars"]:
        out.append(
            {
                "dataset": dataset,
                "method": "Zero-shot CLIP",
                "top1_mean_pct": ZERO_SHOT.get(dataset),
                "top1_std_pct": "",
                "macro_mean_pct": "",
                "macro_std_pct": "",
                "n": "paper",
                "run_id": "",
                "source": ZERO_SHOT_SOURCE,
            }
        )
        for method in METHOD_ORDER:
            method_rows = grouped.get((dataset, method), [])
            top1, top1_std = mean_std([float(row["test_top1"]) for row in method_rows])
            macro, macro_std = mean_std([float(row["test_macro"]) for row in method_rows if row.get("test_macro") is not None])
            run_ids = sorted({row["run_id"] for row in method_rows})
            out.append(
                {
                    "dataset": dataset,
                    "method": METHOD_LABEL[method],
                    "top1_mean_pct": "" if top1 is None else top1 * 100,
                    "top1_std_pct": "" if top1_std is None else top1_std * 100,
                    "macro_mean_pct": "" if macro is None else macro * 100,
                    "macro_std_pct": "" if macro_std is None else macro_std * 100,
                    "n": len(method_rows),
                    "run_id": ";".join(run_ids),
                    "source": "local PromptSRC-NC run" if method_rows else "not completed / not available",
                }
            )
    return out


def summarize_deltas(rows: list[dict]) -> list[dict]:
    per_seed: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("test_top1") is None:
            continue
        per_seed[(row["display_dataset"], int(row["seed"]))][row["method"]] = row

    out = []
    for dataset in ["EuroSAT", "Flowers102"]:
        top1_vs_promptsrc = []
        top1_vs_randomized = []
        macro_vs_promptsrc = []
        macro_vs_randomized = []
        paired = 0
        for (seed_dataset, _seed), values in sorted(per_seed.items()):
            if seed_dataset != dataset:
                continue
            real = values.get("PromptSRC-NC real")
            prompt = values.get("PromptSRC")
            shuffled = values.get("PromptSRC-NC shuffled")
            if not (real and prompt and shuffled):
                continue
            paired += 1
            top1_vs_promptsrc.append(float(real["test_top1"]) - float(prompt["test_top1"]))
            top1_vs_randomized.append(float(real["test_top1"]) - float(shuffled["test_top1"]))
            macro_vs_promptsrc.append(float(real["test_macro"]) - float(prompt["test_macro"]))
            macro_vs_randomized.append(float(real["test_macro"]) - float(shuffled["test_macro"]))

        top1_prompt_mean, top1_prompt_std = mean_std(top1_vs_promptsrc)
        top1_rand_mean, top1_rand_std = mean_std(top1_vs_randomized)
        macro_prompt_mean, macro_prompt_std = mean_std(macro_vs_promptsrc)
        macro_rand_mean, macro_rand_std = mean_std(macro_vs_randomized)
        out.append(
            {
                "dataset": dataset,
                "n_paired_seeds": paired,
                "top1_real_neighbors_minus_promptsrc_pp": "" if top1_prompt_mean is None else top1_prompt_mean * 100,
                "top1_real_neighbors_minus_promptsrc_std_pp": "" if top1_prompt_std is None else top1_prompt_std * 100,
                "top1_real_neighbors_minus_randomized_neighbors_pp": "" if top1_rand_mean is None else top1_rand_mean * 100,
                "top1_real_neighbors_minus_randomized_neighbors_std_pp": "" if top1_rand_std is None else top1_rand_std * 100,
                "macro_real_neighbors_minus_promptsrc_pp": "" if macro_prompt_mean is None else macro_prompt_mean * 100,
                "macro_real_neighbors_minus_promptsrc_std_pp": "" if macro_prompt_std is None else macro_prompt_std * 100,
                "macro_real_neighbors_minus_randomized_neighbors_pp": "" if macro_rand_mean is None else macro_rand_mean * 100,
                "macro_real_neighbors_minus_randomized_neighbors_std_pp": "" if macro_rand_std is None else macro_rand_std * 100,
            }
        )
    return out


def summarize_diagnostics(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("edge_disagreement") is not None:
            grouped[(row["display_dataset"], row["method"])].append(row)

    out = []
    for dataset in ["EuroSAT", "Flowers102"]:
        for method in METHOD_ORDER:
            method_rows = grouped.get((dataset, method), [])
            def avg(key: str) -> float | str:
                values = [float(row[key]) for row in method_rows if row.get(key) is not None]
                return "" if not values else statistics.fmean(values)

            out.append(
                {
                    "dataset": dataset,
                    "checkpoint_trained_as": METHOD_LABEL[method],
                    "diagnostic_graph": "fixed real frozen-CLIP neighbor edges",
                    "n": len(method_rows),
                    "edge_prediction_disagreement_pct": "" if not method_rows else avg("edge_disagreement") * 100,
                    "mean_js_on_real_neighbor_edges": avg("mean_js"),
                    "mean_entropy": avg("mean_entropy"),
                    "mean_confidence_pct": "" if not method_rows else avg("mean_confidence") * 100,
                }
            )
    return out


def per_seed_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("test_top1") is None:
            continue
        out.append(
            {
                "dataset": row["display_dataset"],
                "seed": row["seed"],
                "method": METHOD_LABEL[row["method"]],
                "test_top1_pct": float(row["test_top1"]) * 100,
                "test_macro_pct": float(row["test_macro"]) * 100,
                "val_top1_pct": "" if row.get("val_top1") is None else float(row["val_top1"]) * 100,
                "val_macro_pct": "" if row.get("val_macro") is None else float(row["val_macro"]) * 100,
                "checkpoint_role": row.get("checkpoint_role"),
                "run_id": row.get("run_id"),
            }
        )
    return out


def fmt_csv_num(value: object, places: int = 2) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{places}f}"


def text(svg: list[str], x: float, y: float, value: str, size: int = 14, anchor: str = "start", fill: str = "#1f2937", weight: str = "400") -> None:
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{escape(value)}</text>'
    )


def rect(svg: list[str], x: float, y: float, w: float, h: float, fill: str, stroke: str = "none") -> None:
    svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}"/>')


def line(svg: list[str], x1: float, y1: float, x2: float, y2: float, stroke: str = "#d1d5db", width: float = 1.0) -> None:
    svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"/>')


def write_macro_accuracy_svg(summary_rows: list[dict]) -> None:
    local = [row for row in summary_rows if row["method"] != "Zero-shot CLIP" and row["n"]]
    datasets = ["EuroSAT", "Flowers102"]
    methods = [METHOD_LABEL[m] for m in METHOD_ORDER]
    colors = {
        methods[0]: "#6b7280",
        methods[1]: "#2563eb",
        methods[2]: "#d97706",
    }
    by_key = {(row["dataset"], row["method"]): row for row in local}
    width, height = 1320, 760
    left, top = 290, 122
    plot_w = 850
    panel_gap = 58
    row_h = 44
    panel_h = row_h * len(methods)
    x_min, x_max = 88.0, 97.0
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    rect(svg, 0, 0, width, height, "#ffffff")
    text(svg, 54, 46, "Macro accuracy by method", 28, weight="700")
    text(svg, 54, 78, "Completed local runs only. Macro accuracy is the primary plotted metric.", 16, fill="#4b5563")

    chart_bottom = top + len(datasets) * panel_h + (len(datasets) - 1) * panel_gap
    for tick in range(88, 98):
        x = left + (tick - x_min) / (x_max - x_min) * plot_w
        line(svg, x, top - 16, x, chart_bottom + 24, "#e5e7eb")
        text(svg, x, chart_bottom + 48, str(tick), 12, anchor="middle", fill="#6b7280")
    line(svg, left, chart_bottom + 24, left + plot_w, chart_bottom + 24, "#9ca3af", 1.2)
    text(svg, left + plot_w / 2, chart_bottom + 78, "Test macro accuracy (%)", 14, anchor="middle", fill="#374151", weight="700")

    for i, dataset in enumerate(datasets):
        panel_top = top + i * (panel_h + panel_gap)
        text(svg, 54, panel_top + 28, dataset, 20, weight="700")
        n_values = []
        for method in methods:
            row = by_key.get((dataset, method))
            if row and row["n"]:
                n_values.append(str(row["n"]))
        if n_values:
            text(svg, 54, panel_top + 54, f"n={n_values[0]} completed seed{'s' if n_values[0] != '1' else ''}", 13, fill="#6b7280")
        for j, method in enumerate(methods):
            row = by_key.get((dataset, method))
            if not row or row["macro_mean_pct"] == "":
                continue
            macro = float(row["macro_mean_pct"])
            y = panel_top + j * row_h + 16
            text(svg, left - 18, y + 22, method, 14, anchor="end", fill="#374151")
            x0 = left
            x1 = left + (macro - x_min) / (x_max - x_min) * plot_w
            rect(svg, x0, y, max(x1 - x0, 2), 26, colors[method])
            label = f"{macro:.2f}"
            if row["macro_std_pct"] != "":
                label += f" +/- {float(row['macro_std_pct']):.2f}"
            text(svg, min(x1 + 10, left + plot_w + 8), y + 19, label, 13, fill="#111827", weight="700")
    svg.append("</svg>")
    (ROOT / "fig_macro_accuracy_by_method.svg").write_text("\n".join(svg), encoding="utf-8")


def write_macro_delta_svg(delta_rows: list[dict]) -> None:
    width, height = 1320, 620
    left, top = 430, 132
    plot_w = 760
    row_h = 54
    x_min, x_max = 0.0, 1.6
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    rect(svg, 0, 0, width, height, "#ffffff")
    text(svg, 54, 46, "Macro accuracy gain from real-neighbor Stage 2", 28, weight="700")
    text(svg, 54, 78, "Paired seed deltas. Positive values favor PromptSRC-NC trained with real neighbors.", 16, fill="#4b5563")

    rows_to_plot: list[tuple[str, str, float, str]] = []
    for row in delta_rows:
        rows_to_plot.append((row["dataset"], "NC real - PromptSRC", float(row["macro_real_neighbors_minus_promptsrc_pp"]), "#2563eb"))
        rows_to_plot.append((row["dataset"], "NC real - NC randomized", float(row["macro_real_neighbors_minus_randomized_neighbors_pp"]), "#059669"))

    chart_bottom = top + len(rows_to_plot) * row_h + 10
    for tick in [0.0, 0.4, 0.8, 1.2, 1.6]:
        x = left + (tick - x_min) / (x_max - x_min) * plot_w
        line(svg, x, top - 18, x, chart_bottom, "#e5e7eb")
        text(svg, x, chart_bottom + 25, f"{tick:.1f}", 12, anchor="middle", fill="#6b7280")
    line(svg, left, top - 18, left, chart_bottom, "#9ca3af", 1.2)
    text(svg, left + plot_w / 2, chart_bottom + 55, "Macro accuracy delta (percentage points)", 14, anchor="middle", weight="700", fill="#374151")

    last_dataset = None
    for i, (dataset, label, value, color) in enumerate(rows_to_plot):
        y = top + i * row_h
        if dataset != last_dataset:
            text(svg, 54, y + 23, dataset, 18, weight="700")
            last_dataset = dataset
        text(svg, left - 18, y + 23, label, 14, anchor="end", fill="#374151")
        w = (value - x_min) / (x_max - x_min) * plot_w
        rect(svg, left, y + 4, max(w, 2), 26, color)
        text(svg, left + w + 10, y + 23, f"{value:+.2f} pp", 13, fill="#111827", weight="700")
    svg.append("</svg>")
    (ROOT / "fig_macro_delta.svg").write_text("\n".join(svg), encoding="utf-8")


def write_disagreement_svg(diag_rows: list[dict]) -> None:
    width, height = 1320, 760
    left, top = 430, 132
    plot_w = 760
    row_h = 50
    x_min, x_max = 0.0, 12.0
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    rect(svg, 0, 0, width, height, "#ffffff")
    text(svg, 54, 46, "Prediction disagreement on real-neighbor edges", 28, weight="700")
    text(svg, 54, 78, "Post-hoc diagnostic on the same frozen-CLIP real-neighbor graph for every checkpoint. Lower is smoother.", 16, fill="#4b5563")

    colors = {
        METHOD_LABEL["PromptSRC"]: "#6b7280",
        METHOD_LABEL["PromptSRC-NC real"]: "#2563eb",
        METHOD_LABEL["PromptSRC-NC shuffled"]: "#d97706",
    }
    datasets = ["EuroSAT", "Flowers102"]
    by_key = {(row["dataset"], row["checkpoint_trained_as"]): row for row in diag_rows}
    rows_to_plot = []
    for dataset in datasets:
        for method in [METHOD_LABEL[m] for m in METHOD_ORDER]:
            row = by_key.get((dataset, method))
            if row:
                rows_to_plot.append((dataset, method, float(row["edge_prediction_disagreement_pct"])))

    chart_bottom = top + len(rows_to_plot) * row_h + 10
    for tick in range(0, 13, 2):
        x = left + (tick - x_min) / (x_max - x_min) * plot_w
        line(svg, x, top - 18, x, chart_bottom, "#e5e7eb")
        text(svg, x, chart_bottom + 25, str(tick), 12, anchor="middle", fill="#6b7280")
    line(svg, left, top - 18, left, chart_bottom, "#9ca3af", 1.2)
    text(svg, left + plot_w / 2, chart_bottom + 55, "Edge prediction disagreement (%)", 14, anchor="middle", weight="700", fill="#374151")

    last_dataset = None
    for i, (dataset, method, value) in enumerate(rows_to_plot):
        y = top + i * row_h
        if dataset != last_dataset:
            text(svg, 54, y + 23, dataset, 18, weight="700")
            last_dataset = dataset
        text(svg, left - 18, y + 23, method, 14, anchor="end", fill="#374151")
        w = (value - x_min) / (x_max - x_min) * plot_w
        rect(svg, left, y + 4, max(w, 2), 26, colors[method])
        text(svg, left + w + 10, y + 23, f"{value:.2f}", 13, fill="#111827", weight="700")
    svg.append("</svg>")
    (ROOT / "fig_edge_disagreement.svg").write_text("\n".join(svg), encoding="utf-8")


def write_markdown(summary_rows: list[dict], delta_rows: list[dict], diag_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# PromptSRC-NC Constrained Results Tables")
    lines.append("")
    lines.append("These tables summarize the completed constrained experiments at **16 shots per class** with **ViT-B/16 / OpenAI CLIP**. No line charts are included because shot count was not varied.")
    lines.append("")
    lines.append("## Scope And Caveats")
    lines.append("")
    lines.append("- EuroSAT completed for three seeds: `promptsrc-nc-main-20260510-192529`.")
    lines.append("- Flowers102 completed for one seed only: `promptsrc-nc-main-20260510-192533`. Treat this as a pilot result.")
    lines.append("- Stanford Cars did not complete Stage 1/evaluation: `promptsrc-nc-main-20260510-192531`. It is included only for the external zero-shot reference.")
    lines.append(f"- Zero-shot CLIP values are from **{ZERO_SHOT_SOURCE}**: [{ZERO_SHOT_URL}]({ZERO_SHOT_URL}). These are paper reference values, not rerun in this workspace.")
    lines.append("- Local PromptSRC-NC uses the unlabeled pool policy `full_training_split_minus_fewshot_labeled_train`; test images are not used as unlabeled data.")
    lines.append("- PromptSRC rows in the diagnostics section are diagnostic baselines: the checkpoint is evaluated on the same real-neighbor graph, but it was not trained with the neighborhood loss.")
    lines.append("")
    lines.append("## Local Macro Accuracy Table")
    lines.append("")
    lines.append("Macro test accuracy is the primary local metric in this constrained report. Top-1 is retained only in the CSV for auditability and for the external zero-shot reference, because the sourced zero-shot paper table does not report macro accuracy.")
    lines.append("")
    lines.append("| Dataset | Method | Macro (%) | n | Source |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in summary_rows:
        if row["method"] == "Zero-shot CLIP":
            continue
        macro = "-"
        if row["macro_mean_pct"] != "":
            mean = float(row["macro_mean_pct"])
            std = row["macro_std_pct"]
            macro = f"{mean:.2f}" if std == "" else f"{mean:.2f} +/- {float(std):.2f}"
        lines.append(f"| {row['dataset']} | {row['method']} | {macro} | {row['n']} | {row['source']} |")
    lines.append("")
    lines.append("## External Zero-Shot Reference")
    lines.append("")
    lines.append("The available sourced zero-shot values are top-1 only, so they are kept separate from the macro-primary local comparison.")
    lines.append("")
    lines.append("| Dataset | Zero-shot CLIP top-1 (%) | Source |")
    lines.append("| --- | --- | --- |")
    for dataset in ["EuroSAT", "Flowers102", "Stanford Cars"]:
        lines.append(f"| {dataset} | {ZERO_SHOT[dataset]:.1f} | {ZERO_SHOT_SOURCE} |")
    lines.append("")
    lines.append("## Paired Effect Sizes")
    lines.append("")
    lines.append("Deltas are percentage points. Positive values mean **PromptSRC-NC trained with real neighbors** is higher than the named comparator.")
    lines.append("")
    lines.append("| Dataset | Paired seeds | Macro: NC real - PromptSRC | Macro: NC real - NC randomized |")
    lines.append("| --- | --- | --- | --- |")
    for row in delta_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    str(row["n_paired_seeds"]),
                    fmt_csv_num(row["macro_real_neighbors_minus_promptsrc_pp"]),
                    fmt_csv_num(row["macro_real_neighbors_minus_randomized_neighbors_pp"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Diagnostics Table")
    lines.append("")
    lines.append("This table does **not** say every method used neighborhood training. It asks: after training each checkpoint, how often do its predictions disagree across the fixed real frozen-CLIP neighbor edges? Lower edge disagreement means smoother predictions on that graph.")
    lines.append("")
    lines.append("| Dataset | Checkpoint trained as | Diagnostic graph | n | Edge prediction disagreement (%) | Mean JS | Mean entropy | Mean confidence (%) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in diag_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["checkpoint_trained_as"],
                    row["diagnostic_graph"],
                    str(row["n"]),
                    fmt_csv_num(row["edge_prediction_disagreement_pct"]),
                    fmt_csv_num(row["mean_js_on_real_neighbor_edges"], 4),
                    fmt_csv_num(row["mean_entropy"], 4),
                    fmt_csv_num(row["mean_confidence_pct"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Visualizations")
    lines.append("")
    lines.append("![Macro accuracy by method](fig_macro_accuracy_by_method.svg)")
    lines.append("")
    lines.append("![Macro accuracy gain from real-neighbor Stage 2](fig_macro_delta.svg)")
    lines.append("")
    lines.append("![Prediction disagreement on fixed real-neighbor graph](fig_edge_disagreement.svg)")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `summary_accuracy.csv`: method comparison table with macro accuracy plus top-1 retained for audit/reference.")
    lines.append("- `delta_summary.csv`: paired macro and top-1 effect-size table; report and charts use macro.")
    lines.append("- `per_seed_accuracy.csv`: per-seed local test/validation metrics and checkpoint roles.")
    lines.append("- `diagnostics_summary.csv`: averaged diagnostics by dataset and checkpoint type.")
    (ROOT / "results_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_runs()
    summary_rows = summarize_accuracy(rows)
    delta_rows = summarize_deltas(rows)
    diag_rows = summarize_diagnostics(rows)
    seed_rows = per_seed_rows(rows)

    write_csv(
        ROOT / "summary_accuracy.csv",
        summary_rows,
        ["dataset", "method", "top1_mean_pct", "top1_std_pct", "macro_mean_pct", "macro_std_pct", "n", "run_id", "source"],
    )
    write_csv(
        ROOT / "delta_summary.csv",
        delta_rows,
        [
            "dataset",
            "n_paired_seeds",
            "top1_real_neighbors_minus_promptsrc_pp",
            "top1_real_neighbors_minus_promptsrc_std_pp",
            "top1_real_neighbors_minus_randomized_neighbors_pp",
            "top1_real_neighbors_minus_randomized_neighbors_std_pp",
            "macro_real_neighbors_minus_promptsrc_pp",
            "macro_real_neighbors_minus_promptsrc_std_pp",
            "macro_real_neighbors_minus_randomized_neighbors_pp",
            "macro_real_neighbors_minus_randomized_neighbors_std_pp",
        ],
    )
    write_csv(
        ROOT / "diagnostics_summary.csv",
        diag_rows,
        [
            "dataset",
            "checkpoint_trained_as",
            "diagnostic_graph",
            "n",
            "edge_prediction_disagreement_pct",
            "mean_js_on_real_neighbor_edges",
            "mean_entropy",
            "mean_confidence_pct",
        ],
    )
    write_csv(
        ROOT / "per_seed_accuracy.csv",
        seed_rows,
        ["dataset", "seed", "method", "test_top1_pct", "test_macro_pct", "val_top1_pct", "val_macro_pct", "checkpoint_role", "run_id"],
    )
    write_markdown(summary_rows, delta_rows, diag_rows)
    write_macro_accuracy_svg(summary_rows)
    write_macro_delta_svg(delta_rows)
    write_disagreement_svg(diag_rows)
    for stale in ("fig_accuracy_by_method.svg", "fig_delta_vs_baselines.svg"):
        path = ROOT / stale
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()

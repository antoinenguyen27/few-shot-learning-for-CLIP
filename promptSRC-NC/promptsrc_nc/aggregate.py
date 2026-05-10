"""Aggregate PromptSRC-NC evaluation, diagnostics, runtime, and cost logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

from .config import results_dir
from .structured_logging import read_jsonl, write_json


def _write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output


def _variant(row: dict[str, Any]) -> str:
    method = row.get("method", "")
    pair_mode = row.get("pair_mode", "")
    if method == "PromptSRC":
        return "PromptSRC"
    if pair_mode == "real":
        return "PromptSRC-NC real"
    if pair_mode == "shuffled":
        return "PromptSRC-NC shuffled"
    return f"{method} {pair_mode}".strip()


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def aggregate_run(run_root: str | Path, run_id: str) -> dict[str, Any]:
    root = Path(run_root) / run_id
    out_dir = results_dir(run_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = read_jsonl(root / "logs" / "eval_summary.jsonl")
    diag_rows = read_jsonl(root / "logs" / "diagnostics.jsonl")
    runtime_rows = []
    runtime_rows.extend(read_jsonl(root / "logs" / "train_stage1.jsonl"))
    runtime_rows.extend(read_jsonl(root / "logs" / "train_stage2.jsonl"))
    for path in sorted(root.glob("stage1/**/logs/runtime.jsonl")):
        for row in read_jsonl(path):
            runtime_rows.append({"runtime_log": str(path), **row})
    for path in sorted(root.glob("stage2/**/logs/runtime.jsonl")):
        for row in read_jsonl(path):
            runtime_rows.append({"runtime_log": str(path), **row})
    cost_rows = read_jsonl(root / "logs" / "cost_profile.jsonl")

    run_index: dict[tuple[str, int, int, str, str, str], dict[str, Any]] = {}
    for row in eval_rows:
        split = row.get("split")
        if split not in {"val", "test"}:
            continue
        variant = _variant(row)
        key = (
            str(row.get("dataset")),
            int(row.get("shots")),
            int(row.get("seed")),
            str(row.get("backbone")),
            str(row.get("pair_mode")),
            variant,
        )
        existing = run_index.setdefault(
            key,
            {
                "run_id": run_id,
                "method": variant,
                "pair_mode": row.get("pair_mode"),
                "dataset": row.get("dataset"),
                "shots": row.get("shots"),
                "seed": row.get("seed"),
                "backbone": row.get("backbone"),
                "checkpoint": row.get("checkpoint"),
                "val_top1": None,
                "val_macro": None,
                "test_top1": None,
                "test_macro": None,
                "edge_disagreement": None,
                "mean_js": None,
                "mean_entropy": None,
                "mean_confidence": None,
            }
        )
        existing["checkpoint"] = row.get("checkpoint") or existing.get("checkpoint")
        existing[f"{split}_top1"] = row.get("top1_accuracy")
        existing[f"{split}_macro"] = row.get("macro_accuracy")

    run_rows = list(run_index.values())
    for row in run_rows:
        diagnostic = next(
            (
                diag
                for diag in diag_rows
                if diag.get("dataset") == row.get("dataset")
                and diag.get("shots") == row.get("shots")
                and diag.get("seed") == row.get("seed")
                and _variant(diag) == row.get("method")
            ),
            {},
        )
        row["edge_disagreement"] = diagnostic.get("edge_disagreement")
        row["mean_js"] = diagnostic.get("mean_js")
        row["mean_entropy"] = diagnostic.get("mean_entropy")
        row["mean_confidence"] = diagnostic.get("mean_confidence")
    with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in run_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    summary_rows = []
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    per_seed: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in run_rows:
        if row.get("test_top1") is None:
            continue
        key = (str(row["dataset"]), int(row["shots"]))
        variant = str(row["method"])
        grouped.setdefault(key, {}).setdefault(variant, []).append(float(row["test_top1"]))
        per_seed.setdefault((str(row["dataset"]), int(row["shots"]), int(row["seed"])), {})[variant] = float(row["test_top1"])
    for (dataset, shots), variants in sorted(grouped.items()):
        prompt_mean, prompt_std = _mean_std(variants.get("PromptSRC", []))
        shuffled_mean, shuffled_std = _mean_std(variants.get("PromptSRC-NC shuffled", []))
        real_mean, real_std = _mean_std(variants.get("PromptSRC-NC real", []))
        seed_deltas_prompt = []
        seed_deltas_shuffled = []
        for (seed_dataset, seed_shots, _seed), seed_values in per_seed.items():
            if seed_dataset != dataset or seed_shots != shots:
                continue
            if "PromptSRC-NC real" in seed_values and "PromptSRC" in seed_values:
                seed_deltas_prompt.append(seed_values["PromptSRC-NC real"] - seed_values["PromptSRC"])
            if "PromptSRC-NC real" in seed_values and "PromptSRC-NC shuffled" in seed_values:
                seed_deltas_shuffled.append(seed_values["PromptSRC-NC real"] - seed_values["PromptSRC-NC shuffled"])
        delta_prompt, _ = _mean_std(seed_deltas_prompt)
        delta_shuffled, _ = _mean_std(seed_deltas_shuffled)
        summary_rows.append(
            {
                "dataset": dataset,
                "shots": shots,
                "promptsrc_mean": prompt_mean,
                "promptsrc_std": prompt_std,
                "promptsrc_nc_shuffled_mean": shuffled_mean,
                "promptsrc_nc_shuffled_std": shuffled_std,
                "promptsrc_nc_real_mean": real_mean,
                "promptsrc_nc_real_std": real_std,
                "real_minus_promptsrc": delta_prompt,
                "real_minus_shuffled": delta_shuffled,
                "num_promptsrc": len(variants.get("PromptSRC", [])),
                "num_shuffled": len(variants.get("PromptSRC-NC shuffled", [])),
                "num_real": len(variants.get("PromptSRC-NC real", [])),
            }
        )
    if summary_rows:
        by_shots: dict[int, list[dict[str, Any]]] = {}
        for row in summary_rows:
            by_shots.setdefault(int(row["shots"]), []).append(row)
        for shots, rows in sorted(by_shots.items()):
            avg_row = {"dataset": "Average", "shots": shots}
            for col in (
                "promptsrc_mean",
                "promptsrc_nc_shuffled_mean",
                "promptsrc_nc_real_mean",
                "real_minus_promptsrc",
                "real_minus_shuffled",
            ):
                vals = [float(row[col]) for row in rows if row.get(col) is not None]
                avg_row[col] = float(mean(vals)) if vals else None
            summary_rows.append(avg_row)

    diagnostics_summary = [
        {
            "run_id": run_id,
            "method": _variant(row),
            "pair_mode": row.get("pair_mode"),
            "dataset": row.get("dataset"),
            "shots": row.get("shots"),
            "seed": row.get("seed"),
            "edge_disagreement": row.get("edge_disagreement"),
            "mean_js": row.get("mean_js"),
            "mean_entropy": row.get("mean_entropy"),
            "mean_confidence": row.get("mean_confidence"),
            "mean_real_cosine": row.get("mean_real_cosine"),
            "mean_shuffled_cosine": row.get("mean_shuffled_cosine"),
        }
        for row in diag_rows
    ]
    _write_csv(out_dir / "eval_summary.csv", summary_rows)
    write_json(out_dir / "eval_summary.json", {"rows": summary_rows})
    _write_csv(out_dir / "diagnostics_summary.csv", diagnostics_summary)
    _write_csv(out_dir / "runtime_summary.csv", runtime_rows)
    _write_csv(out_dir / "cost_profile_summary.csv", cost_rows)
    return {
        "run_id": run_id,
        "results_dir": str(out_dir),
        "num_eval_rows": len(eval_rows),
        "num_test_run_rows": len(run_rows),
        "num_diagnostics_rows": len(diag_rows),
        "summary_rows": summary_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate PromptSRC-NC run artifacts.")
    parser.add_argument("--run-root", default="/vol/runs")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate_run(args.run_root, args.run_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

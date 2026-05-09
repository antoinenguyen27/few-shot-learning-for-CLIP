"""Small JSON/JSONL and runtime logging helpers."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {file_path}:{line_number}") from exc
    return records


class JsonlLogger:
    def __init__(self, path: str | Path, base: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.base = dict(base or {})
        self.start = time.perf_counter()

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "timestamp": utc_now(),
            "seconds_since_start": time.perf_counter() - self.start,
            **self.base,
            **payload,
        }
        append_jsonl(self.path, record)


def runtime_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp": utc_now(),
    }
    try:
        import psutil

        process = psutil.Process(os.getpid())
        record["rss_mb"] = process.memory_info().rss / (1024**2)
        record["cpu_percent"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            record["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            record["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            record["max_cuda_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            record["max_cuda_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024**2)
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        record["gpu_utilization_percent"] = util.gpu
        record["gpu_memory_used_mb"] = mem.used / (1024**2)
        record["gpu_memory_total_mb"] = mem.total / (1024**2)
    except Exception:
        pass
    return record


from __future__ import annotations

import io
import json
from pathlib import Path

from promptsrc_nc.structured_logging import JsonlLogger, emit_status


class FlushCountingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_emit_status_writes_one_flushable_json_line() -> None:
    stream = FlushCountingStream()

    record = emit_status(
        "train_step",
        stream=stream,
        run_id="run-a",
        stage="stage1",
        epoch=2,
        step=20,
        loss_total=1.25,
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    decoded = json.loads(lines[0])
    assert decoded["event"] == "train_step"
    assert decoded["run_id"] == "run-a"
    assert decoded["stage"] == "stage1"
    assert decoded["epoch"] == 2
    assert decoded["step"] == 20
    assert decoded["loss_total"] == 1.25
    assert "timestamp" in decoded
    assert record == decoded
    assert stream.flush_count == 1


def test_jsonl_logger_can_mirror_selected_events_to_status_stream(tmp_path: Path) -> None:
    stream = FlushCountingStream()
    logger = JsonlLogger(
        tmp_path / "train.jsonl",
        base={"run_id": "run-a", "stage": "stage1"},
        live=True,
        live_events={"epoch_end"},
        stream=stream,
    )

    logger.log("train_step", epoch=1, step=20, loss_total=2.0)
    logger.log("epoch_end", epoch=1, val_top1=0.7)

    live_lines = stream.getvalue().splitlines()
    file_lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(live_lines) == 1
    assert len(file_lines) == 2
    decoded = json.loads(live_lines[0])
    assert decoded["event"] == "epoch_end"
    assert decoded["run_id"] == "run-a"
    assert decoded["stage"] == "stage1"
    assert decoded["val_top1"] == 0.7

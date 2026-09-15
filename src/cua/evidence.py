"""
Evidence: structured logs and richer signals on failure.

One JSONL file per run. Every entry is a typed event with a monotonic sequence number, so the
log is diffable between runs and greppable by step. Screenshots are captured at every step
during discovery (cheap, and we want the record) but only on failure during replay (replay runs
in production volume, where writing a PNG per step is a real cost).

Everything written here has already passed through the Redactor at the observation boundary,
so no scrubbing happens at this layer. That is deliberate: one scrubbing site, not two.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EvidenceWriter:
    run_id: str
    run_kind: str  # "discovery" | "replay"
    root: Path
    # Scrubbing happens here, at the single point where anything is written to disk.
    # The surface redacts what it READS off the screen; this redacts what we WRITE about the
    # run, including planner arguments, which never touch the surface at all. Two boundaries,
    # because there are genuinely two directions data can flow.
    redactor: Any | None = None
    _seq: int = field(default=0, init=False)
    _t0: float = field(default_factory=time.monotonic, init=False)

    @classmethod
    def create(
        cls, run_kind: str, root: str | Path = "evidence", redactor: Any | None = None
    ) -> EvidenceWriter:
        run_id = f"{run_kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        path = Path(root) / run_id
        path.mkdir(parents=True, exist_ok=True)
        w = cls(run_id=run_id, run_kind=run_kind, root=path, redactor=redactor)
        w.log("run_started", kind=run_kind, run_id=run_id)
        return w

    def _scrub(self, value: Any) -> Any:
        """Recursively scrub strings anywhere in a logged structure."""
        if self.redactor is None:
            return value
        if isinstance(value, str):
            return self.redactor.scrub(value)
        if isinstance(value, dict):
            return {k: self._scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub(v) for v in value]
        return value

    @property
    def log_path(self) -> Path:
        return self.root / "run.jsonl"

    def log(self, event: str, **fields: Any) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "t_ms": round((time.monotonic() - self._t0) * 1000),
            "event": event,
            **self._scrub(fields),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def screenshot_path(self, label: str) -> str:
        return str(self.root / f"{self._seq:03d}-{label}.png")

    def write_json(self, name: str, data: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

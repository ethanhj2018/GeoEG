"""Per-run directory runs/<timestamp>_<image>/ with an append-only JSONL event log."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, runs_dir: str, image_path: str):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(runs_dir) / f"{stamp}_{Path(image_path).stem[:40]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "crops").mkdir(exist_ok=True)
        (self.dir / "masks").mkdir(exist_ok=True)
        self.path = self.dir / "log.jsonl"
        self.timings: dict[str, float] = {}

    def event(self, stage: str, **data) -> None:
        rec = {"t": round(time.time(), 3), "stage": stage, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")

    def timed(self, stage: str):
        return _Timer(self, stage)

    def write_json(self, name: str, obj) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
        return p


class _Timer:
    def __init__(self, log: RunLog, stage: str):
        self.log, self.stage = log, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.log.timings[self.stage] = round(self.log.timings.get(self.stage, 0.0) + dt, 2)
        self.log.event(self.stage, elapsed_s=round(dt, 2), error=repr(exc[1]) if exc[1] else None)
        return False


def _default(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    return str(o)

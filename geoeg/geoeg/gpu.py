"""Load-on-demand model manager + whole-card VRAM sampler.

Only one local model is resident at a time (vLLM runs in its own process and is counted by the sampler);
each `with gpu.use(...)` loads, yields, then deletes the model and empties the CUDA cache.
"""
from __future__ import annotations

import contextlib
import gc
import logging
import subprocess
import threading
import time

log = logging.getLogger("geoeg.gpu")


def card_used_mib() -> float | None:
    """Whole-card used memory (all processes incl. vLLM and the Windows desktop), via nvidia-smi.

    torch.cuda.mem_get_info() is not used: under WSL2 it only reflects this process, not the whole card.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class GPUManager:
    def __init__(self, budget_mib: float = 14336, interval_s: float = 0.25):
        self.budget_mib = budget_mib
        self.interval_s = interval_s
        self.baseline_mib = card_used_mib()     # before this run loads anything (vLLM + desktop)
        self.peak_mib = 0.0
        self.peaks: dict[str, float] = {}      # per-model whole-card peak
        self.own_peaks: dict[str, float] = {}  # per-model peak reserved by this process
        self._current: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- sampler -------------------------------------------------------------
    def start(self) -> "GPUManager":
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        used = card_used_mib()
        if used is None:
            return
        self.peak_mib = max(self.peak_mib, used)
        if self._current:
            self.peaks[self._current] = max(self.peaks.get(self._current, 0.0), used)
        if used > self.budget_mib:
            log.warning("整卡显存 %.0f MiB 超过预算 %.0f MiB（当前模型：%s）", used, self.budget_mib, self._current)

    # --- load / release ------------------------------------------------------
    @contextlib.contextmanager
    def use(self, name: str, loader):
        """`loader()` returns the model object; it is released on exit."""
        import torch
        self._current = name
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = loader()
        log.info("loaded %s in %.1fs (card %.0f MiB)", name, time.time() - t0, card_used_mib() or -1)
        try:
            yield model
        finally:
            self._sample()
            if torch.cuda.is_available():
                self.own_peaks[name] = torch.cuda.max_memory_reserved() / 2**20
            # The caller's `as` name still points at the wrapper after the with-block, so drop the
            # wrapper's attributes (the actual nn.Modules) to really free the weights.
            if hasattr(model, "__dict__"):
                model.__dict__.clear()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._current = None
            self._sample()

    def report(self) -> dict:
        return {**self._report(), "per_model_own_peak_reserved_mib": {k: round(v) for k, v in self.own_peaks.items()}}

    def _report(self) -> dict:
        return {"card_peak_mib": round(self.peak_mib), "budget_mib": self.budget_mib,
                "card_baseline_mib": round(self.baseline_mib) if self.baseline_mib is not None else None,
                "per_model_card_peak_mib": {k: round(v) for k, v in self.peaks.items()}}

"""Progress reporting with an ETA (requirement R19).

Falls back to ``logging`` when ``tqdm`` is not installed, so there is no hard dependency.
"""

from __future__ import annotations

import logging
import sys
import time

logger = logging.getLogger(__name__)

try:  # optional accelerator
    from tqdm import tqdm as _tqdm

    _HAVE_TQDM = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_TQDM = False


class ProgressReporter:
    """A small progress reporter that prints to stderr with an ETA.

    Parameters
    ----------
    total : int
        Total number of steps.
    description : str
        Label shown in front of the bar.
    use_bar : bool
        Use a tqdm bar when available; otherwise log at ~10% intervals.
    """

    def __init__(self, total: int, description: str = "", use_bar: bool = True) -> None:
        self.total = max(1, int(total))
        self.description = description
        self.start = time.time()
        self._last_log = 0.0
        self._bar = None
        if use_bar and _HAVE_TQDM:
            self._bar = _tqdm(total=self.total, desc=description, file=sys.stderr)

    def __call__(self, index: int, total: int, message: str = "") -> None:
        """Update the progress; matches the ``progress`` callback signature of the pipeline."""
        if total and total != self.total:
            self.total = max(1, int(total))
            if self._bar is not None:
                self._bar.total = self.total
        if self._bar is not None:
            self._bar.n = min(index + 1, self.total)
            self._bar.set_postfix_str(message[:40])
            self._bar.refresh()
            if index + 1 >= self.total:
                self._bar.close()
            return
        now = time.time()
        if now - self._last_log < 2.0 and index + 1 < self.total:
            return
        self._last_log = now
        elapsed = now - self.start
        frac = (index + 1) / self.total
        eta = elapsed / max(frac, 1e-9) * (1.0 - frac)
        logger.info(
            "[%s] %d/%d (%.0f%%) elapsed %.0fs, ETA %.0fs %s",
            self.description or "progress",
            index + 1,
            self.total,
            100 * frac,
            elapsed,
            eta,
            message,
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

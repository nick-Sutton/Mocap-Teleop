"""
freq_meter.py — Lightweight rolling-window frequency estimator.

Usage
─────
    meter = FrequencyMeter(window=100)

    # Call once per event whose rate you want to measure:
    meter.tick()

    # Read the current estimate (Hz):
    print(meter.hz)
"""

import collections
import time


class FrequencyMeter:
    """Estimates event frequency over a sliding window of recent timestamps."""

    def __init__(self, window: int = 100):
        """
        Parameters
        ----------
        window:
            Maximum number of timestamps kept. Frequency is computed from
            the oldest to the newest timestamp in the buffer, so larger
            windows give smoother but slower-to-respond estimates.
            At 1000 Hz a window of 100 covers the last ~0.1 s.
            At 240 Hz a window of 100 covers the last ~0.4 s.
        """
        self._times: collections.deque[float] = collections.deque(maxlen=window)

    def tick(self) -> None:
        """Record the current wall-clock time as one event."""
        self._times.append(time.monotonic())

    @property
    def hz(self) -> float:
        """Current frequency estimate in Hz. Returns 0.0 if not enough data."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0.0 else 0.0

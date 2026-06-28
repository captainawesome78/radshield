"""
radshield.guard
===============

The protection primitives. Two cheap, composable mechanisms that together kill
the catastrophic failure mode:

1. WeightGuard       -- detect-and-repair corrupted weights via fast checksums
                        plus a verified golden reference. This is "ECC for your
                        model" implemented in software, so it works on any chip,
                        including commodity GPUs that lack hardware ECC on all
                        memory paths.

2. OutputSanitizer   -- a last line of defence between checksum passes. Clamps
                        activations/outputs to a calibrated plausible range and
                        scrubs NaN/Inf, so a flip that happens mid-inference can
                        not propagate an explosion to the user.

Both are designed to be near-free: a checksum is a few MB/ms; sanitizing is one
elementwise clip. The expensive option (redundant recompute) is opt-in.
"""

from __future__ import annotations
import numpy as np
import zlib


def _checksum(a: np.ndarray) -> int:
    """Fast, dependency-free checksum over raw bytes (CRC32)."""
    return zlib.crc32(np.ascontiguousarray(a).tobytes())


class WeightGuard:
    """Wraps a dict of named float32 weight arrays. Holds a verified golden copy
    plus per-tensor checksums, and repairs any tensor whose checksum drifts.

    In real hardware the golden copy would itself be triple-redundant / in
    ECC-protected storage; this v0 keeps the mechanism honest and measurable.
    """

    def __init__(self, params: dict[str, np.ndarray]):
        self._golden = {k: v.copy() for k, v in params.items()}
        self._sums = {k: _checksum(v) for k, v in params.items()}
        self.repairs = 0
        self.checks = 0

    def verify_and_repair(self, params: dict[str, np.ndarray]) -> int:
        """Check each tensor; restore from golden on mismatch. Returns number of
        tensors repaired this call. Mutates ``params`` in place."""
        repaired = 0
        for k, v in params.items():
            self.checks += 1
            if _checksum(v) != self._sums[k]:
                params[k][...] = self._golden[k]
                self.repairs += 1
                repaired += 1
        return repaired


class OutputSanitizer:
    """Clamps values to a plausible range and removes NaN/Inf. Calibrate the
    range once on clean data, then apply on every forward pass."""

    def __init__(self, lo: float, hi: float):
        self.lo = float(lo)
        self.hi = float(hi)

    @classmethod
    def from_calibration(cls, clean_values: np.ndarray, margin: float = 4.0):
        """Set the clamp range from the magnitude of clean activations, with a
        safety ``margin`` so normal variation is never touched."""
        peak = float(np.max(np.abs(clean_values))) * margin
        return cls(-peak, peak)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=self.hi, neginf=self.lo)
        return np.clip(x, self.lo, self.hi)

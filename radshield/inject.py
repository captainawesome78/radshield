"""
radshield.inject
================

Simulates single-event upsets (SEUs): cosmic-ray-induced bit flips in the
memory that holds model weights and activations.

A float32 number is laid out as:

    bit 31      bits 30..23      bits 22..0
    [ sign ]    [ exponent  ]    [ mantissa ]

Flipping a high exponent bit can turn 0.01 into ~1e30 (or NaN/Inf) -- this is
the catastrophic failure mode that destroys a model's output. Flipping a low
mantissa bit barely changes the value. radshield exists to make the first case
survivable. The injector here is both the test harness *and* a feature: users
inject faults to validate their own protection before they ever reach orbit.
"""

from __future__ import annotations
import numpy as np

FLOAT32_BITS = 32
SIGN_BIT = 31
EXPONENT_BITS = tuple(range(23, 31))   # 8 bits
MANTISSA_BITS = tuple(range(0, 23))    # 23 bits


def flip_bits_at(x: np.ndarray, flat_indices: np.ndarray, bit: int) -> np.ndarray:
    """Return a copy of float32 array ``x`` with ``bit`` flipped at the given
    flattened element indices. Pure, non-mutating."""
    if x.dtype != np.float32:
        raise TypeError(f"radshield operates on float32, got {x.dtype}")
    out = x.copy()
    u = out.reshape(-1).view(np.uint32)
    mask = np.uint32(1) << np.uint32(bit)
    u[flat_indices] ^= mask
    return out


def inject_bit_flips(
    x: np.ndarray,
    n_flips: int = 1,
    bit: int | None = None,
    region: str = "any",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject ``n_flips`` random bit flips into a copy of ``x``.

    Parameters
    ----------
    n_flips : how many bits to flip (one SEU each).
    bit     : pin a specific bit position 0..31; if None, chosen per ``region``.
    region  : 'any' | 'sign' | 'exponent' | 'mantissa' -- which bits are eligible
              when ``bit`` is None. Lets you study where the damage comes from.
    """
    rng = rng or np.random.default_rng()
    out = x
    n_elems = x.size
    if region == "sign":
        bit_pool = [SIGN_BIT]
    elif region == "exponent":
        bit_pool = list(EXPONENT_BITS)
    elif region == "mantissa":
        bit_pool = list(MANTISSA_BITS)
    else:
        bit_pool = list(range(FLOAT32_BITS))

    for _ in range(n_flips):
        idx = rng.integers(0, n_elems)
        b = bit if bit is not None else int(rng.choice(bit_pool))
        out = flip_bits_at(out, np.array([idx]), b)
    return out


def sweep_single_bit(x: np.ndarray) -> dict[int, np.ndarray]:
    """For diagnostics: return {bit: corrupted_copy} flipping one fixed element
    at every bit position. Useful to visualise per-bit blast radius."""
    mid = x.size // 2
    return {b: flip_bits_at(x, np.array([mid]), b) for b in range(FLOAT32_BITS)}

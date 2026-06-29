"""
radshield.torch
===============

PyTorch adapter. Protect a real model with one call::

    import radshield.torch as rst

    handle = rst.protect(model, clean_batch)   # checksums weights + calibrates
    # ... model runs in orbit; bit flips happen ...
    y = model(x)                                # weights auto-repaired, output sane
    print(handle.guard.repairs, "upsets repaired")

The mechanism mirrors the numpy core but stays torch-native (no per-inference
host<->device copies): checksums are reductions over the integer bit-pattern
view, repair is an in-place ``copy_`` from a golden clone, sanitization is one
``nan_to_num`` + ``clamp``. Works on CPU and GPU.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

__all__ = ["protect", "ProtectedModel", "TorchWeightGuard",
           "TorchOutputSanitizer", "inject_bit_flips"]


# --------------------------------------------------------------------------- #
# fault injection (test harness / validation feature)                         #
# --------------------------------------------------------------------------- #
_EXPONENT_BITS = tuple(range(23, 31))
_MANTISSA_BITS = tuple(range(0, 23))


def _mask_for_bit(bit: int) -> int:
    # correct signed-int32 value with `bit` set (bit 31 -> negative)
    return int(np.int32(np.uint32(1) << np.uint32(bit)))


def inject_bit_flips(t: torch.Tensor, n_flips: int = 1, bit: int | None = None,
                     region: str = "any",
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Return a copy of float32 tensor ``t`` with ``n_flips`` single-bit upsets.
    ``region`` in {'any','sign','exponent','mantissa'} when ``bit`` is None."""
    if t.dtype != torch.float32:
        raise TypeError(f"radshield operates on float32, got {t.dtype}")
    pool = {"sign": [31], "exponent": list(_EXPONENT_BITS),
            "mantissa": list(_MANTISSA_BITS)}.get(region, list(range(32)))
    out = t.clone()
    iview = out.view(-1).view(torch.int32)
    n = iview.numel()
    for _ in range(n_flips):
        idx = int(torch.randint(0, n, (1,), generator=generator).item())
        b = bit if bit is not None else int(
            pool[torch.randint(0, len(pool), (1,), generator=generator).item()])
        iview[idx] ^= _mask_for_bit(b)
    return out


# --------------------------------------------------------------------------- #
# guards                                                                       #
# --------------------------------------------------------------------------- #
def _checksum(t: torch.Tensor) -> int:
    # sum over the int32 bit-pattern view -> any single-bit flip changes it.
    return int(t.detach().contiguous().view(torch.int32)
               .to(torch.int64).sum().item())


class TorchWeightGuard:
    """Holds golden clones + checksums of a module's float32 parameters and
    repairs any that drift. Guard all params, or only a named subset."""

    def __init__(self, module: nn.Module, names: list[str] | None = None):
        self._params = {}
        for name, p in module.named_parameters():
            if p.dtype == torch.float32 and (names is None or name in names):
                self._params[name] = p
        self._golden = {n: p.detach().clone() for n, p in self._params.items()}
        self._sums = {n: _checksum(p) for n, p in self._params.items()}
        self.repairs = 0
        self.checks = 0

    @torch.no_grad()
    def verify_and_repair(self) -> int:
        repaired = 0
        for name, p in self._params.items():
            self.checks += 1
            if _checksum(p) != self._sums[name]:
                p.data.copy_(self._golden[name])
                self.repairs += 1
                repaired += 1
        return repaired


class TorchOutputSanitizer:
    """Clamp to a calibrated range and scrub NaN/Inf."""

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = float(lo), float(hi)

    @classmethod
    def from_peak(cls, peak: float, margin: float = 4.0):
        return cls(-peak * margin, peak * margin)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=self.hi, neginf=self.lo)
        return x.clamp(self.lo, self.hi)


# --------------------------------------------------------------------------- #
# one-call protection                                                          #
# --------------------------------------------------------------------------- #
class ProtectedModel:
    """Handle returned by ``protect``. Installs a forward-pre-hook that repairs
    weights before each forward, and forward-hooks that sanitize chosen layers'
    outputs. Call ``.remove()`` to uninstall."""

    def __init__(self, module: nn.Module, guard: TorchWeightGuard,
                 sanitizers: dict[str, TorchOutputSanitizer]):
        self.module = module
        self.guard = guard
        self.sanitizers = sanitizers
        self._handles = []
        self._install()

    def _install(self):
        self._handles.append(
            self.module.register_forward_pre_hook(
                lambda m, inp: self.guard.verify_and_repair() and None))
        submods = dict(self.module.named_modules())
        for name, san in self.sanitizers.items():
            def hook(m, inp, out, _san=san):
                return _san(out) if isinstance(out, torch.Tensor) else out
            self._handles.append(submods[name].register_forward_hook(hook))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _leaf_layers_with_params(module: nn.Module) -> list[str]:
    out = []
    for name, m in module.named_modules():
        if name == "":
            continue
        if any(p.requires_grad for p in m.parameters(recurse=False)):
            out.append(name)
    return out


@torch.no_grad()
def protect(module: nn.Module, clean_batch: torch.Tensor,
            layers: list[str] | None = None, margin: float = 4.0) -> ProtectedModel:
    """Protect ``module``. Checksums all float32 params; calibrates output
    sanitizers on ``clean_batch`` for the chosen ``layers`` (default: every leaf
    layer that owns parameters)."""
    guard = TorchWeightGuard(module)
    target_names = layers if layers is not None else _leaf_layers_with_params(module)

    # calibrate: record peak |output| per target layer on the clean batch
    submods = dict(module.named_modules())
    peaks, tmp = {}, []
    for name in target_names:
        def rec(m, inp, out, _n=name):
            if isinstance(out, torch.Tensor):
                peaks[_n] = max(peaks.get(_n, 0.0), float(out.abs().max().item()))
        tmp.append(submods[name].register_forward_hook(rec))
    module(clean_batch)
    for h in tmp:
        h.remove()

    sanitizers = {n: TorchOutputSanitizer.from_peak(peaks.get(n, 1.0), margin)
                  for n in target_names}
    return ProtectedModel(module, guard, sanitizers)

"""
quickstart.py -- the 30-second demo: inject a fault, watch radshield catch it.
Run:  python examples/quickstart.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radshield as rs

rng = np.random.default_rng(0)

# A pretend weight tensor and a pretend forward pass.
params = {"W": rng.normal(0, 0.1, size=(256, 256)).astype(np.float32)}
x = rng.normal(0, 1, size=(8, 256)).astype(np.float32)
forward = lambda p: x @ p["W"]

clean = forward(params)
guard = rs.WeightGuard(params)
sanitize = rs.OutputSanitizer.from_calibration(clean, margin=4.0)

# A cosmic ray strikes the worst place: the top float32 exponent bit (bit 30).
params["W"] = rs.inject.inject_bit_flips(params["W"], n_flips=1, bit=30, rng=rng)

corrupted = forward(params)
print(f"after upset, output max magnitude : {np.abs(corrupted).max():.3e}  "
      f"finite={np.all(np.isfinite(corrupted))}")

repaired = guard.verify_and_repair(params)
safe = sanitize(forward(params))
print(f"tensors repaired by WeightGuard   : {repaired}")
print(f"after radshield, max magnitude    : {np.abs(safe).max():.3e}  "
      f"finite={np.all(np.isfinite(safe))}")
print(f"matches clean output              : {np.allclose(safe, clean)}")

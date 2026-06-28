"""
radshield -- fault-tolerant inference runtime for compute in harsh environments.

"ECC for your model." Detects and repairs radiation-induced bit flips in model
weights and clamps corrupted activations, so models keep producing correct
output under single-event upsets -- in orbit today, in safety-critical edge
deployments right now.

Quick start
-----------
    import numpy as np
    import radshield as rs

    guard = rs.WeightGuard(params)               # checksums + golden copy
    sanitize = rs.OutputSanitizer.from_calibration(clean_logits, margin=4.0)

    # ... a cosmic ray flips a bit in params['W1'] ...
    guard.verify_and_repair(params)              # detected + restored
    safe = sanitize(model_forward(x, params))    # explosion contained
"""

from .guard import WeightGuard, OutputSanitizer
from . import inject

__all__ = ["WeightGuard", "OutputSanitizer", "inject"]
__version__ = "0.0.1"

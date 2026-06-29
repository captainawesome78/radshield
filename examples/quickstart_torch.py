"""
examples/quickstart_torch.py -- protect a real torch model in one call.
Run:  python examples/quickstart_torch.py
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radshield.torch as rst

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4)).eval()
x = torch.randn(8, 64)
clean = model(x)

# One call: checksum weights, calibrate output ranges on a clean batch.
handle = rst.protect(model, clean_batch=x)

# A cosmic ray flips the worst bit (top exponent) somewhere in a weight tensor.
W = model[0].weight
with torch.no_grad():
    W.data.copy_(rst.inject_bit_flips(W.data, n_flips=1, bit=30))

# Forward pass: the pre-hook detects + repairs the weight, output stays sane.
y = model(x)
print(f"upsets repaired by radshield : {handle.guard.repairs}")
print(f"output finite                : {bool(torch.all(torch.isfinite(y)))}")
print(f"matches clean output         : {bool(torch.allclose(y, clean))}")

handle.remove()

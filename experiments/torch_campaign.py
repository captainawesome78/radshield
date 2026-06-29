"""
experiments/torch_campaign.py
=============================

Reproduces Study B (accumulated-upset sweep) on a real PyTorch CNN using the
one-call ``radshield.torch.protect`` adapter. Synthetic data, no downloads,
runs on CPU in well under a minute.

Shows: as bit-flip upsets accumulate in the model's parameters, an unprotected
CNN fails catastrophically at a rising rate, while the radshield-protected model
holds catastrophic failures at ~0%.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radshield.torch as rst

torch.manual_seed(0)
np.random.seed(0)
g = torch.Generator().manual_seed(0)


# ----- synthetic image-like data ------------------------------------------- #
def make_data(n=1200, k=4, size=8):
    protos = torch.randn(k, 1, size, size)
    X = torch.empty(n, 1, size, size)
    y = torch.empty(n, dtype=torch.long)
    for i in range(n):
        c = np.random.randint(k)
        X[i] = protos[c] + 0.6 * torch.randn(1, size, size)
        y[i] = c
    return X, y, k


# ----- a real CNN ----------------------------------------------------------- #
class CNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, k)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.c1(x))
        x = self.r(self.c2(x))
        x = x.flatten(1)
        x = self.r(self.fc1(x))
        return self.fc2(x)


def accuracy(model, X, y):
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean().item())


def is_catastrophic(logits, acc, base):
    if not torch.all(torch.isfinite(logits)):
        return True
    return acc < base - 0.20


def inject_into_params(model, n_flips):
    names = [n for n, p in model.named_parameters() if p.dtype == torch.float32]
    with torch.no_grad():
        for _ in range(n_flips):
            name = names[torch.randint(0, len(names), (1,), generator=g).item()]
            p = dict(model.named_parameters())[name]
            p.data.copy_(rst.inject_bit_flips(p.data, n_flips=1, generator=g))


def campaign(golden_state, Xte, yte, clean_batch, base, n_flips,
             trials=150, protected=False):
    model = CNN(K)
    catastrophic = 0
    for _ in range(trials):
        model.load_state_dict(golden_state)
        model.eval()
        handle = None
        if protected:
            handle = rst.protect(model, clean_batch)   # snapshot clean + calibrate
        inject_into_params(model, n_flips)             # upsets strike
        with torch.no_grad():
            logits = model(Xte)                        # pre-hook repairs if protected
        acc = float((logits.argmax(1) == yte).float().mean().item())
        if is_catastrophic(logits, acc, base):
            catastrophic += 1
        if handle:
            handle.remove()
    return catastrophic / trials


# --------------------------------------------------------------------------- #
X, y, K = make_data()
Xtr, ytr, Xte, yte = X[:900], y[:900], X[900:], y[900:]

model = CNN(K)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
lossf = nn.CrossEntropyLoss()
for _ in range(300):
    opt.zero_grad()
    lossf(model(Xtr), ytr).backward()
    opt.step()

base = accuracy(model, Xte, yte)
print(f"clean test accuracy: {base:.3f}")

golden_state = copy.deepcopy(model.state_dict())
clean_batch = Xte[:64]

flip_counts = [1, 5, 10, 20, 40, 80]
unprot, prot = [], []
print("\naccumulated-upset sweep (real CNN, via radshield.torch.protect):")
for nf in flip_counts:
    u = campaign(golden_state, Xte, yte, clean_batch, base, nf, protected=False)
    p = campaign(golden_state, Xte, yte, clean_batch, base, nf, protected=True)
    unprot.append(u * 100); prot.append(p * 100)
    print(f"  {nf:3d} upsets : unprotected {u*100:5.1f}%   radshield {p*100:5.1f}%")

# ----- chart -----
plt.figure(figsize=(7, 4.8))
plt.plot(flip_counts, unprot, "o-", color="#d62728", lw=2.5, ms=7, label="unprotected")
plt.plot(flip_counts, prot, "o-", color="#2ca02c", lw=2.5, ms=7, label="radshield.torch")
plt.title("radshield on a real PyTorch CNN\ncatastrophic failure vs accumulated bit-flip upsets")
plt.xlabel("accumulated single-bit upsets in model parameters")
plt.ylabel("catastrophic outputs (%)")
plt.ylim(-3, 103)
plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_torch.png")
plt.savefig(out, dpi=130)
print(f"\nchart saved -> {out}")

"""
experiments/fault_campaign.py
=============================

The headline experiment. Trains a small MLP classifier, then runs two studies:

  STUDY A -- Per-bit blast radius:
      Flip one bit at a fixed weight, sweep the bit position 0..31, and measure
      accuracy loss. Shows empirically that high exponent bits are catastrophic
      and mantissa bits are nearly harmless -- the core physics the runtime
      exploits.

  STUDY B -- Protected vs unprotected campaign:
      Inject random single-bit flips into the weights many times. Measure how
      often the model suffers a CATASTROPHIC failure (NaN/Inf output or accuracy
      collapse) with no protection vs with radshield (WeightGuard +
      OutputSanitizer). This is the chart that goes in the README.

Pure numpy, runs on a laptop in seconds, no downloads.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radshield as rs

SEED = 7
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# A tiny but real MLP classifier (numpy, trained with SGD)                     #
# --------------------------------------------------------------------------- #
def make_blobs(n=1500, k=4, dim=8):
    centers = rng.normal(0, 5, size=(k, dim)).astype(np.float32)
    X = np.empty((n, dim), np.float32)
    y = np.empty(n, np.int64)
    for i in range(n):
        c = rng.integers(0, k)
        X[i] = centers[c] + rng.normal(0, 1.6, size=dim).astype(np.float32)
        y[i] = c
    return X, y, k


def init_params(dim, hidden, k):
    def he(a, b):
        return (rng.normal(0, np.sqrt(2.0 / a), size=(a, b))).astype(np.float32)
    return {
        "W1": he(dim, hidden), "b1": np.zeros(hidden, np.float32),
        "W2": he(hidden, hidden), "b2": np.zeros(hidden, np.float32),
        "W3": he(hidden, k), "b3": np.zeros(k, np.float32),
    }


def forward(X, p, return_hidden=False):
    h1 = np.maximum(0, X @ p["W1"] + p["b1"])
    h2 = np.maximum(0, h1 @ p["W2"] + p["b2"])
    logits = h2 @ p["W3"] + p["b3"]
    if return_hidden:
        return logits, (h1, h2)
    return logits


def softmax_xent_grad(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z); sm = e / e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(sm[np.arange(n), y] + 1e-9).mean()
    g = sm.copy(); g[np.arange(n), y] -= 1; g /= n
    return loss, g


def train(X, y, p, steps=400, lr=0.2):
    for _ in range(steps):
        h1 = np.maximum(0, X @ p["W1"] + p["b1"])
        h2 = np.maximum(0, h1 @ p["W2"] + p["b2"])
        logits = h2 @ p["W3"] + p["b3"]
        _, dlog = softmax_xent_grad(logits, y)
        dW3 = h2.T @ dlog; db3 = dlog.sum(0)
        dh2 = (dlog @ p["W3"].T) * (h2 > 0)
        dW2 = h1.T @ dh2; db2 = dh2.sum(0)
        dh1 = (dh2 @ p["W2"].T) * (h1 > 0)
        dW1 = X.T @ dh1; db1 = dh1.sum(0)
        for k_, g_ in [("W1", dW1), ("b1", db1), ("W2", dW2),
                       ("b2", db2), ("W3", dW3), ("b3", db3)]:
            p[k_] -= (lr * g_).astype(np.float32)
    return p


def accuracy(X, y, p):
    return float((forward(X, p).argmax(1) == y).mean())


# --------------------------------------------------------------------------- #
# Studies                                                                     #
# --------------------------------------------------------------------------- #
def per_bit_blast_radius(Xte, yte, params, trials_per_bit=60):
    base = accuracy(Xte, yte, params)
    losses = []
    for bit in range(32):
        accs = []
        for _ in range(trials_per_bit):
            p = {k: v.copy() for k, v in params.items()}
            tgt = rng.choice(["W1", "W2", "W3"])
            p[tgt] = rs.inject.inject_bit_flips(p[tgt], n_flips=1, bit=bit, rng=rng)
            accs.append(accuracy(Xte, yte, p))
        losses.append(base - float(np.mean(accs)))
    return base, losses


def is_catastrophic(acc, logits, base_acc):
    if not np.all(np.isfinite(logits)):
        return True
    return acc < base_acc - 0.20          # >20-point collapse


def campaign(Xte, yte, params, n_flips, trials=400, protected=False):
    """Inject ``n_flips`` accumulated upsets per trial (models the bit flips that
    pile up between memory scrubs) and measure the catastrophic failure rate."""
    base = accuracy(Xte, yte, params)
    clean_logits = forward(Xte, params)
    sanitize = rs.OutputSanitizer.from_calibration(clean_logits, margin=4.0)

    catastrophic = 0
    for _ in range(trials):
        p = {k: v.copy() for k, v in params.items()}
        # accumulate upsets across random tensors
        for _ in range(n_flips):
            tgt = rng.choice(["W1", "W2", "W3"])
            p[tgt] = rs.inject.inject_bit_flips(p[tgt], n_flips=1, rng=rng)

        if protected:
            guard = rs.WeightGuard(params)
            guard.verify_and_repair(p)                 # detect + restore weights
            logits = sanitize(forward(Xte, p))         # contain any residual
        else:
            logits = forward(Xte, p)

        acc = float((logits.argmax(1) == yte).mean())
        if is_catastrophic(acc, logits, base):
            catastrophic += 1

    return catastrophic / trials


# --------------------------------------------------------------------------- #
def main():
    X, y, k = make_blobs()
    dim = X.shape[1]
    ntr = 1000
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    params = init_params(dim, 64, k)
    params = train(Xtr, ytr, params)

    base = accuracy(Xte, yte, params)
    print(f"clean test accuracy: {base:.3f}")

    print("\n[Study A] per-bit blast radius ...")
    base_a, losses = per_bit_blast_radius(Xte, yte, params)

    print("[Study B] accumulated-upset sweep ...")
    flip_counts = [1, 5, 10, 20, 40, 80]
    unprot_rates, prot_rates = [], []
    for nf in flip_counts:
        u = campaign(Xte, yte, params, n_flips=nf, protected=False)
        p_ = campaign(Xte, yte, params, n_flips=nf, protected=True)
        unprot_rates.append(u * 100)
        prot_rates.append(p_ * 100)
        print(f"  {nf:3d} upsets/trial : unprotected {u*100:5.1f}%   radshield {p_*100:5.1f}%")

    print("\n================  RESULTS  ================")
    print(f"clean accuracy                       : {base:.3f}")
    print(f"unprotected catastrophic @80 upsets  : {unprot_rates[-1]:5.1f}%")
    print(f"radshield  catastrophic @80 upsets   : {prot_rates[-1]:5.1f}%")
    print("==========================================")

    # ----- chart -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    bits = np.arange(32)
    colors = ["#444"] * 32
    colors[31] = "#e8743b"                       # sign
    for b in range(23, 31):
        colors[b] = "#d62728"                    # exponent
    ax1.bar(bits, np.array(losses) * 100, color=colors)
    ax1.set_title("Study A — per-bit blast radius\n(red = exponent, orange = sign)")
    ax1.set_xlabel("flipped bit position (float32)")
    ax1.set_ylabel("accuracy loss (points)")
    ax1.axvspan(22.5, 30.5, color="#d62728", alpha=0.07)

    ax2.plot(flip_counts, unprot_rates, "o-", color="#d62728",
             lw=2.5, ms=7, label="unprotected")
    ax2.plot(flip_counts, prot_rates, "o-", color="#2ca02c",
             lw=2.5, ms=7, label="radshield")
    ax2.set_title("Study B — failure rate vs accumulated upsets\n(bit flips between memory scrubs)")
    ax2.set_xlabel("accumulated single-bit upsets")
    ax2.set_ylabel("catastrophic outputs (%)")
    ax2.set_ylim(-3, 103)
    ax2.legend(loc="center right")
    ax2.grid(alpha=0.25)

    fig.suptitle("radshield: keeping ML inference correct under radiation-induced bit flips",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.png")
    fig.savefig(out, dpi=130)
    print(f"\nchart saved -> {out}")

    return base, unprot_rates, prot_rates


if __name__ == "__main__":
    main()

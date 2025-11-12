#!/usr/bin/env python3
"""Linear regression demo (single feature) using batch gradient descent.

Generates a synthetic house-prices dataset (house size -> price), fits a line
using gradient descent, and saves two images into `outputs/`:

- `fit.png` : scatter of data with fitted line
- `cost.png` : cost value vs iteration number

Run: python3 scripts/linear_regression_demo.py
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate_data(m=100, seed=42):
    rng = np.random.default_rng(seed)
    # x: sizes in 1000s sqft between 0.4 and 3.0
    x = rng.uniform(0.4, 3.0, size=m)
    # true relation: price (thousands) = 50 + 120 * x
    true_w = 120.0
    true_b = 50.0
    noise = rng.normal(0, 20.0, size=m)  # noise in $1000s
    y = true_w * x + true_b + noise
    return x.reshape(-1, 1), y.reshape(-1, 1), true_w, true_b


def compute_cost(x, y, w, b):
    m = x.shape[0]
    preds = x * w + b
    errs = preds - y
    return float((errs ** 2).sum() / (2 * m))


def gradient_step(x, y, w, b, lr):
    m = x.shape[0]
    preds = x * w + b
    errs = preds - y
    # derivatives
    dw = (errs * x).sum() / m
    db = errs.sum() / m
    w = w - lr * dw
    b = b - lr * db
    return w, b, float(dw), float(db)


def train(x, y, w0=0.0, b0=0.0, lr=0.01, iters=2000, verbose=True):
    w = float(w0)
    b = float(b0)
    history = []
    for it in range(1, iters + 1):
        w, b, dw, db = gradient_step(x, y, w, b, lr)
        cost = compute_cost(x, y, w, b)
        history.append(cost)
        if verbose and (it % max(1, iters // 10) == 0):
            print(f"iter {it:4d} | cost={cost:8.3f} | w={w:7.3f} | b={b:7.3f}")
    return w, b, history


def plot_fit(x, y, w, b, out_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="data", alpha=0.6)
    xs = np.array([x.min() - 0.1, x.max() + 0.1])
    ys = w * xs + b
    plt.plot(xs, ys, color="red", label=f"fit: y={w:.2f}x+{b:.2f}")
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price (thousands $)")
    plt.title("House Prices: Data and Linear Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cost(history, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)")
    plt.title("Cost vs Iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, true_w, true_b = generate_data(m=100)
    print(f"Generated {x.shape[0]} examples. True w={true_w}, b={true_b}")

    # Hyperparameters (students can experiment)
    lr = 0.05
    iters = 500

    w0, b0 = 0.0, 0.0
    w, b, history = train(x, y, w0=w0, b0=b0, lr=lr, iters=iters)

    print("\nFinal parameters:")
    print(f"w = {w:.4f}, b = {b:.4f}")
    print(f"Final cost: {history[-1]:.4f}")

    fit_path = out_dir / "fit.png"
    cost_path = out_dir / "cost.png"
    plot_fit(x, y, w, b, fit_path)
    plot_cost(history, cost_path)

    print(f"Saved fit plot to: {fit_path}")
    print(f"Saved cost plot to: {cost_path}")


if __name__ == "__main__":
    main()

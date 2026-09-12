from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acquisition import acquire, dense_angles, random_sparse_angles, uniform_sparse_angles
from src.metrics import hidden_feature_iou, mse, ssim
from src.phantom import make_heritage_phantom
from src.reconstruction import reconstruct
from src.strategies import adaptive_angle_indices


def evaluate(name, budget_pct, idx, sino, angles, phantom, hidden_mask, elapsed):
    rec = reconstruct(sino[:, idx], angles[idx], phantom.shape[0])
    return rec, {
        "strategy": name,
        "budget_pct": budget_pct,
        "angles": int(len(idx)),
        "ray_count": int(sino.shape[0] * len(idx)),
        "mse": mse(phantom, rec),
        "ssim": ssim(phantom, rec),
        "hidden_iou": hidden_feature_iou(phantom, rec, hidden_mask),
        "runtime_s": float(elapsed),
    }


def main():
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    phantom, hidden_mask = make_heritage_phantom(size=160, seed=7)
    angles = dense_angles(180)
    sino = acquire(phantom, angles)

    budgets = [5, 10, 20, 40, 100]
    rng = np.random.default_rng(2026)
    rows = []
    recon_cache = {}

    for pct in budgets:
        n = max(6, int(round(len(angles) * pct / 100)))
        n = min(n, len(angles))

        # Uniform sparse.
        t0 = time.perf_counter()
        idx = uniform_sparse_angles(angles, n)
        rec, row = evaluate("uniform_sparse", pct, idx, sino, angles, phantom, hidden_mask, time.perf_counter() - t0)
        rows.append(row)
        recon_cache[("uniform_sparse", pct)] = rec

        # Random sparse.
        t0 = time.perf_counter()
        idx = random_sparse_angles(angles, n, rng)
        rec, row = evaluate("random_sparse", pct, idx, sino, angles, phantom, hidden_mask, time.perf_counter() - t0)
        rows.append(row)
        recon_cache[("random_sparse", pct)] = rec

        # Adaptive sparse.
        t0 = time.perf_counter()
        idx = adaptive_angle_indices(sino, angles, n, phantom.shape[0], seed_count=min(6, n))
        selection_time = time.perf_counter() - t0
        rec, row = evaluate("adaptive", pct, idx, sino, angles, phantom, hidden_mask, selection_time)
        rows.append(row)
        recon_cache[("adaptive", pct)] = rec

    # Dense reference row is reported separately, even though 100% sparse methods converge to it.
    dense_idx = np.arange(len(angles))
    t0 = time.perf_counter()
    dense_rec, dense_row = evaluate(
        "dense_reference", 100, dense_idx, sino, angles, phantom, hidden_mask,
        time.perf_counter() - t0,
    )
    rows.append(dense_row)
    recon_cache[("dense_reference", 100)] = dense_rec

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "metrics.csv", index=False)

    summary = {
        "phantom_size": list(phantom.shape),
        "dense_angles": len(angles),
        "detector_bins": int(sino.shape[0]),
        "budgets_pct": budgets,
        "strategies": ["uniform_sparse", "random_sparse", "adaptive", "dense_reference"],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: phantom + 20% reconstructions.
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    panels = [
        ("Ground truth", phantom),
        ("Dense reference", dense_rec),
        ("Uniform 20%", recon_cache[("uniform_sparse", 20)]),
        ("Random 20%", recon_cache[("random_sparse", 20)]),
        ("Adaptive 20%", recon_cache[("adaptive", 20)]),
    ]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "reconstruction_comparison.png", dpi=180)
    plt.close(fig)

    # Figure 2: primary hidden-feature recovery curve.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sparse_df = df[df["strategy"] != "dense_reference"]
    for strategy in ["uniform_sparse", "random_sparse", "adaptive"]:
        d = sparse_df[sparse_df["strategy"] == strategy].sort_values("ray_count")
        ax.plot(d["ray_count"], d["hidden_iou"], marker="o", label=strategy)
    ax.set_xlabel("Acquired rays (detector bins × angles)")
    ax.set_ylabel("Hidden-feature IoU")
    ax.set_title("Hidden-structure recovery vs acquisition budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "hidden_recovery_vs_ray_budget.png", dpi=180)
    plt.close(fig)

    # Figure 3: SSIM vs rays.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for strategy in ["uniform_sparse", "random_sparse", "adaptive"]:
        d = sparse_df[sparse_df["strategy"] == strategy].sort_values("ray_count")
        ax.plot(d["ray_count"], d["ssim"], marker="o", label=strategy)
    ax.set_xlabel("Acquired rays (detector bins × angles)")
    ax.set_ylabel("SSIM")
    ax.set_title("Reconstruction quality vs acquisition budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "ssim_vs_ray_budget.png", dpi=180)
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nWrote: {results_dir / 'metrics.csv'}")
    print(f"Wrote: {figures_dir / 'reconstruction_comparison.png'}")
    print(f"Wrote: {figures_dir / 'hidden_recovery_vs_ray_budget.png'}")


if __name__ == "__main__":
    main()

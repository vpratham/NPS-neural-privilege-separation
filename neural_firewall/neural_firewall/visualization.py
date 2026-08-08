"""
visualization.py

Plotting helpers for Phase 9: activation trajectories, projection
magnitudes, layer contributions, intervention strength, risk timeline.
Each function returns a matplotlib Figure rather than calling plt.show(),
so callers (notebook cells, saved artifacts, a future dashboard) control
display/saving themselves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_risk_timeline(token_positions: list[int], risk_scores: list[float], threshold: float | None = None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(token_positions, risk_scores, marker="o", markersize=3, linewidth=1)
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="red", label=f"threshold={threshold:.2f}")
        ax.legend()
    ax.set_xlabel("generated token position")
    ax.set_ylabel("risk score")
    ax.set_title("Risk over generation time")
    fig.tight_layout()
    return fig


def plot_layer_contributions(per_layer_scores: dict[int, float], threshold_by_layer: dict[int, float] | None = None):
    layers = sorted(per_layer_scores.keys())
    scores = [per_layer_scores[l] for l in layers]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar([str(l) for l in layers], scores)
    if threshold_by_layer:
        thr = [threshold_by_layer.get(l, np.nan) for l in layers]
        ax.plot([str(l) for l in layers], thr, linestyle="--", color="red", marker="x")
    ax.set_xlabel("layer")
    ax.set_ylabel("raw probe score")
    ax.set_title("Per-layer contribution")
    fig.tight_layout()
    return fig


def plot_projection_magnitude(token_positions: list[int], projections: list[float]):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(token_positions, projections, marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("generated token position")
    ax.set_ylabel("projection onto policy direction")
    ax.set_title("Projection magnitude over generation")
    fig.tight_layout()
    return fig


def plot_intervention_strength_sweep(strengths: list[float], metric_values: list[float], metric_name: str = "recall"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strengths, metric_values, marker="o")
    ax.set_xlabel("intervention strength (alpha)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs. intervention strength")
    fig.tight_layout()
    return fig


def plot_streaming_result(result, threshold: float | None = None):
    """Convenience wrapper: build a risk timeline directly from a
    streaming.StreamingResult (Phase 4/8 output) without the caller having
    to unpack `.trajectory` themselves."""
    positions, risks = result.risk_series()
    fig = plot_risk_timeline(positions, risks, threshold=threshold)
    if result.stopped_early:
        fig.axes[0].axvline(positions[-1], color="black", linestyle=":", label=result.stop_reason)
        fig.axes[0].legend()
    return fig


def plot_activation_trajectory_2d(coords: np.ndarray, labels: list[int] | None = None):
    """coords: (n_points, 2) e.g. from PCA/UMAP of pooled activations."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if labels is not None:
        labels_arr = np.array(labels)
        for lab in sorted(set(labels)):
            mask = labels_arr == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lab), alpha=0.6, s=15)
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=15)
    ax.set_title("Activation trajectory (2D projection)")
    fig.tight_layout()
    return fig

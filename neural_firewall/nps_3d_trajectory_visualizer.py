"""
NPS 3D Activation Trajectory Visualizer
=======================================

Visualizes the full 36-layer XSTest activation trajectories as a 3D
projection.

Input:
    phase1_real/xstest/full_layer_trajectories/trajectories.npz
    phase1_real/xstest/full_layer_trajectories/metadata.json

The 2048-D activation at each layer is reduced to 3D with PCA fitted
jointly across ALL examples and layers. This means the axes represent
global directions of variance in the measured activation states.

Each prompt becomes a trajectory:
    layer 0 -> layer 1 -> ... -> layer 35

Groups:
    SAFE
    UNSAFE DETECTED
    UNSAFE MISSED

Controls:
    Space       play/pause all trajectories
    Left/Right  previous/next frame
    R           restart
    Q/Esc       quit

Command:
    python nps_3d_trajectory_visualizer.py

Optional:
    --data <path-to-trajectories.npz>
    --metadata <path-to-metadata.json>
    --components pca
    --max-points 135

IMPORTANT:
This is a visualization of the measured activation trajectory after a
3D dimensionality reduction. It is NOT a claim that the model itself
represents prompts in 3D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA


DEFAULT_DATA = (
    "phase1_real/xstest/full_layer_trajectories/trajectories.npz"
)

DEFAULT_METADATA = (
    "phase1_real/xstest/full_layer_trajectories/metadata.json"
)


def load_data(data_path: Path, metadata_path: Path):
    z = np.load(data_path)

    activations = z["activations"].astype(np.float32)
    labels = z["labels"].astype(np.int64)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    examples = metadata["examples"]

    if activations.ndim != 3:
        raise RuntimeError(
            f"Expected [examples, layers, hidden], got {activations.shape}"
        )

    n_examples, n_layers, hidden = activations.shape

    print("# NPS 3D ACTIVATION TRAJECTORIES")
    print()
    print(f"examples:    {n_examples}")
    print(f"layers:      {n_layers}")
    print(f"hidden size: {hidden}")

    return activations, labels, examples


def project_to_3d(activations):
    n_examples, n_layers, hidden = activations.shape

    # Fit PCA jointly over every measured state.
    flat = activations.reshape(-1, hidden)

    print("[PCA] fitting 3D projection...")
    pca = PCA(n_components=3, random_state=42)
    projected = pca.fit_transform(flat)

    projected = projected.reshape(n_examples, n_layers, 3)

    explained = pca.explained_variance_ratio_

    print(
        "[PCA] explained variance:",
        ", ".join(f"{x:.4f}" for x in explained),
    )
    print(
        "[PCA] cumulative:",
        f"{explained.sum():.4f}",
    )

    return projected, pca


def group_name(label, predicted):
    if int(label) == 0:
        return "SAFE"

    if int(predicted) == 1:
        return "UNSAFE DETECTED"

    return "UNSAFE MISSED"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA,
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    metadata_path = Path(args.metadata)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    activations, labels, examples = load_data(
        data_path,
        metadata_path,
    )

    if args.max_points is not None:
        activations = activations[:args.max_points]
        labels = labels[:args.max_points]
        examples = examples[:args.max_points]

    # ---------------------------------------------------------
    # 3D projection
    # ---------------------------------------------------------

    projected, pca = project_to_3d(activations)

    # Save projection so it can be reused without recomputing PCA.
    output_dir = data_path.parent

    np.savez_compressed(
        output_dir / "trajectory_3d_projection.npz",
        projection=projected,
        labels=labels,
        layers=np.arange(projected.shape[1]),
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title(
        "NPS — 3D Projection of Internal Activation Trajectories",
        fontsize=15,
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    # Group masks.
    safe = labels == 0
    unsafe = labels == 1

    predicted = np.array(
        [int(x["predicted"]) for x in examples],
        dtype=np.int64,
    )

    detected = unsafe & (predicted == 1)
    missed = unsafe & (predicted == 0)

    # Plot full trajectories with modest transparency.
    for mask, label in [
        (safe, "SAFE"),
        (detected, "UNSAFE DETECTED"),
        (missed, "UNSAFE MISSED"),
    ]:
        indices = np.where(mask)[0]

        for i in indices:
            traj = projected[i]

            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                alpha=0.16,
                linewidth=0.8,
            )

        if len(indices):
            # Legend proxy.
            ax.plot(
                [np.nan],
                [np.nan],
                [np.nan],
                linewidth=2,
                label=label,
            )

    ax.legend(loc="upper right")

    # Animated current state for every trajectory.
    current_points = ax.scatter(
        projected[:, 0, 0],
        projected[:, 0, 1],
        projected[:, 0, 2],
        s=18,
    )

    # Highlight one selected trajectory.
    selected = 0

    selected_line, = ax.plot(
        projected[selected, :, 0],
        projected[selected, :, 1],
        projected[selected, :, 2],
        linewidth=3,
    )

    selected_point = ax.scatter(
        [projected[selected, 0, 0]],
        [projected[selected, 0, 1]],
        [projected[selected, 0, 2]],
        s=100,
    )

    info = fig.text(
        0.02,
        0.02,
        "",
        fontsize=10,
    )

    frame_text = fig.text(
        0.02,
        0.055,
        "",
        fontsize=11,
    )

    frame = {"layer": 0, "selected": selected}

    def update(_):
        layer = frame["layer"]
        idx = frame["selected"]

        # Current position of every trajectory.
        current = projected[:, layer, :]

        current_points._offsets3d = (
            current[:, 0],
            current[:, 1],
            current[:, 2],
        )

        # Highlight selected trajectory.
        selected_line.set_data(
            projected[idx, :, 0],
            projected[idx, :, 1],
        )
        selected_line.set_3d_properties(
            projected[idx, :, 2]
        )

        selected_point._offsets3d = (
            [projected[idx, layer, 0]],
            [projected[idx, layer, 1]],
            [projected[idx, layer, 2]],
        )

        ex = examples[idx]

        status = group_name(
            ex["label"],
            ex["predicted"],
        )

        prompt = str(ex.get("prompt", ""))

        if len(prompt) > 180:
            prompt = prompt[:177] + "..."

        frame_text.set_text(
            f"Layer {layer}/{projected.shape[1] - 1}   "
            f"Selected: {idx + 1}/{len(examples)}   "
            f"Status: {status}"
        )

        info.set_text(
            f"ID: {ex.get('id', 'n/a')}   "
            f"Category: {ex.get('type', 'n/a')}\n"
            f"Prompt: {prompt}\n"
            f"Controls: Space play/pause | "
            f"←/→ layer | ↑/↓ selected prompt | "
            f"R restart | Q quit"
        )

        return (
            current_points,
            selected_line,
            selected_point,
        )

    playing = {"value": True}

    def on_key(event):
        if event.key == " ":
            playing["value"] = not playing["value"]

        elif event.key == "right":
            frame["layer"] = min(
                frame["layer"] + 1,
                projected.shape[1] - 1,
            )
            playing["value"] = False

        elif event.key == "left":
            frame["layer"] = max(
                frame["layer"] - 1,
                0,
            )
            playing["value"] = False

        elif event.key == "up":
            frame["selected"] = (
                frame["selected"] + 1
            ) % len(examples)
            playing["value"] = False

        elif event.key == "down":
            frame["selected"] = (
                frame["selected"] - 1
            ) % len(examples)
            playing["value"] = False

        elif event.key == "r":
            frame["layer"] = 0
            playing["value"] = True

        elif event.key in ("q", "escape"):
            plt.close(fig)
            return

        update(None)
        fig.canvas.draw_idle()

    def animate(_):
        if playing["value"]:
            frame["layer"] += 1

            if frame["layer"] >= projected.shape[1]:
                frame["layer"] = 0
                frame["selected"] = (
                    frame["selected"] + 1
                ) % len(examples)

            update(None)

        return (
            current_points,
            selected_line,
            selected_point,
        )

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    animation = FuncAnimation(
        fig,
        animate,
        interval=450,
        blit=False,
        cache_frame_data=False,
    )

    update(None)

    print()
    print("[DONE] 3D projection saved to:")
    print(output_dir / "trajectory_3d_projection.npz")

    plt.show()


if __name__ == "__main__":
    main()

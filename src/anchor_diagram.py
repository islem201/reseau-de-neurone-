"""Draw a reproducible diagram of two ground truths assigned to one cell."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from src.anchors import default_anchors, encode_targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/anchor_mechanism.png"))
    args = parser.parse_args(argv)
    boxes = torch.tensor([[21., 18., 12., 12.], [22., 19., 18., 18.]])
    anchors = default_anchors(64)
    target, dropped = encode_targets(boxes, anchors, 64)
    assert dropped == 0

    fig, (grid_ax, table_ax) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor("#f8fafc")
    grid_ax.set_facecolor("white")
    grid_ax.set_xlim(0, 64)
    grid_ax.set_ylim(64, 0)
    grid_ax.set_aspect("equal")
    grid_ax.set_xticks(range(0, 65, 8))
    grid_ax.set_yticks(range(0, 65, 8))
    grid_ax.grid(color="#cbd5e1", linewidth=0.8)
    grid_ax.add_patch(Rectangle((16, 16), 8, 8, facecolor="#fef08a", edgecolor="#a16207", lw=1.5, zorder=2))
    for box, color, label, marker in zip(boxes.tolist(), ("#2563eb", "#dc2626"), ("A", "B"), ("o", "x")):
        cx, cy, w, h = box
        grid_ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=color, lw=2, zorder=3))
        grid_ax.plot(cx, cy, marker=marker, color=color, ms=7, mew=2, zorder=4)
        grid_ax.text(cx + w / 2 + 1, cy + h / 2, label, color=color, fontsize=12, weight="bold")
    grid_ax.set_title("64×64 image: same center cell", fontsize=13, weight="bold", pad=14)
    grid_ax.set_xlabel("x pixels → column")
    grid_ax.set_ylabel("y pixels → row")
    grid_ax.text(0, 70, "Highlighted cell: row 2, column 2\nEach cell spans 8×8 pixels.", fontsize=10, va="top")

    table_ax.axis("off")
    table_ax.set_title("Four slots in that cell", fontsize=13, weight="bold", pad=14)
    rows = []
    for a, (aw, ah) in enumerate(anchors):
        obj, dx, dy, tw, th = target[a, 2, 2].tolist()
        assigned = "A" if a == 1 else "B" if a == 2 else "—"
        rows.append([f"{aw:.0f} × {ah:.0f}", assigned, f"{obj:.0f}", f"{dx:.3f}",
                     f"{dy:.3f}", f"{tw:.3f}", f"{th:.3f}"])
    table = table_ax.table(cellText=rows, colLabels=["Anchor\n(w × h)", "Object", "obj", "dx", "dy", "tw", "th"],
                           cellLoc="center", loc="upper center", bbox=[0, 0.40, 1, 0.54])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for col in range(7):
        table[(0, col)].set_facecolor("#dbeafe")
        table[(2, col)].set_facecolor("#dbeafe")  # 14×14 slot: object A
        table[(3, col)].set_facecolor("#fee2e2")  # 18×18 slot: object B
    table_ax.text(0.02, 0.32, "A = (21, 18, 12, 12)  →  anchor 14 × 14", color="#1d4ed8", fontsize=11, transform=table_ax.transAxes)
    table_ax.text(0.02, 0.26, "B = (22, 19, 18, 18)  →  anchor 18 × 18", color="#b91c1c", fontsize=11, transform=table_ax.transAxes)
    table_ax.text(0.02, 0.14, "Target: [obj, dx, dy, log(w / aw), log(h / ah)]", fontsize=10, transform=table_ax.transAxes)
    table_ax.text(0.02, 0.07, "The head predicts 4 × 5 = 20 values at each cell.", fontsize=10, transform=table_ax.transAxes)
    fig.suptitle("Why anchors help when two centers share a cell", fontsize=16, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Anchor diagram saved to {args.output}")


if __name__ == "__main__":
    main()

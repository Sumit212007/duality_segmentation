"""
visualize.py - Generate high-contrast visualisations from raw predictions
Usage:
    python visualize.py --pred_dir runs/predictions --rgb_dir dataset/testImages
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES


COLORMAP = np.array(CLASS_COLORS, dtype=np.uint8)


def colorize(pred_np):
    h, w = pred_np.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[pred_np == c] = COLORMAP[c]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="./runs/predictions")
    p.add_argument("--rgb_dir",  default="./dataset/testImages")
    p.add_argument("--out_dir",  default="./runs/visualizations")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred_files = [f for f in os.listdir(args.pred_dir)
                  if f.endswith("_pred.png")]

    if not pred_files:
        print("No prediction files found. Run test.py first.")
        return

    legend = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0,
                       label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]

    for fname in pred_files:
        stem     = fname.replace("_pred.png", "")
        pred_np  = np.array(Image.open(os.path.join(args.pred_dir, fname)))
        color_np = colorize(pred_np)

        # Try to find matching original
        orig_np  = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(args.rgb_dir, stem + ext)
            if os.path.isfile(candidate):
                orig_np = np.array(Image.open(candidate).convert("RGB"))
                break

        if orig_np is not None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax[0].imshow(orig_np);   ax[0].set_title("Original"); ax[0].axis("off")
            ax[1].imshow(color_np);  ax[1].set_title("Segmentation"); ax[1].axis("off")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(color_np); ax.set_title("Segmentation"); ax.axis("off")

        fig.legend(handles=legend, loc="lower center",
                   ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
        out_path = os.path.join(args.out_dir, f"{stem}_vis.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\nDone! Visualizations saved → {args.out_dir}")


if __name__ == "__main__":
    main()
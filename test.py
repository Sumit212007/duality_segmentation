"""
test.py  - Run inference on unseen test images and compute IoU on val set.
Usage:
    python test.py                                     # uses best_model.pth
    python test.py --checkpoint runs/checkpoints/best_model.pth
    python test.py --test_dir dataset/testImages --save_vis
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from dataset  import (TestDataset, DesertSegDataset,
                      get_val_transforms, NUM_CLASSES,
                      CLASS_NAMES, CLASS_COLORS, RAW_TO_IDX)
from model    import build_model
from metrics  import IoUMetric
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
#  ARGS
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Duality AI Segmentation - Test")
    p.add_argument("--checkpoint",  type=str,
                   default="./runs/checkpoints/best_model.pth")
    p.add_argument("--data_root",   type=str, default="./dataset")
    p.add_argument("--test_dir",    type=str, default=None,
                   help="Folder of unseen test images (overrides data_root/testImages)")
    p.add_argument("--output_dir",  type=str, default="./runs/predictions")
    p.add_argument("--save_vis",    action="store_true",
                   help="Save coloured segmentation visualisations")
    p.add_argument("--encoder",     type=str, default="resnet50")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# ─────────────────────────────────────────────
#  COLOR MAP
# ─────────────────────────────────────────────
def build_colormap():
    """Returns an array of shape (NUM_CLASSES, 3) with RGB colors."""
    return np.array(CLASS_COLORS, dtype=np.uint8)

COLORMAP = build_colormap()


def index_to_color(mask_np):
    """Convert (H,W) class index array → (H,W,3) RGB image."""
    h, w  = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask_np == c] = COLORMAP[c]
    return color


def make_legend():
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0,
                       label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    return patches


# ─────────────────────────────────────────────
#  VISUALISE SIDE-BY-SIDE
# ─────────────────────────────────────────────
def save_visualization(original_np, pred_mask, save_path, title="Prediction"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image"); axes[0].axis("off")

    axes[1].imshow(index_to_color(pred_mask))
    axes[1].set_title("Predicted Segmentation"); axes[1].axis("off")

    fig.legend(handles=make_legend(), loc="lower center",
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model(checkpoint_path, encoder, device):
    model = build_model(NUM_CLASSES, encoder, pretrained=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch    = ckpt.get("epoch", "?")
    best_iou = ckpt.get("best_miou", "?")
    print(f"[Checkpoint] Loaded epoch={epoch}  best_miou={best_iou}")
    model.eval()
    return model


# ─────────────────────────────────────────────
#  VALIDATE on val split (compute IoU)
# ─────────────────────────────────────────────
@torch.no_grad()
def run_validation(model, data_root, batch_size, num_workers, device):
    print("\n[Validation] Computing IoU on val split ...")
    val_ds  = DesertSegDataset(data_root, "val", get_val_transforms())
    loader  = DataLoader(val_ds, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers,
                         pin_memory=True)
    metric  = IoUMetric(NUM_CLASSES)

    for images, masks in tqdm(loader, desc="  Validating"):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
        preds = logits.argmax(dim=1)
        metric.update(preds, masks)

    results = metric.print_report()
    return results


# ─────────────────────────────────────────────
#  INFERENCE on test images
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, test_dir, output_dir, batch_size,
                  num_workers, device, save_vis):
    print(f"\n[Inference] Running on: {test_dir}")
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    transform = get_val_transforms()
    ds        = TestDataset(test_dir, transform)
    loader    = DataLoader(ds, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    for images, names in tqdm(loader, desc="  Predicting"):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()      # (B, H, W)

        for pred, name in zip(preds, names):
            # Save raw prediction mask (grayscale class indices)
            stem       = os.path.splitext(name)[0]
            raw_path   = os.path.join(output_dir, f"{stem}_pred.png")
            Image.fromarray(pred.astype(np.uint8)).save(raw_path)

            # Save coloured visualisation
            if save_vis:
                # Re-load original for side-by-side
                orig_path = os.path.join(test_dir, name)
                orig_np   = np.array(Image.open(orig_path).convert("RGB"))
                vis_path  = os.path.join(vis_dir, f"{stem}_vis.png")
                save_visualization(orig_np, pred, vis_path, title=name)

    print(f"\n[Done] Predictions saved → {output_dir}")
    if save_vis:
        print(f"[Done] Visualizations  → {vis_dir}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test] Device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train first with:  python train.py"
        )

    model = load_model(args.checkpoint, args.encoder, device)

    # 1. Validate on val split
    run_validation(model, args.data_root,
                   args.batch_size, args.num_workers, device)

    # 2. Inference on test images
    test_dir = args.test_dir or os.path.join(args.data_root, "testImages")
    if os.path.isdir(test_dir):
        run_inference(model, test_dir, args.output_dir,
                      args.batch_size, args.num_workers, device,
                      save_vis=args.save_vis)
    else:
        print(f"[Warning] Test dir not found: {test_dir}  (skipping inference)")


if __name__ == "__main__":
    main()
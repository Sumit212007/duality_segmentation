"""
metrics.py - IoU and other evaluation metrics
"""

import torch
import numpy as np
from dataset import NUM_CLASSES, CLASS_NAMES


class IoUMetric:
    """
    Computes per-class and mean IoU (Intersection over Union).
    IoU = TP / (TP + FP + FN)
    """

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.reset()

    def reset(self):
        # confusion matrix: rows=ground truth, cols=predicted
        self.confusion = np.zeros((self.num_classes, self.num_classes),
                                  dtype=np.int64)

    def update(self, preds, targets):
        """
        preds   : torch.Tensor (B, H, W) – class indices
        targets : torch.Tensor (B, H, W) – class indices
        """
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # remove ignored pixels
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # accumulate
        for t, p in zip(targets, preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion[t, p] += 1

    def compute(self):
        """Returns dict with per-class IoU and mIoU."""
        iou_per_class = {}
        iou_values    = []

        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp
            denom = tp + fp + fn

            if denom == 0:
                iou = float("nan")
            else:
                iou = tp / denom

            iou_per_class[CLASS_NAMES[c]] = iou
            if not np.isnan(iou):
                iou_values.append(iou)

        miou = np.mean(iou_values) if iou_values else 0.0
        return {"miou": miou, "per_class": iou_per_class}

    def print_report(self):
        results = self.compute()
        print("\n" + "="*50)
        print(f"  mIoU : {results['miou']:.4f}  ({results['miou']*100:.2f}%)")
        print("="*50)
        for name, iou in results["per_class"].items():
            bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
            val = f"{iou:.4f}" if not np.isnan(iou) else "  N/A "
            print(f"  {name:<20} {val}  {bar}")
        print("="*50 + "\n")
        return results
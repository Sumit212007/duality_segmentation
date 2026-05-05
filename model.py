"""
model.py - DeepLabV3+ with ResNet-50 backbone for Offroad Segmentation
Uses segmentation_models_pytorch (smp) library.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from dataset import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, encoder="resnet50", pretrained=True):
    """
    Build DeepLabV3+ model.

    Args:
        num_classes  : number of segmentation classes (10 for this challenge)
        encoder      : backbone encoder name
        pretrained   : use ImageNet pretrained weights

    Returns:
        model (nn.Module)
    """
    weights = "imagenet" if pretrained else None

    model = smp.DeepLabV3Plus(
        encoder_name        = encoder,
        encoder_weights     = weights,
        in_channels         = 3,
        classes             = num_classes,
        activation          = None,          # raw logits; loss fn handles softmax
    )

    print(f"[Model] DeepLabV3+ | backbone={encoder} | classes={num_classes} "
          f"| pretrained={pretrained}")
    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total/1e6:.2f}M  |  "
          f"Trainable: {trainable/1e6:.2f}M")
    return trainable


# ─────────────────────────────────────────────
#  LOSS FUNCTION
#  Combines Cross-Entropy + Dice for better
#  handling of class imbalance
# ─────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ce_weight=0.6, dice_weight=0.4):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce   = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
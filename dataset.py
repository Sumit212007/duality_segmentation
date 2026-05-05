"""
dataset.py - Duality AI Offroad Segmentation Dataset Loader
Handles loading of RGB images and segmentation masks.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────
#  CLASS MAPPING  (raw pixel ID → 0-based index)
# ─────────────────────────────────────────────
CLASS_INFO = {
    # raw_id : (index, name, color_for_viz)
    100:  (0,  "Trees",          (34,  139, 34)),
    200:  (1,  "Lush Bushes",    (0,   200, 0)),
    300:  (2,  "Dry Grass",      (210, 180, 140)),
    500:  (3,  "Dry Bushes",     (160, 120, 60)),
    550:  (4,  "Ground Clutter", (128, 64,  0)),
    600:  (5,  "Flowers",        (255, 105, 180)),
    700:  (6,  "Logs",           (139, 69,  19)),
    800:  (7,  "Rocks",          (169, 169, 169)),
    7100: (8,  "Landscape",      (194, 178, 128)),
    10000:(9,  "Sky",            (135, 206, 235)),
}

NUM_CLASSES   = len(CLASS_INFO)           # 10
CLASS_NAMES   = [v[1] for v in sorted(CLASS_INFO.values(), key=lambda x: x[0])]
CLASS_COLORS  = [v[2] for v in sorted(CLASS_INFO.values(), key=lambda x: x[0])]

# Build lookup table (raw pixel value → class index)
# Raw IDs can be up to 10000, use a dict for safety
RAW_TO_IDX = {raw: info[0] for raw, info in CLASS_INFO.items()}


def mask_to_index(mask_pil):
    """Convert a raw segmentation mask PIL image to a 0-based class index array."""
    mask_np = np.array(mask_pil)

    # Masks may be RGB (all channels equal) or single-channel
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]          # take R channel

    # Handle 16-bit masks (some Falcon exports)
    mask_np = mask_np.astype(np.int32)

    # Remap raw IDs → class indices (unknown pixels → Landscape=8)
    out = np.full(mask_np.shape, fill_value=8, dtype=np.int64)
    for raw_id, cls_idx in RAW_TO_IDX.items():
        out[mask_np == raw_id] = cls_idx

    return out


# ─────────────────────────────────────────────
#  AUGMENTATION PIPELINES
# ─────────────────────────────────────────────
IMAGE_SIZE = (512, 512)          # resize all images to this

def get_train_transforms():
    return A.Compose([
        A.Resize(*IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GridDistortion(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(*IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────
class DesertSegDataset(Dataset):
    """
    Expects folder structure:
        dataset/
            train/
                rgb/        ← colour images  (.png / .jpg)
                seg/        ← mask images    (.png)
            val/
                rgb/
                seg/
            testImages/     ← unseen test images (no masks)
    """

    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.split     = split
        self.transform = transform

        rgb_dir = os.path.join(root, split, "rgb")
        seg_dir = os.path.join(root, split, "seg")

        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError(f"Seg folder not found: {seg_dir}")

        # Match rgb files to seg files by stem name
        rgb_files = sorted([f for f in os.listdir(rgb_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        seg_files = sorted([f for f in os.listdir(seg_dir)
                            if f.lower().endswith(".png")])

        # Build pairs (assume same base names, possibly different extensions)
        rgb_stems = {os.path.splitext(f)[0]: f for f in rgb_files}
        seg_stems = {os.path.splitext(f)[0]: f for f in seg_files}
        common    = sorted(set(rgb_stems) & set(seg_stems))

        if len(common) == 0:
            raise RuntimeError(f"No matching rgb/seg pairs found in {root}/{split}")

        self.pairs = [(os.path.join(rgb_dir, rgb_stems[s]),
                       os.path.join(seg_dir, seg_stems[s]))
                      for s in common]

        print(f"[Dataset] {split}: {len(self.pairs)} image pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, seg_path = self.pairs[idx]

        image = np.array(Image.open(rgb_path).convert("RGB"))
        mask  = mask_to_index(Image.open(seg_path))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]          # Tensor C×H×W
            mask  = transformed["mask"].long()    # Tensor H×W
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask


# ─────────────────────────────────────────────
#  TEST DATASET (no masks)
# ─────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.files = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"[Dataset] test: {len(self.files)} images found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        image = np.array(Image.open(path).convert("RGB"))
        name  = os.path.basename(path)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, name
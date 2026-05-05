"""
train.py - Full training pipeline for Duality AI Offroad Segmentation
Usage:  python train.py
        python train.py --data_root ./dataset --epochs 50 --lr 1e-4
"""

import os
import argparse
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset  import DesertSegDataset, get_train_transforms, get_val_transforms, NUM_CLASSES
from model    import build_model, CombinedLoss, count_parameters
from metrics  import IoUMetric


# ─────────────────────────────────────────────
#  ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Duality AI Segmentation Trainer")
    p.add_argument("--data_root",  type=str,   default="./dataset",
                   help="Root folder of the dataset")
    p.add_argument("--output_dir", type=str,   default="./runs",
                   help="Where to save checkpoints and logs")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=4,
                   help="Reduce to 2 if GPU OOM")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--encoder",    type=str,   default="resnet50",
                   help="Backbone: resnet50 | resnet101 | efficientnet-b4")
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_pretrain",action="store_true",
                   help="Disable ImageNet pretrained weights")
    return p.parse_args()


# ─────────────────────────────────────────────
#  TRAINING / VALIDATION LOOPS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():          # mixed precision
            logits = model(images)
            loss   = criterion(logits, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, iou_metric):
    model.eval()
    iou_metric.reset()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        from contextlib import nullcontext

        use_cuda = torch.cuda.is_available()
        autocast = torch.amp.autocast('cuda') if use_cuda else nullcontext()

        with autocast:
            logits = model(images)
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        iou_metric.update(preds, masks)

    avg_loss = total_loss / len(loader)
    results  = iou_metric.compute()
    return avg_loss, results["miou"], results


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────
def save_loss_plot(train_losses, val_losses, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue")
    plt.plot(val_losses,   label="Val Loss",   color="tomato")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Plot] Loss graph saved → {out_path}")


def save_iou_plot(miou_history, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(miou_history, label="mIoU", color="seagreen", marker="o", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("mIoU")
    plt.title("Validation mIoU over Training")
    plt.legend(); plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Plot] mIoU graph saved → {out_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Duality AI - Offroad Segmentation Training")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Encoder    : {args.encoder}")
    print(f"{'='*60}\n")

    # ── Output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Datasets & Loaders
    train_ds = DesertSegDataset(args.data_root, "train", get_train_transforms())
    val_ds   = DesertSegDataset(args.data_root, "val",   get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model
    model = build_model(NUM_CLASSES, args.encoder,
                        pretrained=not args.no_pretrain).to(device)
    count_parameters(model)

    # ── Loss, Optimiser, Scheduler, Scaler
    criterion = CombinedLoss(NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.epochs,
                                                     eta_min=1e-6)
    use_cuda = torch.cuda.is_available()

    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    iou_metric = IoUMetric(NUM_CLASSES)

    # ── Resume from checkpoint
    start_epoch = 0
    best_miou   = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"[Resume] Resuming from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    # ── TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    # ── History
    train_losses, val_losses, miou_history = [], [], []
    history_log = []

    # ─────────────────────────────────────────
    #  TRAINING LOOP
    # ─────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\nEpoch [{epoch+1}/{args.epochs}]  LR={scheduler.get_last_lr()[0]:.2e}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, scaler)
        # Validate
        val_loss, miou, results = validate(model, val_loader, criterion,
                                           device, iou_metric)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  mIoU       : {miou:.4f}  ({miou*100:.2f}%)")
        print(f"  Time       : {elapsed:.1f}s")

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("mIoU/val", miou, epoch)

        # History
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        miou_history.append(miou)
        history_log.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss, 5),
            "miou":       round(miou, 5),
        })

        # Save plots every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            save_loss_plot(train_losses, val_losses,
                           os.path.join(args.output_dir, "loss_curve.png"))
            save_iou_plot(miou_history,
                          os.path.join(args.output_dir, "miou_curve.png"))

        # Save best checkpoint
        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "args":      vars(args),
            }, best_path)
            print(f"  ✔ Best model saved → {best_path}  (mIoU={best_miou:.4f})")

        # Save latest checkpoint
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou": best_miou,
        }, os.path.join(ckpt_dir, "latest.pth"))

    # ── Save history JSON
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\n[Done] Training history saved → {history_path}")

    # ── Final per-class IoU report
    iou_metric.print_report()

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete!  Best mIoU = {best_miou:.4f} ({best_miou*100:.2f}%)")
    print(f"  Checkpoints  → {ckpt_dir}")
    print(f"  TensorBoard  → tensorboard --logdir {args.output_dir}/tb_logs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
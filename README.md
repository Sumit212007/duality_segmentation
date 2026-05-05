# Duality AI – Offroad Semantic Scene Segmentation
### DeepLabV3+ | Windows + NVIDIA GPU | 10 Desert Classes

---

## 📁 Project Structure

```
duality_segmentation/
│
├── ENV_SETUP/
│   └── setup_env.bat        ← Run this FIRST to install everything
│
├── dataset/                 ← Put your downloaded dataset here
│   ├── train/
│   │   ├── rgb/             ← Training colour images
│   │   └── seg/             ← Training segmentation masks
│   ├── val/
│   │   ├── rgb/
│   │   └── seg/
│   └── testImages/          ← Unseen test images (no masks)
│
├── dataset.py               ← Dataset loader + class definitions
├── model.py                 ← DeepLabV3+ model + loss function
├── metrics.py               ← IoU metric computation
├── train.py                 ← MAIN TRAINING SCRIPT
├── test.py                  ← MAIN TEST / INFERENCE SCRIPT
├── visualize.py             ← Generate coloured visualisations
└── README.md
```

---

## ⚡ Step-by-Step Setup

### Step 1 – Install Anaconda
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) for Windows.

---

### Step 2 – Setup Environment (ONE TIME ONLY)

Open **Anaconda Prompt** (search for it in Start Menu), then navigate to your project folder:

```bat
cd C:\path\to\duality_segmentation\ENV_SETUP
setup_env.bat
```

This installs:
- PyTorch + CUDA 11.8
- segmentation_models_pytorch
- albumentations
- opencv, matplotlib, tqdm, tensorboard

> ⚠️ If your CUDA version is different, edit `setup_env.bat` and change `pytorch-cuda=11.8`  
> Check your version with: `nvidia-smi`

---

### Step 3 – Download & Place Dataset

1. Go to: https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert
2. Create a free Falcon account if needed
3. Navigate to the **"Segmentation Track"** section
4. Download the dataset
5. Extract it so your folder looks like:

```
duality_segmentation/
└── dataset/
    ├── train/
    │   ├── rgb/      ← .png or .jpg images
    │   └── seg/      ← .png mask files
    ├── val/
    │   ├── rgb/
    │   └── seg/
    └── testImages/   ← unseen test images
```

---

### Step 4 – Activate Environment

Every time you open a new terminal, run:

```bat
conda activate EDU
```

---

## 🚀 Training

### Basic Training (Recommended to start)

```bat
conda activate EDU
cd C:\path\to\duality_segmentation
python train.py
```

This will:
- Train for **50 epochs** with batch size 4
- Save best model → `runs/checkpoints/best_model.pth`
- Save loss & mIoU graphs → `runs/`
- Log to TensorBoard → `runs/tb_logs/`

---

### Advanced Training Options

```bat
# Change epochs and learning rate
python train.py --epochs 80 --lr 5e-5

# Use stronger backbone (needs more VRAM)
python train.py --encoder resnet101

# Smaller batch if you get "CUDA out of memory"
python train.py --batch_size 2

# Resume interrupted training
python train.py --resume runs/checkpoints/latest.pth

# All options together
python train.py --epochs 60 --batch_size 4 --lr 1e-4 --encoder resnet50
```

---

### Watch Training in Real-Time (TensorBoard)

Open a **second** Anaconda Prompt window and run:

```bat
conda activate EDU
tensorboard --logdir runs/tb_logs
```
Then open your browser at: **http://localhost:6006**

---

## 🧪 Testing & Evaluation

### Run on Validation + Test Images

```bat
python test.py
```

This will:
1. Compute **IoU scores** on the validation set
2. Run inference on all images in `dataset/testImages/`
3. Save predicted masks → `runs/predictions/`

---

### Save Colour Visualisations

```bat
python test.py --save_vis
```

Side-by-side comparisons (original vs coloured segmentation) will be saved to:
`runs/predictions/visualizations/`

---

### Generate Visualisations from Saved Predictions

```bat
python visualize.py
```

---

## 📊 Understanding the Output

### During Training, you will see:

```
Epoch [1/50]  LR=1.00e-04
  Train Loss : 0.8432
  Val Loss   : 0.9102
  mIoU       : 0.3218  (32.18%)
  ✔ Best model saved → runs/checkpoints/best_model.pth
```

### IoU Report (printed after test.py):

```
==================================================
  mIoU : 0.6543  (65.43%)
==================================================
  Trees                0.7821  ████████████████
  Lush Bushes          0.6102  ████████████
  Dry Grass            0.5984  ████████████
  Rocks                0.4231  ████████
  Sky                  0.9201  ██████████████████
==================================================
```

### Files generated:

| File | Description |
|------|-------------|
| `runs/checkpoints/best_model.pth` | Best model weights |
| `runs/checkpoints/latest.pth` | Latest checkpoint |
| `runs/loss_curve.png` | Train vs Val loss graph |
| `runs/miou_curve.png` | mIoU over epochs |
| `runs/training_history.json` | All metrics per epoch |
| `runs/predictions/*.png` | Raw predicted masks |
| `runs/predictions/visualizations/*.png` | Coloured overlays |

---

## 🎯 Segmentation Classes

| ID | Class | Description |
|----|-------|-------------|
| 100 | Trees | Desert trees (e.g. Joshua trees) |
| 200 | Lush Bushes | Green/lush vegetation |
| 300 | Dry Grass | Dried grasses |
| 500 | Dry Bushes | Dried shrubs |
| 550 | Ground Clutter | Rocks, debris on ground |
| 600 | Flowers | Desert flowers |
| 700 | Logs | Fallen logs / wood |
| 800 | Rocks | Rocks and boulders |
| 7100 | Landscape | General ground (sand, dirt) |
| 10000 | Sky | Sky area |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Add `--batch_size 2` to train.py command |
| `No matching rgb/seg pairs found` | Check folder names: must be `rgb/` and `seg/` |
| Training very slow | Reduce `--num_workers 2`, check GPU with `nvidia-smi` |
| Low mIoU on rare classes | Try `--epochs 80`, the model needs more time |
| `ModuleNotFoundError` | Make sure you ran `conda activate EDU` first |
| Loss not decreasing | Try lowering `--lr 5e-5` |

---

## 🏆 Tips to Maximise IoU Score

1. **Train longer** – Try 80-100 epochs
2. **Use stronger backbone** – `--encoder resnet101` (needs ~8GB VRAM)
3. **Lower learning rate late** – Already handled by cosine scheduler
4. **Data augmentation is on** – Flip, rotate, colour jitter all enabled
5. **Combined loss** – Using CE + Dice loss for class imbalance handling
6. **Resume training** – `--resume runs/checkpoints/latest.pth`

---

## 📦 Submission Checklist

- [ ] `runs/checkpoints/best_model.pth` – trained model weights
- [ ] `train.py`, `test.py`, `dataset.py`, `model.py`, `metrics.py`
- [ ] `runs/training_history.json` – training log
- [ ] `runs/loss_curve.png` – loss graph
- [ ] `runs/miou_curve.png` – mIoU graph
- [ ] Hackathon Report (PDF) covering methodology, results, challenges
- [ ] `README.md` – this file

---

*Built for Duality AI Offroad Autonomy Segmentation Challenge*
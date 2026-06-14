# Project 2: Neural Network Training on CIFAR-10 & Batch Normalization Analysis

Course project for *Neural Network and Deep Learning* (2026).

## Overview

- **Part 1**: Train CNN_Residual on CIFAR-10. Controlled-variable experiments comparing width, activations, loss functions, and optimizers. Best result: **93.04% test accuracy**.
- **Part 2**: Analyze how Batch Normalization helps optimization via loss landscape, gradient predictiveness, and α-smoothness experiments on VGG-A.

## Directory Structure

```
PJ2_2026/
├── report.tex              # LaTeX report
├── project_2_2026.pdf      # Project specification
├── codes/
│   ├── CIFAR10_CNN/        # Part 1: CNN training on CIFAR-10
│   │   ├── main.py         # Training entry point
│   │   ├── config.py       # Hyperparameters (modify to run different exps)
│   │   ├── comparison.py   # Cross-experiment comparison plots
│   │   ├── models/cnn.py   # BaselineCNN, CNN_BatchNorm, CNN_Residual
│   │   ├── utils/          # Training & visualization utilities
│   │   ├── data/           # Data loader with augmentation
│   │   └── saved/          # Trained models, results, figures
│   └── VGG_BatchNorm/      # Part 2: BN analysis
│       ├── analysis.py     # Three-experiment BN analysis (main script)
│       ├── VGG_Loss_Landscape.py
│       ├── models/vgg.py   # VGG_A, VGG_A_BatchNorm, VGG_A_Dropout
│       ├── utils/          # Weight initialization
│       ├── data/           # CIFAR-10 loader
│       └── reports/        # Output figures & saved models
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision, matplotlib, numpy, scikit-learn, tqdm

## Quick Start

### Part 1: Train CNN on CIFAR-10

```bash
cd codes/CIFAR10_CNN

# Edit config.py to choose model / activation / optimizer / loss / width
# Then run:
python main.py
```

Results (model weights, learning curves, confusion matrix, filter visualization) are saved under `saved/<model_tag>/`.

To generate comparison plots across experiments:
```bash
python comparison.py
```

### Part 2: Batch Normalization Analysis

```bash
cd codes/VGG_BatchNorm

# Quick validation (small data, few epochs):
python analysis.py --mode quick

# Full experiment:
python analysis.py --mode full
```

Output: `reports/figures/` (visualizations) and `reports/models/` (trained weights).

## Key Results

### Part 1 — Best Configuration

| Component | Choice |
|-----------|--------|
| Architecture | CNN_Residual (ResBlock + BatchNorm) |
| Width | 1.0 (3,005,706 params) |
| Activation | ReLU |
| Optimizer | Adam (lr=0.01) |
| Loss | CrossEntropy + Label Smoothing (ε=0.1) |
| Weight Decay | 10⁻⁴ |
| **Test Accuracy** | **93.04%** |

### Part 2 — BN Experiments

All three experiments confirm BN smooths the optimization landscape:
- Narrower loss variation across learning rates
- Lower gradient prediction error
- Smaller gradient change under parameter perturbations

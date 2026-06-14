"""
Batch Normalization Analysis — Complete Three-Experiment Script
================================================
Experiment 1: Loss Landscape (Lipschitzness)
Experiment 2: Gradient Predictiveness
Experiment 3: Alpha-Smoothness (Maximum Gradient Difference)

Usage:
    cd VGG_BatchNorm/
    python analysis.py --mode quick   # Quick validation (small data + few epochs)
    python analysis.py --mode full    # Full experiment

Output:
    reports/figures/  ← All visualization images
    reports/models/   ← Trained model weights
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import random
import argparse
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader


# ============================================================
# Configuration
# ============================================================

def setup_paths():
    """Create output directories and return paths."""
    base = os.path.dirname(os.path.abspath(__file__))
    figures_path = os.path.join(base, 'reports', 'figures')
    models_path  = os.path.join(base, 'reports', 'models')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path,  exist_ok=True)
    return figures_path, models_path


def setup_device():
    """Auto-select available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no GPU)")
    return device


# ============================================================
# Utility Functions
# ============================================================

def set_random_seeds(seed_value=0, device='cpu'):
    """Set random seeds for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, data_loader, device):
    """Compute classification accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total


# ============================================================
# Training Function (provides trained models for subsequent experiments)
# ============================================================

def train_model(model, optimizer, criterion, train_loader, val_loader,
                device, epochs_n=100, best_model_path=None,
                record_per_step=True, verbose=True):
    """
    Train model and return training history.

    Returns:
        dict: {
            'losses_list':       list[list]  per-epoch per-step loss values,
            'grads_list':        list[list]  per-epoch per-step gradient norms,
            'learning_curve':    list        epoch-averaged loss,
            'train_acc_curve':   list        per-epoch training accuracy,
            'val_acc_curve':     list        per-epoch validation accuracy,
            'best_val_acc':      float,
            'model_state_dict':  dict        best model weights,
        }
    """
    model.to(device)

    epochs_n_val = [np.nan] * epochs_n
    learning_curve = epochs_n_val.copy()
    train_acc_curve = epochs_n_val.copy()
    val_acc_curve = epochs_n_val.copy()

    losses_list = []
    grads_list = []
    max_val_acc = 0.0
    best_state = None

    batches_n = len(train_loader)
    iterator = tqdm(range(epochs_n), unit='epoch', disable=not verbose)

    for epoch in iterator:
        model.train()

        step_losses = []
        step_grads = []
        learning_curve[epoch] = 0.0

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            # Record loss
            loss_val = loss.item()
            step_losses.append(loss_val)
            learning_curve[epoch] += loss_val

            loss.backward()

            # Record L2 norm of gradients across all parameters
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            step_grads.append(total_grad_norm ** 0.5)

            optimizer.step()

        losses_list.append(step_losses)
        grads_list.append(step_grads)
        learning_curve[epoch] /= batches_n

        # Evaluation
        train_acc = get_accuracy(model, train_loader, device)
        val_acc   = get_accuracy(model, val_loader, device)
        train_acc_curve[epoch] = train_acc
        val_acc_curve[epoch]   = val_acc

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

        if verbose:
            iterator.set_postfix(
                loss=f"{learning_curve[epoch]:.3f}",
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_acc:.3f}"
            )

    return {
        'losses_list':      losses_list,
        'grads_list':       grads_list,
        'learning_curve':   learning_curve,
        'train_acc_curve':  train_acc_curve,
        'val_acc_curve':    val_acc_curve,
        'best_val_acc':     max_val_acc,
        'model_state_dict': best_state,
    }


# ============================================================
# Experiment 1: Loss Landscape (Lipschitzness)
# ============================================================

def experiment_1_loss_landscape(device, figures_path, models_path,
                                 learning_rates=None, epochs_n=20, n_items=-1):
    """
    Train models with/without BN using multiple learning rates, collect per-step loss,
    build max/min curves and plot loss landscape with fill_between.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Loss Landscape (Lipschitzness)")
    print("=" * 60)

    if learning_rates is None:
        learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

    train_loader = get_cifar_loader(train=True, n_items=n_items)
    val_loader   = get_cifar_loader(train=False, n_items=n_items)

    criterion = nn.CrossEntropyLoss()

    # ---- Train models without BN (one per lr) ----
    print("\n[1/4] Training models without BN (VGG_A)...")
    all_losses_noBN = []
    for lr in learning_rates:
        print(f"  lr={lr:.0e}")
        set_random_seeds(2020, str(device))
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        result = train_model(model, optimizer, criterion,
                             train_loader, val_loader, device,
                             epochs_n=epochs_n, verbose=False)
        # Flatten: 2D list[epoch][step] -> 1D list[step]
        flat = [loss for epoch_losses in result['losses_list'] for loss in epoch_losses]
        all_losses_noBN.append(flat)

    # ---- Train models with BN (one per lr) ----
    print("\n[2/4] Training models with BN (VGG_A_BatchNorm)...")
    all_losses_BN = []
    for lr in learning_rates:
        print(f"  lr={lr:.0e}")
        set_random_seeds(2020, str(device))
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        result = train_model(model, optimizer, criterion,
                             train_loader, val_loader, device,
                             epochs_n=epochs_n, verbose=False)
        flat = [loss for epoch_losses in result['losses_list'] for loss in epoch_losses]
        all_losses_BN.append(flat)

    # ---- Build max/min curves ----
    print("\n[3/4] Building max/min curves...")
    min_len_noBN = min(len(l) for l in all_losses_noBN)
    min_len_BN   = min(len(l) for l in all_losses_BN)

    max_noBN, min_noBN = [], []
    max_BN,   min_BN   = [], []

    for step in range(min_len_noBN):
        vals = [l[step] for l in all_losses_noBN]
        max_noBN.append(max(vals))
        min_noBN.append(min(vals))

    for step in range(min_len_BN):
        vals = [l[step] for l in all_losses_BN]
        max_BN.append(max(vals))
        min_BN.append(min(vals))

    # ---- Plotting ----
    print("[4/4] Plotting loss landscape comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    steps_noBN = range(len(max_noBN))
    steps_BN   = range(len(max_BN))

    # Left: Without BN
    axes[0].plot(steps_noBN, max_noBN, color='red', alpha=0.6, linewidth=0.3)
    axes[0].plot(steps_noBN, min_noBN, color='red', alpha=0.6, linewidth=0.3)
    axes[0].fill_between(steps_noBN, min_noBN, max_noBN,
                         color='red', alpha=0.15)
    axes[0].set_title('VGG-A (Without BN)')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(bottom=0)

    # Right: With BN
    axes[1].plot(steps_BN, max_BN, color='blue', alpha=0.6, linewidth=0.3)
    axes[1].plot(steps_BN, min_BN, color='blue', alpha=0.6, linewidth=0.3)
    axes[1].fill_between(steps_BN, min_BN, max_BN,
                         color='blue', alpha=0.15)
    axes[1].set_title('VGG-A (With BN)')
    axes[1].set_xlabel('Training Step')

    fig.suptitle('Experiment 1: Loss Landscape — VGG-A with vs without BatchNorm',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(figures_path, 'exp1_loss_landscape_split.png')
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved: {save_path}")

    # ---- Overlay on same figure + supplementary (normalized comparison) ----
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(steps_noBN, max_noBN, color='red', alpha=0.5, linewidth=0.3)
    ax.plot(steps_noBN, min_noBN, color='red', alpha=0.5, linewidth=0.3)
    ax.fill_between(steps_noBN, min_noBN, max_noBN,
                    color='red', alpha=0.12, label='Without BN')

    ax.plot(steps_BN, max_BN, color='blue', alpha=0.5, linewidth=0.3)
    ax.plot(steps_BN, min_BN, color='blue', alpha=0.5, linewidth=0.3)
    ax.fill_between(steps_BN, min_BN, max_BN,
                    color='blue', alpha=0.12, label='With BN')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Experiment 1: Loss Landscape Comparison (Overlay)')
    ax.legend(loc='upper right')
    save_path2 = os.path.join(figures_path, 'exp1_loss_landscape_overlay.png')
    fig2.savefig(save_path2, dpi=150)
    print(f"  -> Saved: {save_path2}")
    plt.close('all')

    return {
        'all_losses_noBN': all_losses_noBN,
        'all_losses_BN':   all_losses_BN,
        'max_noBN': max_noBN, 'min_noBN': min_noBN,
        'max_BN':   max_BN,   'min_BN':   min_BN,
    }


# ============================================================
# Experiment 2: Gradient Predictiveness
# ============================================================

def compute_gradient_predictiveness(model, criterion, fixed_batch, device,
                                     step_sizes=None):
    """
    Measure gradient predictiveness.

    Linear prediction: L(theta + d) = L(theta) + grad(L)(theta)*d = L(theta) - eta*||g||^2  (when d = -eta*g)
    Actual change: L(theta - eta*g) - L(theta)
    Prediction error: |actual change - predicted change|

    Args:
        model:       model in training mode (use model.train() to reflect real training landscape)
        criterion:   loss function
        fixed_batch: (X, y) fixed data batch
        device:      device
        step_sizes:  list of step sizes eta

    Returns:
        dict: {step_sizes, predicted, actual, errors}
    """
    if step_sizes is None:
        step_sizes = list(np.logspace(-1, 1, 20))  # 0.1 ~ 10

    x, y = fixed_batch
    x, y = x.to(device), y.to(device)

    # Save original parameters
    orig_params = [p.clone().detach() for p in model.parameters()]

    # Step 1: Compute gradient g and ||g||^2
    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    grads = []
    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.clone().detach()
            grads.append(g)
            grad_norm_sq += g.norm().item() ** 2
        else:
            grads.append(torch.zeros_like(p))

    current_loss = loss.item()

    # Step 2: For each step size, measure actual loss after one step
    predicted_list = []
    actual_list = []

    for eta in step_sizes:
        # Move eta along negative gradient direction
        for p, g in zip(model.parameters(), grads):
            p.data -= eta * g

        with torch.no_grad():
            new_loss = criterion(model(x), y).item()

        predicted_change = -eta * grad_norm_sq           # linear predicted change
        actual_change    = new_loss - current_loss       # actual change

        predicted_list.append(predicted_change)
        actual_list.append(actual_change)

        # Restore original parameters
        for p, orig in zip(model.parameters(), orig_params):
            p.data.copy_(orig)

    errors = [abs(p - a) for p, a in zip(predicted_list, actual_list)]

    return {
        'step_sizes': list(step_sizes),
        'predicted':  predicted_list,
        'actual':     actual_list,
        'errors':     errors,
        'grad_norm_sq': grad_norm_sq,
    }


def experiment_2_gradient_predictiveness(device, figures_path, models_path,
                                          epochs_n=20, n_items=-1):
    """
    Train two models (with/without BN), measure at multiple checkpoints during training,
    visualize and compare gradient predictiveness.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Gradient Predictiveness")
    print("=" * 60)

    train_loader = get_cifar_loader(train=True, n_items=n_items)
    val_loader   = get_cifar_loader(train=False, n_items=n_items)
    criterion = nn.CrossEntropyLoss()

    # Fix one batch for all measurements (take the first batch from val set)
    fixed_batch = next(iter(val_loader))

    lr = 0.001
    step_sizes = list(np.logspace(-1, 1.2, 20))  # 0.1 ~ ~15.8

    results_all = {}

    for model_name, ModelClass in [('noBN', VGG_A), ('BN', VGG_A_BatchNorm)]:
        print(f"\n[Training] {model_name} model...")
        set_random_seeds(2020, str(device))
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Measure at multiple training stages
        checkpoints = []
        total_steps = epochs_n * len(train_loader)

        for epoch in range(epochs_n):
            model.train()
            for data in train_loader:
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            # Measure after each epoch
            # Use train() so BN uses batch statistics — this is what
            # the optimizer actually experiences during a gradient step.
            model.train()
            gp_result = compute_gradient_predictiveness(
                model, criterion, fixed_batch, device, step_sizes
            )
            gp_result['epoch'] = epoch + 1
            checkpoints.append(gp_result)
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{epochs_n} - measured")

        results_all[model_name] = checkpoints

        # Save best model
        model_path = os.path.join(models_path, f'exp2_{model_name}.pth')
        torch.save(model.state_dict(), model_path)

    # ---- Plotting ----
    print("\n[Plotting] Gradient Predictiveness visualization...")

    epochs_to_plot = [1, 5, 10, epochs_n]
    # Generate color maps
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(epochs_to_plot)))
    colors_bn = plt.cm.Blues(np.linspace(0.3, 0.9, len(epochs_to_plot)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Figure 1: Error vs step size (final epoch comparison) ---
    ax = axes[0]
    for name, result_list in results_all.items():
        last = result_list[-1]
        label = 'Without BN' if name == 'noBN' else 'With BN'
        color = 'red' if name == 'noBN' else 'blue'
        ax.plot(last['step_sizes'], last['errors'],
                color=color, linewidth=2, label=label, marker='o', markersize=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Step Size η')
    ax.set_ylabel('Prediction Error |ΔL_actual − ΔL_pred|')
    ax.set_title('Gradient Predictiveness (Final Epoch)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Figure 2: Error curves across different epochs ---
    ax = axes[1]
    for name, result_list in results_all.items():
        for idx, epoch_target in enumerate(epochs_to_plot):
            ep_idx = min(epoch_target - 1, len(result_list) - 1)
            r = result_list[ep_idx]
            color = colors[idx] if name == 'noBN' else colors_bn[idx]
            ls = '-' if name == 'noBN' else '--'
            label_base = 'NoBN' if name == 'noBN' else 'BN'
            ax.plot(r['step_sizes'], r['errors'],
                    color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label_base} ep={r["epoch"]}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Step Size η')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Gradient Predictiveness Over Training')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Figure 3: Prediction ratio = actual / predicted vs step size ---
    ax = axes[2]
    for name, result_list in results_all.items():
        last = result_list[-1]
        label = 'Without BN' if name == 'noBN' else 'With BN'
        color = 'red' if name == 'noBN' else 'blue'
        # Compute ratio of actual to predicted change
        # Guard division-by-zero: skip where predicted is tiny
        ratios = []
        valid_etas = []
        for eta, pred, act in zip(last['step_sizes'], last['predicted'], last['actual']):
            if abs(pred) > 1e-12:
                ratios.append(abs(act / pred))
                valid_etas.append(eta)
        ax.plot(valid_etas, ratios, color=color, linewidth=2,
                label=label, marker='o', markersize=4)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='perfect prediction')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Step Size η')
    ax.set_ylabel('|Actual / Predicted| Loss Change')
    ax.set_title('Prediction Accuracy Ratio vs Step Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 2: Gradient Predictiveness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(figures_path, 'exp2_gradient_predictiveness.png')
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved: {save_path}")
    plt.close('all')

    # ---- Supplementary: Per-model training evolution across epochs ----
    for name, result_list in results_all.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epoch_nums = [r['epoch'] for r in result_list]
        mid_errors = [r['errors'][len(r['errors'])//2] for r in result_list]
        mid_step = step_sizes[len(step_sizes)//2]

        ax1.plot(epoch_nums, mid_errors, 'o-', color='darkred' if name=='noBN' else 'darkblue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(f'Prediction Error at η={mid_step:.2f}')
        ax1.set_title(f'Predictiveness Evolution ({name})')
        ax1.grid(True, alpha=0.3)

        colors_ep = plt.cm.viridis(np.linspace(0, 1, len(result_list)))
        for i, r in enumerate(result_list):
            if i % max(1, len(result_list)//8) == 0:
                ax2.plot(r['step_sizes'], r['errors'], color=colors_ep[i],
                        alpha=0.5, linewidth=1,
                        label=f'ep {r["epoch"]}')
        ax2.set_xscale('log'); ax2.set_yscale('log')
        ax2.set_xlabel('Step Size η')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title(f'Error Curves Across Training ({name})')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        sp = os.path.join(figures_path, f'exp2_predictiveness_detail_{name}.png')
        fig.savefig(sp, dpi=120)
        print(f"  → Saved: {sp}")
        plt.close('all')

    return results_all


# ============================================================
# Experiment 3: Alpha-Smoothness (Maximum Gradient Difference)
# ============================================================

def compute_alpha_smoothness(model, criterion, fixed_batch, device,
                              distances=None, n_trials=5):
    """
    Measure alpha-smoothness: gradient variation under small perturbations in parameter space.

    For each distance delta:
      1. Generate n_trials random directions d (unit vectors)
      2. Compute g1 = grad(L)(theta)
      3. Compute g2 = grad(L)(theta + delta*d)
      4. Record ||g1 - g2||
      5. Take average

    Args:
        model:       model in eval mode
        criterion:   loss function
        fixed_batch: fixed data batch
        device:      device
        distances:   list of perturbation distances
        n_trials:    number of random directions per distance

    Returns:
        dict: {distances, grad_diffs, grad_diffs_std}
    """
    if distances is None:
        distances = list(np.logspace(-2, 0.5, 15))  # 0.01 ~ 3.16

    # NOTE: model mode (train/eval) is set by the caller.
    # We do NOT call model.eval() here — the caller decides whether
    # BN uses batch stats (train) or running stats (eval).
    x, y = fixed_batch
    x, y = x.to(device), y.to(device)

    params = list(model.parameters())
    results = {'distances': distances, 'grad_diffs': [], 'grad_diffs_std': []}

    for delta in distances:
        trial_diffs = []

        for _ in range(n_trials):
            # Step 1: Compute gradient g1 at current position
            model.zero_grad()
            loss1 = criterion(model(x), y)
            loss1.backward()

            grad1 = []
            total_norm_g1_sq = 0.0
            for p in params:
                if p.grad is not None:
                    g = p.grad.clone().detach()
                    grad1.append(g)
                    total_norm_g1_sq += g.norm().item() ** 2
                else:
                    grad1.append(torch.zeros_like(p))

            # Step 2: Generate random unit direction d
            direction = []
            total_norm_sq = 0.0
            for p in params:
                d = torch.randn_like(p)
                total_norm_sq += d.norm().item() ** 2
                direction.append(d)

            total_norm = total_norm_sq ** 0.5 + 1e-12
            direction = [d / total_norm for d in direction]

            # Step 3: Move to theta + delta*d
            for p, d in zip(params, direction):
                p.data += delta * d

            # Step 4: Compute gradient g2
            model.zero_grad()
            loss2 = criterion(model(x), y)
            loss2.backward()

            grad2 = []
            for p in params:
                if p.grad is not None:
                    grad2.append(p.grad.clone().detach())
                else:
                    grad2.append(torch.zeros_like(p))

            # Step 5: Restore parameters
            for p, d in zip(params, direction):
                p.data -= delta * d

            # Step 6: Compute ||g1 - g2||
            diff_sq = 0.0
            for g1, g2 in zip(grad1, grad2):
                diff_sq += (g1 - g2).norm().item() ** 2
            trial_diffs.append(diff_sq ** 0.5)

        results['grad_diffs'].append(float(np.mean(trial_diffs)))
        results['grad_diffs_std'].append(float(np.std(trial_diffs)))

    return results


def experiment_3_alpha_smoothness(device, figures_path, models_path,
                                   epochs_n=20, n_items=-1):
    """
    Train two models, measure alpha-smoothness, visualize.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Alpha-Smoothness (Maximum Gradient Difference)")
    print("=" * 60)

    train_loader = get_cifar_loader(train=True, n_items=n_items)
    val_loader   = get_cifar_loader(train=False, n_items=n_items)
    criterion = nn.CrossEntropyLoss()
    fixed_batch = next(iter(val_loader))

    lr = 0.001
    distances = list(np.logspace(-2, 0.6, 18))  # 0.01 ~ ~3.98
    n_trials = 5

    results_all = {}

    for model_name, ModelClass in [('noBN', VGG_A), ('BN', VGG_A_BatchNorm)]:
        print(f"\n[Training] {model_name} model...")
        set_random_seeds(2020, str(device))
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        checkpoint_epochs = [1, 5, 10, 15, epochs_n]
        checkpoints = []

        for epoch in range(epochs_n):
            model.train()
            for data in train_loader:
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) in checkpoint_epochs:
                # Use train() so BN uses batch statistics for a fair comparison
                model.train()
                sm_result = compute_alpha_smoothness(
                    model, criterion, fixed_batch, device, distances, n_trials
                )
                sm_result['epoch'] = epoch + 1
                checkpoints.append(sm_result)
                print(f"  epoch {epoch+1}/{epochs_n} - alpha-smoothness measured")

        results_all[model_name] = checkpoints

        # Save model
        model_path = os.path.join(models_path, f'exp3_{model_name}.pth')
        torch.save(model.state_dict(), model_path)

    # ---- Plotting ----
    print("\n[Plotting] Alpha-Smoothness visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Figure 1: Final epoch comparison ---
    ax = axes[0]
    for name, cp_list in results_all.items():
        last = cp_list[-1]
        label = 'Without BN' if name == 'noBN' else 'With BN'
        color = 'red' if name == 'noBN' else 'blue'
        diffs = np.array(last['grad_diffs'])
        stds  = np.array(last['grad_diffs_std'])
        ax.plot(last['distances'], diffs, color=color, linewidth=2,
                label=label, marker='o', markersize=4)
        ax.fill_between(last['distances'], diffs - stds, diffs + stds,
                        color=color, alpha=0.1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Perturbation Distance δ')
    ax.set_ylabel('||∇L(θ+δd) − ∇L(θ)||')
    ax.set_title('α-Smoothness (Final Epoch)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Figure 2: Smoothness evolution across epochs ---
    ax = axes[1]
    colors_epoch = plt.cm.viridis(np.linspace(0, 1, len(checkpoint_epochs)))
    for name, cp_list in results_all.items():
        ls = '-' if name == 'noBN' else '--'
        for idx, cp in enumerate(cp_list):
            base_label = 'NoBN' if name == 'noBN' else 'BN'
            ax.plot(cp['distances'], cp['grad_diffs'],
                    color=colors_epoch[idx], linestyle=ls,
                    linewidth=1.5, alpha=0.7,
                    label=f'{base_label} ep={cp["epoch"]}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Perturbation Distance δ')
    ax.set_ylabel('||∇L(θ+δd) − ∇L(θ)||')
    ax.set_title('α-Smoothness Over Training')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Figure 3: Effective smoothness constant evolution ---
    ax = axes[2]
    for name, cp_list in results_all.items():
        label = 'Without BN' if name == 'noBN' else 'With BN'
        color = 'red' if name == 'noBN' else 'blue'
        # Fit slope using least squares (in log-log space)
        epochs_recorded = []
        smoothness_constants = []
        for cp in cp_list:
            log_d = np.log(np.array(cp['distances']))
            log_g = np.log(np.array(cp['grad_diffs']))
            # Slope ~ 1 implies linear smoothness
            slope, intercept = np.polyfit(log_d, log_g, 1)
            # Effective smoothness constant: exp(intercept)
            smoothness_constants.append(np.exp(intercept))
            epochs_recorded.append(cp['epoch'])
        ax.plot(epochs_recorded, smoothness_constants,
                'o-', color=color, linewidth=2, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Smoothness Constant L')
    ax.set_title('Smoothness Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: α-Smoothness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(figures_path, 'exp3_alpha_smoothness.png')
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved: {save_path}")
    plt.close('all')

    return results_all


# ============================================================
# Summary Visualization: Combined Figure for All Three Experiments
# ============================================================

def plot_summary(figures_path, exp1_results, exp2_results, exp3_results):
    """Generate a combined comparison figure for the report."""
    print("\n[Plotting] Summary figure...")

    fig = plt.figure(figsize=(14, 10))

    # (a) Loss Landscape - top left
    ax1 = fig.add_subplot(2, 2, 1)
    steps_noBN = range(len(exp1_results['max_noBN']))
    steps_BN   = range(len(exp1_results['max_BN']))
    ax1.fill_between(steps_noBN, exp1_results['min_noBN'], exp1_results['max_noBN'],
                     color='red', alpha=0.15, label='Without BN')
    ax1.fill_between(steps_BN, exp1_results['min_BN'], exp1_results['max_BN'],
                     color='blue', alpha=0.15, label='With BN')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
    ax1.set_title('(a) Loss Landscape')
    ax1.legend(fontsize=8)

    # (b) Gradient Predictiveness - top right
    ax2 = fig.add_subplot(2, 2, 2)
    for name in ['noBN', 'BN']:
        last = exp2_results[name][-1]
        color = 'red' if name == 'noBN' else 'blue'
        label = 'Without BN' if name == 'noBN' else 'With BN'
        ax2.plot(last['step_sizes'], last['errors'],
                 color=color, linewidth=2, label=label, marker='o', markersize=3)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('Step Size η'); ax2.set_ylabel('Prediction Error')
    ax2.set_title('(b) Gradient Predictiveness')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (c) Alpha-Smoothness - bottom left
    ax3 = fig.add_subplot(2, 2, 3)
    for name in ['noBN', 'BN']:
        last = exp3_results[name][-1]
        color = 'red' if name == 'noBN' else 'blue'
        label = 'Without BN' if name == 'noBN' else 'With BN'
        ax3.plot(last['distances'], last['grad_diffs'],
                 color=color, linewidth=2, label=label, marker='o', markersize=4)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel('Perturbation Distance δ')
    ax3.set_ylabel('||∇L(θ+δd) − ∇L(θ)||')
    ax3.set_title('(c) α-Smoothness')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # (d) Text summary - bottom right
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    summary_lines = [
        "Summary: How BN Helps Optimization",
        "-" * 38,
        "",
        "(a) Loss Landscape:",
        "  BN narrows the loss variation band",
        "  -> landscape is more Lipschitz.",
        "",
        "(b) Gradient Predictiveness:",
        "  Lower prediction error with BN",
        "  -> linear approximation more reliable.",
        "",
        "(c) Alpha-Smoothness:",
        "  Smaller gradient variation over",
        "  distance -> more stable updates.",
        "",
        "Conclusion: BN reparametrizes the",
        "underlying optimization problem, making",
        "the loss landscape significantly smoother",
        "and enabling faster, more stable training.",
    ]
    summary_text = "\n".join(summary_lines)
    ax4.text(0.05, 0.98, summary_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment='top', linespacing=1.1,
             bbox=dict(boxstyle='round', pad=0.6, facecolor='wheat', alpha=0.8))

    fig.suptitle('Batch Normalization Analysis — Summary',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(figures_path, 'summary_all_experiments.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  -> Saved: {save_path}")
    plt.close('all')


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch Normalization three-experiment analysis script'
    )
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='quick: small data + few epochs for quick validation; full: complete training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides default)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu, cuda:0, auto')

    args = parser.parse_args()

    # Determine device and paths
    figures_path, models_path = setup_paths()

    if args.device == 'auto':
        device = setup_device()
    else:
        device = torch.device(args.device)
        print(f"[Device] {device}")

    # Determine mode and parameters
    if args.mode == 'quick':
        epochs_n = args.epochs if args.epochs else 5
        n_items = 500
        lrs = [1e-3, 2e-3]  # Use only 2 lrs to speed up
        print(f"\n[Quick Mode] epochs={epochs_n}, n_items={n_items}, lrs={lrs}")
        print("   (For code validation only, use --mode full for final report)\n")
    else:
        epochs_n = args.epochs if args.epochs else 30
        n_items = -1  # Full dataset
        lrs = [1e-3, 2e-3, 1e-4, 5e-4]
        print(f"\n📊 Full Mode: epochs={epochs_n}, n_items=all, lrs={lrs}\n")

    # ---- Run three experiments ----
    exp1_results = experiment_1_loss_landscape(
        device, figures_path, models_path,
        learning_rates=lrs, epochs_n=epochs_n, n_items=n_items
    )

    exp2_results = experiment_2_gradient_predictiveness(
        device, figures_path, models_path,
        epochs_n=epochs_n, n_items=n_items
    )

    exp3_results = experiment_3_alpha_smoothness(
        device, figures_path, models_path,
        epochs_n=epochs_n, n_items=n_items
    )

    # ---- Summary ----
    plot_summary(figures_path, exp1_results, exp2_results, exp3_results)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Output directory: {figures_path}")
    print("Generated files:")
    for f in sorted(os.listdir(figures_path)):
        print(f"  • {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
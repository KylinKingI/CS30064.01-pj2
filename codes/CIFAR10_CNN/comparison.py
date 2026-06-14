import pickle
import os
import matplotlib.pyplot as plt

SAVE_DIR = './saved'

# Specify the experiments to be compared
experiments = {
    'adam':       'cnn_residual_relu_adam_ce_wd0.0001_wid1.0_history.pkl',
    'momentum_sgd':       'cnn_residual_relu_momentum_sgd_ce_wd0.0001_wid1.0_history.pkl',
    # 'gelu':        'cnn_residual_gelu_adam_ce_wd0.0001_wid1.0_history.pkl',
    # 'ce_smooth_1e-3':       'cnn_residual_relu_adam_ce_smooth_wd0.001_wid1.0_history.pkl',
}

# load the histories
all_histories = {}
for label, fname in experiments.items():
    path = os.path.join(SAVE_DIR, fname)
    with open(path, 'rb') as f:
        all_histories[label] = pickle.load(f)

# plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

# Train Loss
for (label, h), color in zip(all_histories.items(), colors):
    axes[0].plot(h['train_loss'], label=label, color=color, linewidth=1.5)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Train Accuracy
for (label, h), color in zip(all_histories.items(), colors):
    axes[1].plot(h['train_acc'], label=label, color=color, linewidth=1.5)
axes[1].set_title('Training Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

# Validation Accuracy
for (label, h), color in zip(all_histories.items(), colors):
    axes[2].plot(h['val_acc'], label=label, color=color, linewidth=1.5)
axes[2].set_title('Validation Accuracy')
axes[2].set_xlabel('Epoch')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'compare_optim.png'), dpi=150)

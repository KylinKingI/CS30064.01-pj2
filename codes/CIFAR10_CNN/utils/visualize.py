import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ========== plot training history ==========
def plot_learning_curves(history, save_path='learning_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


# ========== visualize convolutional filters ==========
def visualize_conv_filters(model, layer_name='conv1', save_path='filters.png'):
    weight = model.state_dict()[layer_name + '.weight'].cpu()
    out_channels, in_channels, kH, kW = weight.shape
    cols = 8
    rows = (out_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i in range(out_channels):
        ax = axes[i // cols][i % cols]
        kernel = weight[i].mean(dim=0)
        ax.imshow(kernel, cmap='coolwarm')
        ax.axis('off')

    for i in range(out_channels, rows * cols):
        axes[i // cols][i % cols].axis('off')
    
    plt.suptitle(f'Filters of {layer_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


# ========== visualize confusion matrix ==========
def plot_confusion_matrix(model, data_loader, device, classes, save_path='confusion_matrix.png'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


# ========== show misclassified samples ==========
def show_misclassified(model, data_loader, device, classes, num_images=16, save_path='misclassified.png'):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for x, y in data_loader:
            x_gpu = x.to(device)
            outputs = model(x_gpu)
            _, preds = torch.max(outputs, 1)

            mask = preds.cpu() !=y
            for i in range(len(y)):
                if mask[i] and len(misclassified) < num_images:
                    misclassified.append((x[i], y[i].item(), preds[i].item()))
            if len(misclassified) >= num_images:
                break
    
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows*3))
    axes = axes.flatten() if rows > 1 else [axes]

    for i, (img, true_label, pred_label) in enumerate(misclassified):
        img = img * 0.5 + 0.5
        img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]}', color='red')
        axes[i].axis('off')
    
    for i in range(len(misclassified), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
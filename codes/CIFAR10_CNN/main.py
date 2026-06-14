import torch
import torch.nn as nn
import random
import numpy as np
import os
import pickle

from config import *
from config import get_model_tag, get_model_name
from data.loaders import get_cifar_loader
from models.cnn import build_model
from utils.train_utils import train, get_accuracy
from utils.visualize import plot_learning_curves, visualize_conv_filters, plot_confusion_matrix, show_misclassified


# ========== set the random seed ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ========== main function entry ==========
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Model type: {MODEL_TYPE}')
    print(f'Model tag: {get_model_tag()}')

    # ======= load the data =======
    train_loader = get_cifar_loader(root=DATA_ROOT, batch_size=BATCH_SIZE, train=True, num_workers=NUM_WORKERS, augment=True)
    val_loader = get_cifar_loader(root=DATA_ROOT, batch_size=BATCH_SIZE, train=False, num_workers=NUM_WORKERS)
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # ======= build the model =======
    model = build_model(model_type=MODEL_TYPE, activation=ACTIVATION, width=WIDTH)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # ======= select the optimizer =======
    if OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'momentum_sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f'Unkown OPTIMIZER: {OPTIMIZER}')
    
    # ======= loss function =======
    if LOSS_TYPE == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif LOSS_TYPE == 'ce_smooth':
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    else:
        raise ValueError(f'Unkown LOSS TYPE: {LOSS_TYPE}')

    # ======= scheduler =======
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ======= training =======
    model_tag = get_model_tag()
    model_save_path = os.path.join(SAVE_DIR, get_model_name())
    history = train(model, train_loader, val_loader, optimizer, criterion, device, EPOCHS, scheduler, save_path=model_save_path)
    history_path = os.path.join(SAVE_DIR, f'{model_tag}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    # ======= test the best model =======
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_acc = get_accuracy(model, val_loader, device)
    print(f'\nFinal test accuracy: {test_acc:.4f}')

    # ========== visualization ==========
    print('\nGenerating visualizations...')

    # learning curves
    plot_learning_curves(history, save_path=os.path.join(SAVE_DIR, f'{model_tag}_learning_curve.png'))

    # convolution filters
    if MODEL_TYPE == 'cnn':
        filter_layer = 'conv1'
    elif MODEL_TYPE == 'cnn_bn':
        filter_layer = 'conv_block1.0'
    elif MODEL_TYPE == 'cnn_residual':
        filter_layer = 'conv1.0'
    else:
        raise ValueError(f'Unknown MODEL_TYPE: {MODEL_TYPE}')

    visualize_conv_filters(model, layer_name=filter_layer, 
                           save_path=os.path.join(SAVE_DIR, f'{model_tag}_conv_filters.png'))

    # confusion matrix
    plot_confusion_matrix(model, val_loader, device, CIFAR10_CLASSES,
                          save_path=os.path.join(SAVE_DIR, f'{model_tag}_confusion_matrix.png'))

    # misclassified samples
    show_misclassified(model, val_loader, device, CIFAR10_CLASSES,
                       save_path=os.path.join(SAVE_DIR, f'{model_tag}_misclassified.png'))

    print(f'\nAll results saved to: {SAVE_DIR}')



if __name__ == '__main__':
    main()

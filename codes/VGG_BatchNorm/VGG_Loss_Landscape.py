import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    # print the shape of X and y
    print(f'X shape: {X.shape}, y shape: {y.shape}')

    # print the first 10 values of y
    print(f'y[:10]: {y[:10]}')

    # print the range of values in X (min and max)
    print(f'X range: min={X.min().item():.3f}, max={X.max().item():.3f}')
    ## --------------------
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    ## --------------------
    # Add code as needed
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    model.train()
    return accuracy
    #
    #
    #
    ## --------------------

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss_value = loss.item()
            loss_list.append(loss_value)
            learning_curve[epoch] += loss_value
            loss.backward()
            #
            # record the gradient norm of the last linear layer in classifier
            grad_norm = model.classifier[4].weight.grad.clone().norm().item()
            grad.append(grad_norm)
            #
            ## --------------------

            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        train_acc = get_accuracy(model, train_loader, device)
        train_accuracy_curve[epoch] = train_acc
        val_acc = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_acc

        axes[1].plot(train_accuracy_curve, label='Train Accuracy')
        axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
        axes[1].legend()

        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

        plt.savefig(os.path.join(figures_path, 'train_curves.png'))
        plt.close(f)
        #
        ## --------------------

    return losses_list, grads, learning_curve, train_accuracy_curve, val_accuracy_curve


# Train your model
# feel free to modify
epo = 20
loss_save_path = os.path.join(models_path, 'loss_landscape')
grad_save_path = os.path.join(models_path, 'grad_landscape')
os.makedirs(loss_save_path, exist_ok=True)
os.makedirs(grad_save_path, exist_ok=True)

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, grads, _1, _2, _3 = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
epo = 20

# train model without batch normalization
all_losses_noBN = []
for lr in learning_rates:
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses_lists, grads, _1, _2, _3 = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    flat_losses = [item for sublist in losses_lists for item in sublist]
    all_losses_noBN.append(flat_losses)


# train model with batch normalization
all_losses_BN = []
for lr in learning_rates:
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A_BatchNorm()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses_lists, grads, _1, _2, _3 = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    flat_losses = [item for sublist in losses_lists for item in sublist]
    all_losses_BN.append(flat_losses)

# calculate min_curve and max_curve
min_len_noBN = min(len(losses) for losses in all_losses_noBN)
min_len_BN = min(len(losses) for losses in all_losses_BN)

max_curve_noBN = []
min_curve_noBN = []
max_curve_BN = []
min_curve_BN = []

for i in range(min_len_noBN):
    step_losses = [losses[i] for losses in all_losses_noBN]
    max_curve_noBN.append(max(step_losses))
    min_curve_noBN.append(min(step_losses))

for i in range(min_len_BN):
    step_losses = [losses[i] for losses in all_losses_BN]
    max_curve_BN.append(max(step_losses))
    min_curve_BN.append(min(step_losses))
#
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
save_path = os.path.join(figures_path, 'loss_landscapes.png')

def plot_loss_landscape():
    ## --------------------
    # Add your code
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    steps_noBN = range(len(max_curve_noBN))
    steps_BN = range(len(max_curve_BN))

    ax.plot(steps_noBN, max_curve_noBN, color='green', alpha=0.5, linewidth=0.3)
    ax.plot(steps_noBN, min_curve_noBN, color='green', alpha=0.5, linewidth=0.3)
    ax.fill_between(steps_noBN, min_curve_noBN, max_curve_noBN, color='green', alpha=0.15, label='Without BatchNorm')

    ax.plot(steps_BN, max_curve_BN, color='red', alpha=0.5, linewidth=0.3)
    ax.plot(steps_BN, min_curve_BN, color='red', alpha=0.5, linewidth=0.3)
    ax.fill_between(steps_BN, min_curve_BN, max_curve_BN, color='red', alpha=0.15, label='With BatchNorm')

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Landscape: VGG_A with and without Batch Normalization')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()
    ## --------------------
    

plot_loss_landscape()
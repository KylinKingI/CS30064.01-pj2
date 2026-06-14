import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm


# ========== accuracy calculation ==========
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# ========== training function ==========
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(train_loader.dataset)

def train(model, train_loader, val_loader, optimizer, criterion, device='cpu', num_epochs=50, scheduler=None, save_path='best_model.pth'):
    model = model.to(device)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_val_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc='Training'):
        # --------- train for one epoch ---------
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = get_accuracy(model, train_loader, device)

        # --------- evaluate on validation set ---------
        val_acc = get_accuracy(model, val_loader, device)

        # --------- sheduler step ---------
        if scheduler is not None:
            scheduler.step()

        # --------- record history ---------
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # --------- save best model ---------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'[Epoch {epoch+1}] New best model saved with val_acc: {best_val_acc:.4f}')
        
        # --------- print progress ---------
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    print(f'Training completed. Best Val Acc: {best_val_acc:.4f}')
    return history

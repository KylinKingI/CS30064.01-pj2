import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== activation function factory ==========
def get_activation(name):
    """return activation function by name"""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'elu': nn.ELU(inplace=True),
        'gelu': nn.GELU(),
    }
    return activations.get(name, nn.ReLU(inplace=True))


# ==========  Baseline CNN model ==========
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, activation='relu', width=1.0):
        super().__init__()
        c1 = int(64 * width)
        c2 = int(128 * width)
        c3 = int(256 * width)
        f1 = int(256 * width)
        f2 = int(128 * width)

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(c3 * 4 * 4, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, num_classes)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# ========== BatchNorm CNN model ==========
class CNN_BatchNorm(nn.Module):
    def __init__(self, num_classes=10, activation='relu', width=1.0):
        super().__init__()
        activation_fn = get_activation(activation)
        c1 = int(64 * width)
        c2 = int(128 * width)
        c3 = int(256 * width)
        f1 = int(256 * width)
        f2 = int(128 * width)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c1),
            activation_fn, 
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c2),
            activation_fn, 
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c3),
            activation_fn, 
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(c3 * 4 * 4, f1),
            activation_fn,
            nn.Linear(f1, f2),
            activation_fn,
            nn.Linear(f2, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ========== Residual CNN model ==========
class ResBlock(nn.Module):
    def __init__(self, channels, activation='relu'):
        super().__init__()
        activation_fn = get_activation(activation)
        self.activation_fn = activation_fn
        
        self.conv_bn_act = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(channels),
            activation_fn, 
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = self.conv_bn_act(x)
        out += x
        out = self.activation_fn(out)
        return out
        
class CNN_Residual(nn.Module):
    def __init__(self, num_classes=10, activation='relu', width=1.0):
        super().__init__()
        activation_fn = get_activation(activation)
        c1 = int(64 * width)
        c2 = int(128 * width)
        c3 = int(256 * width)
        f1 = int(256 * width)
        f2 = int(128 * width)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c1),
            activation_fn
        )
        self.res_block1 = ResBlock(c1, activation)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c2),
            activation_fn
        )
        self.res_block2 = ResBlock(c2, activation)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c3),
            activation_fn
        )
        self.res_block3 = ResBlock(c3, activation)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(c3 * 4 * 4, f1),
            activation_fn,
            nn.Linear(f1, f2),
            activation_fn,
            nn.Linear(f2, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.res_block2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.res_block3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

# ========== Model factory ==========
def build_model(model_type='cnn', num_classes=10, activation='relu', width=1.0):
    models = {
        'cnn': BaselineCNN(num_classes, activation, width),
        'cnn_bn': CNN_BatchNorm(num_classes, activation, width),
        'cnn_residual': CNN_Residual(num_classes, activation, width),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type]
import os

DATA_ROOT = '../data'
SAVE_DIR = './saved'
LOG_DIR = os.path.join(SAVE_DIR, 'logs')

BATCH_SIZE = 128
NUM_WORKERS = 4

EPOCHS = 50
LEARNING_RATE = 0.01
LOSS_TYPE = 'ce'          # 'ce' | 'ce_smooth'
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4       # L2 regularization
MOMENTUM = 0.9            # for momentum SGD, ignored for Adam
OPTIMIZER = 'momentum_sgd'        # 'adam' | 'momentum_sgd'
WIDTH = 1.0

MODEL_TYPE = 'cnn_residual'     # 'cnn' | 'cnn_bn' | 'cnn_residual'
ACTIVATION = 'relu'       # 'relu' | 'leaky_relu' | 'elu' | 'gelu'

def get_model_tag():
    parts = [
        MODEL_TYPE, ACTIVATION, OPTIMIZER, LOSS_TYPE, f'wd{WEIGHT_DECAY}', f'wid{WIDTH}'
    ]
    return '_'.join(parts)

def get_model_name():
    return get_model_tag() + '.pth'

SEED = 42
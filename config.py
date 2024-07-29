import os
from tensorflow.keras.initializers import he_normal, he_uniform, lecun_normal, lecun_uniform, glorot_normal, glorot_uniform
from tensorflow.keras.activations import relu, tanh, sigmoid

class Config:
    DATASET_DIR = 'Dataset'
    MODEL_DIR = 'Models'
    HISTORY_DIR = 'histories'
    SEED = 42
    BATCH_SIZE = 64
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    EPOCHS = 2

    KERNEL_INITIALIZERS = {
        'he_normal': he_normal(),
        'he_uniform': he_uniform(),
        'lecun_normal': lecun_normal(),
        'lecun_uniform': lecun_uniform(),
        'glorot_normal': glorot_normal(),
        'glorot_uniform': glorot_uniform()
    }

    ACTIVATION_FUNCTIONS = {
        'relu': relu,
        'tanh': tanh,
        'sigmoid': sigmoid,
        'leaky_relu': 'leaky_relu'
    }
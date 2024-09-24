from autoencoder import logger
from autoencoder.entity.entity_config import PrepareInputOutputData, ModelParametersConfig, TrainingModelConfig

import numpy as np
import os
import cv2
import warnings;
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, Add
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K



class ModelBuilding():
    def __init__(self, params_config = ModelParametersConfig):
        self.params_config = params_config

    def residual_block(self, x, filters, kernel_size, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.ReLU()(x)
        return x
    
    def build_autoencoder(self) -> tf.keras.Model:
        input_shape = self.params_config.input_shape
        kernel_size = self.params_config.kernel_size
        latent_dim = self.params_config.latent_dim
        layer_filters = self.params_config.layer_filters
        logger.info(f'The Input shape is {input_shape}')
        logger.info(f'The kernel size is {kernel_size} ')
        logger.info(f'The latent vector Lenght is {latent_dim}')
        logger.info("Building Encoder Part of the Model")
        inputs = Input(shape=input_shape)
        x = layers.Conv2D(layer_filters[0], (kernel_size, kernel_size), padding='same', strides=2)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.residual_block(x, layer_filters[0], kernel_size)
        x = layers.Conv2D(layer_filters[1], (kernel_size, kernel_size), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.residual_block(x, layer_filters[1], kernel_size)
        x = layers.Conv2D(layer_filters[2], (kernel_size, kernel_size), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.residual_block(x, layer_filters[2], kernel_size)

        # Flatten and Latent Vector
        x = layers.Flatten()(x)
        latent = layers.Dense(latent_dim, name='latent_vector')(x)

        # Decoder
        logger.info("Building Dencoder Part of the Model")
        x = layers.Dense(16 * 16 * layer_filters[2])(latent)  # Adjust dimensions to match encoder output shape
        x = layers.Reshape((16, 16, layer_filters[2]))(x)
        
        x = self.residual_block(x, layer_filters[2], kernel_size)
        x = layers.Conv2DTranspose(layer_filters[1], (kernel_size, kernel_size), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.residual_block(x, layer_filters[1], kernel_size)
        x = layers.Conv2DTranspose(layer_filters[0], (kernel_size, kernel_size), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.residual_block(x, layer_filters[0], kernel_size)
        x = layers.Conv2DTranspose(32, (kernel_size, kernel_size), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.residual_block(x, 32, kernel_size)

        # Output layer
        outputs = layers.Conv2D(3, (kernel_size, kernel_size), activation='sigmoid', padding='same')(x)

        # Autoencoder model
        autoencoder = models.Model(inputs, outputs)
        logger.info("Autoencoder Model Building Process Completed")
        return autoencoder
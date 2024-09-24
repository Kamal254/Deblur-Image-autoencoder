from autoencoder import logger
from autoencoder.utils.util_functions import create_dir
from autoencoder.entity.entity_config import TrainingModelConfig

import numpy as np
import json
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




class TrainModel():
    def __init__(self, train_config = TrainingModelConfig):
        self.train_config = train_config

    def train_model(self, x : np.ndarray, y : np.ndarray, model : tf.keras.Model):
        logger.info(f"Compiling Autoencoder Model with loss : {self.train_config.loss}")
        logger.info(f"Compiling Autoencoder Model with optimizer : {self.train_config.optimizer}")
        logger.info(f"Compiling Autoencoder Model with metrics : {self.train_config.metrics}")
        model.compile(loss=self.train_config.loss, optimizer = self.train_config.optimizer, metrics=self.train_config.metrics)
        logger.info("Model Compiled Successfully")
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)
        callbacks = [lr_reducer]
        logger.info(f'Training Model for Epoch : {self.train_config.epochs}')
        logger.info(f'Training Model with BatchSize : {self.train_config.batch_size}')
        history = model.fit(x, y, epochs=self.train_config.epochs,batch_size=self.train_config.batch_size, callbacks = callbacks)
        logger.info('Model Trained Successfully Now saving model') 
        create_dir([self.train_config.model_dir])
        model.export(self.train_config.model_path)
        model.save(self.train_config.HDFmodel_path)
        with open(self.train_config.model_history_path, 'w') as f:
            json.dump(history.history, f)
        logger.info(f'model saved at {self.train_config.model_path}')
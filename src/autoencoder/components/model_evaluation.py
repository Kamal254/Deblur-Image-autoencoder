from autoencoder.entity.entity_config import ModelEvaluationConfig
from autoencoder import logger
from autoencoder.utils.util_functions import create_dir


import mlflow
import os
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from mlflow.models import infer_signature
import json

import numpy as np
# import cv2
import requests
import gdown


import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.models import Model



class ModelEvaluation:
    def __init__(self, eval_config : ModelEvaluationConfig):
        self.eval_config = eval_config

    def download_test_data(self, source_url:str,dest_path:str  ) ->str:
        try:
            folder_id = source_url.split("/")[-1]
            folder_id = folder_id.split("?")[0]

            # api_token = self.download_config.gdrive_api_key
            api_url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key=AIzaSyD9C-ouf5DmOrKzH5p4DsLcZ6k8FB-o13I"
            logger.info(f"Getting Response from Gdrive api_url : {api_url}")
            response = requests.get(api_url)
            logger.info(f'response : {response.status_code}')
            if response.status_code==200:
                logger.info("Response 200 OK")
                files = response.json().get('files', [])
                logger.info(f'Downloading files in folder : {dest_path}')
                for file in files:
                    file_id = file['id']
                    file_name = file['name']
                    download_url = f'https://drive.google.com/uc?id={file_id}'
                    
                    # Download the file using gdown
                    gdown.download(download_url, os.path.join(dest_path, file_name), quiet=False)
                logger.info("**** Test Images Downloaded ****")
        except Exception as e:
            raise e
        
    def download_testblurimages(self):
        try:
            create_dir([self.eval_config.test_blur_images_path])
            logger.info(f'Downloading Blur Images Test Data into {self.eval_config.test_blur_images_path}')
            self.download_test_data(self.eval_config.test_blurimages_source, self.eval_config.test_blur_images_path)
        except Exception as e:
            raise e

    def download_testcleanimages(self):
        try:
            create_dir([self.eval_config.test_clean_images_path])
            logger.info(f'Downloading Clean Images Test Data into {self.eval_config.test_clean_images_path}')
            self.download_test_data(self.eval_config.test_cleanimages_source, self.eval_config.test_clean_images_path)
        except Exception as e:
            raise e
    
    def psnr_metric(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    

    def image_to_array(self, folder_path) -> np.ndarray:
        array = []
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = image.load_img(img_path)
                img = image.img_to_array(img)
                img = img/255.0
                array.append(img)
            except Exception as e:
                raise e
        logger.info("Image Data Successfully Converted into numpy array")
        return np.array(array)

    def evaluate_model(self) -> str:
        logger.info(f'Loading Model from {self.eval_config.HDFmodel_path}')
        try:
            model = tf.keras.models.load_model(self.eval_config.HDFmodel_path,
                                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

            x_test = self.image_to_array(self.eval_config.test_blur_images_path)
            y_test = self.image_to_array(self.eval_config.test_clean_images_path)
            logger.info('Evaluating Model on Test Data')

            # Get model predictions (deblurred images)
            y_pred = model.predict(x_test)

            # Calculate PSNR for each image and average the scores
            psnr_values = [self.psnr_metric(y_true, y_pred).numpy() for y_true, y_pred in zip(y_test, y_pred)]
            avg_psnr = np.mean(psnr_values)

            logger.info(f'Average PSNR on Test Data: {avg_psnr}')
            # return f'Average PSNR: {avg_psnr}'
        
        except Exception as e:
            logger.error(f'Error occurred during model evaluation: {e}')
            raise e

    def log_into_mlflow(self) ->str:
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'Kamal254'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = '3cbdb442b5873e54a9130d1e83862bb2e7993f55'

        remote_server_uri = "https://dagshub.com/Kamal254/Deblur-Image-autoencoder.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info("Saving Experiments to the mlflow")

        with mlflow.start_run():
            model = tf.keras.models.load_model(self.eval_config.HDFmodel_path,
                                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

            x_test = self.image_to_array(self.eval_config.test_blur_images_path)
            y_test = self.image_to_array(self.eval_config.test_clean_images_path)
            
            logger.info("Logging All the Parameters into mlflow")
            mlflow.log_param("Loss", self.eval_config.loss)
            mlflow.log_param("optimizer",self.eval_config.optimizer)
            mlflow.log_param("metrics",self.eval_config.metrics)
            mlflow.log_param("batch_size",self.eval_config.batch_size)
            mlflow.log_param("epochs",self.eval_config.epochs)

            logger.info("Logging training metrics into mlflow")
            with open(self.eval_config.model_history_path, 'r') as f:
                history = json.load(f)

            for epoch in range(self.eval_config.epochs):
                mlflow.log_metric('train_loss', history['loss'][epoch], step=epoch)
                mlflow.log_metric('train_accuracy', history['acc'][epoch], step=epoch)

            logger.info("Logging Evaluation metrics into mlflow")

            y_pred = model.predict(x_test)

            # Calculate PSNR for each image and average the scores
            psnr_values = [self.psnr_metric(y_true, y_pred).numpy() for y_true, y_pred in zip(y_test, y_pred)]
            avg_psnr = np.mean(psnr_values)
            mlflow.log_metric(' peak signal-to-noise ratio ', avg_psnr)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(model, "model", registered_model_name="autoencoder")
            else:
                mlflow.keras.log_model(model, "model")
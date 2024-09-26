from autoencoder import logger
from autoencoder.entity.entity_config import PrepareInputOutputData

import numpy as np
import os
# import cv2
import warnings;
warnings.filterwarnings('ignore')

from keras.preprocessing import image



class PrepareDataForTraining():
    def __init__(self, model_config:PrepareInputOutputData):
        self.model_config = model_config

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
    
    def prepareinputdata(self)->np.ndarray:
        try:
            logger.info("Converting Blur Image Data into Array for Model Input")
            input_data_path = self.model_config.blur_image_folder
            input_data = self.image_to_array(input_data_path)
        except Exception as e:
            raise e
        return input_data

    def prepareoutputdata(self)->np.ndarray:
        try:
            logger.info("Converting Clean Image Data into Array for Model Output")
            output_data_path = self.model_config.blur_image_folder
            output_data = self.image_to_array(output_data_path)
        except Exception as e:
            raise e
        return output_data
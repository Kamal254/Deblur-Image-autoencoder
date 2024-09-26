from autoencoder import logger
from autoencoder.entity.entity_config import DownloadDataConfig, PreprocessDataConfig, PrepareBlurImageConfig


import requests
import gdown
import os
# import cv2
from keras.preprocessing import image
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter


class DataIngestionPreparation:
    def __init__(self, download_config: DownloadDataConfig,
                  preprocess_config: PreprocessDataConfig,
                  blurimage_config : PrepareBlurImageConfig):
        self.download_config = download_config
        self.preprocess_config = preprocess_config
        self.blurimage_config = blurimage_config

    def download_file(self,file_info, destination_folder):
        file_id = file_info['id']
        file_name = file_info['name']
        download_url = f'https://drive.google.com/uc?id={file_id}'
        output_path = os.path.join(destination_folder, file_name)
        gdown.download(download_url, output_path, quiet=False)

    def download_files_in_parallel(self):
        try:
            source_url = self.download_config.source_gdrive_url
            folder_id = source_url.split("/")[-1]
            folder_id = folder_id.split("?")[0]

            api_token = self.download_config.gdrive_api_key
            api_url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={api_token}"
            response = requests.get(api_url)
            logger.info(f'response : {response.status_code}')
            if response.status_code == 200:
                files = response.json().get('files', [])
                # Use ThreadPoolExecutor to download files in parallel
                with ThreadPoolExecutor(max_workers=32) as executor:
                    executor.map(lambda file: self.download_file(file, self.download_config.download_image_folder), files)
            else:
                print("Failed to retrieve folder content.")
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            raise e


    def preprocess_data(self) ->str:
        try:
            source_folder = self.preprocess_config.download_image_folder
            destination_folder = self.preprocess_config.processed_image_folder
            image_paths = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
            logger.info('Resizing the Images to (128, 128)')
            image_size = (128, 128)
            for image in image_paths:
                src = os.path.join(source_folder, image)
                dst = os.path.join(destination_folder, image)
                with Image.open(src) as img:
                    img = img.resize(image_size)
                    img.save(dst)
            logger.info(f'All Images are resized to (128, 128) and saved in {destination_folder}')
        except Exception as e:
            raise e
        

    def generate_blur_images(self) -> str:
        try:
            source_folder = self.blurimage_config.processed_image_folder
            destination_folder = self.blurimage_config.blur_image_folder
            logger.info("Generating Blur Images for Model Input")
            logger.info('Kernal Size (7, 7)')
            
            # Pillow does not use kernel size explicitly, but the BLUR filter achieves a similar effect.
            for img_name in os.listdir(source_folder):
                img_path = os.path.join(source_folder, img_name)
                img = Image.open(img_path)
                if img is not None:
                    # Apply Gaussian blur using ImageFilter.GaussianBlur
                    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=3.5))  # Adjust the radius to approximate kernel size
                    blurred_img.save(os.path.join(destination_folder, img_name))
            
            logger.info(f'Generated Blur Images and saved in {destination_folder}')
        except Exception as e:
            raise e
        
    # def generate_blur_images(self) ->str:
    #     try:
    #         source_folder = self.blurimage_config.processed_image_folder
    #         destination_folder = self.blurimage_config.blur_image_folder
    #         logger.info("Generating Blur Images for Model Input")
    #         logger.info('Kernal Size (7, 7)')
    #         kernal_size = (7,7)
    #         for img_name in os.listdir(source_folder):
    #             img_path = os.path.join(source_folder, img_name)
    #             img = cv2.imread(img_path)
    #             if img is not None:
    #                 blurred_img = cv2.GaussianBlur(img, kernal_size, 0)
    #                 cv2.imwrite(os.path.join(destination_folder, img_name), blurred_img)
    #         logger.info(f'Generated Blur Images and saved in {destination_folder}')
    #     except Exception as e:
    #         raise e
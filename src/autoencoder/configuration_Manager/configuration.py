from autoencoder.constants import filepath
from autoencoder import logger
from autoencoder.utils.util_functions import read_yaml, create_dir
from autoencoder.entity.entity_config import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = filepath.CONFIG_FILE_PATH,
        param_filepath = filepath.PARAMS_FILE_PATH,
        secret_filepath = filepath.SECRET_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(param_filepath)
        self.secret = read_yaml(secret_filepath)
        # Creating Root folder called artifacts
        create_dir([self.config.root])
    
    def get_download_data_config(self) -> DownloadDataConfig:
        logger.info(f'getting download data configuration')
        config = self.config.data_paths

        logger.info(f'Creating dataset and orignal_image folder Inside artifacts folder')
        create_dir([config.data_directory, config.download_image_folder])


        download_data_config = DownloadDataConfig(
            data_directory = config.data_directory,
            source_gdrive_url = config.source_gdrive_url,
            gdrive_api_key = config.gdrive_api_key,
            download_image_folder = config.download_image_folder
        )

        return download_data_config
    
    def get_preprocess_data_config(self) -> DownloadDataConfig:
        logger.info(f'getting download data configuration')
        config = self.config.data_paths
        logger.info(f'Creating Data Directory Folder')
        create_dir([config.processed_image_folder])

        preprocess_data_config = PreprocessDataConfig(
            download_image_folder = config.download_image_folder,
            processed_image_folder=config.processed_image_folder
        )

        return preprocess_data_config
    
    def get_blurimage_data_config(self) -> PrepareBlurImageConfig:
        logger.info(f'getting download data configuration')
        config = self.config.data_paths
        logger.info(f'Creating Data Directory Folder')
        create_dir([config.blur_image_folder])

        blurimage_data_config = PrepareBlurImageConfig(
            processed_image_folder=config.processed_image_folder,
            blur_image_folder= config.blur_image_folder
        )

        return blurimage_data_config
    
    def get_data_preparation_config(self) -> PrepareInputOutputData:
        config = self.config.data_paths

        input_output_data_config = PrepareInputOutputData(
            processed_image_folder=config.processed_image_folder,
            blur_image_folder= config.blur_image_folder
        )

        return input_output_data_config
    
    def get_model_param_config(self) -> ModelParametersConfig:
        parameters = self.params

        model_param_config = ModelParametersConfig(
            input_shape=tuple(parameters.input_shape),
            batch_size= parameters.batch_size,
            kernel_size= parameters.kernel_size,
            latent_dim=parameters.latent_dim,
            layer_filters=  parameters.layer_filters
        )

        return model_param_config
    
    def get_model_training_config(self) -> TrainingModelConfig:
        config = self.config.model_paths
        params = self.params

        model_training_config = TrainingModelConfig(
            model_dir= config.model_dir,
            HDFmodel_path=config.HDFmodel_path,
            model_path=config.model_path,
            loss=params.loss,
            optimizer=params.optimizer,
            metrics=params.metrics,
            batch_size=params.batch_size,
            epochs=params.epochs,
            model_history_path=config.model_history_path

        )

        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        data_config = self.config.data_paths
        model_config = self.config.model_paths
        param_config = self.params

        create_dir([self.config.data_paths.test_images_path])
        model_evaluation_config = ModelEvaluationConfig(
            HDFmodel_path=model_config.HDFmodel_path,
            model_path=model_config.model_path,
            test_images_path=data_config.test_images_path,
            test_blur_images_path=data_config.test_blur_images_path,
            test_clean_images_path=data_config.test_clean_images_path,
            test_blurimages_source= data_config.test_blurimages_source,
            test_cleanimages_source = data_config.test_cleanimages_source,
            loss=param_config.loss,
            optimizer=param_config.optimizer,
            metrics=param_config.metrics,
            batch_size=param_config.batch_size,
            epochs=param_config.epochs,
            model_history_path=model_config.model_history_path

        )

        return model_evaluation_config
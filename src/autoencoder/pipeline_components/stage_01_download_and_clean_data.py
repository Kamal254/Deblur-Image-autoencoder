from autoencoder.configuration_Manager.configuration import ConfigurationManager

from autoencoder.configuration_Manager.configuration import ConfigurationManager
from autoencoder.components.download_preprocess_data import DataIngestionPreparation
from autoencoder import logger


STAGE_NAME = "Download and Save Processed Data"

class DataDownloadPreprocessPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        download_config = config_manager.get_download_data_config()
        process_config = config_manager.get_preprocess_data_config()
        blurimage_config = config_manager.get_blurimage_data_config()
        downloadandprocess_data = DataIngestionPreparation(download_config, process_config, blurimage_config)
        downloadandprocess_data.download_files_in_parallel()
        downloadandprocess_data.preprocess_data()
        downloadandprocess_data.generate_blur_images()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<')
        obj = DataDownloadPreprocessPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<')
    except Exception as e:
        logger.exception(f'Getting Exception {e}')
        raise e
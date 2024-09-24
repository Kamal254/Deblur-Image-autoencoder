from autoencoder.configuration_Manager.configuration import ConfigurationManager
from autoencoder.components.build_model import ModelBuilding
from autoencoder.components.preprocess_IP_OP_data import PrepareDataForTraining
from autoencoder.components.model_training import TrainModel
from autoencoder import logger


STAGE_NAME = "Preprocess Input Data and Train Model"

class ModelTraining:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_preparation_config = config_manager.get_data_preparation_config()
        preparedata = PrepareDataForTraining(data_preparation_config)
        input_data = preparedata.prepareinputdata()
        output_data = preparedata.prepareoutputdata()

        model_param_config = config_manager.get_model_param_config()
        building_model = ModelBuilding(model_param_config)
        autoencoder = building_model.build_autoencoder()

        model_train_config = config_manager.get_model_training_config()
        training_model = TrainModel(model_train_config)
        training_model.train_model(input_data, output_data, autoencoder)


if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<')
        obj = ModelTraining()
        obj.main()
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<')
    except Exception as e:
        logger.exception(f'Getting Exception {e}')
        raise e
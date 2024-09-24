from autoencoder import logger
from autoencoder.configuration_Manager.configuration import ConfigurationManager
from autoencoder.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Evaluate Model and Tracking Experiment on mlflow"

class EvaluationModel:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(model_eval_config)
        model_evaluation.download_testblurimages()
        model_evaluation.download_testcleanimages()
        model_evaluation.evaluate_model()
        model_evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<')
        obj = EvaluationModel()
        obj.main()
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<')
    except Exception as e:
        logger.exception(f'Getting Exception {e}')
        raise e
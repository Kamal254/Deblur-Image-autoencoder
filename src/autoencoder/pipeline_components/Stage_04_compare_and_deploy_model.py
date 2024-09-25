from autoencoder.components.compare_and_deploy import CompareAndDeployModel
from autoencoder import logger


STAGE_NAME = "Compare and Download the best Model"

class CompareAndDownloadModel:
    def __init__(self):
        pass

    def main(self):
        compare_deploy = CompareAndDeployModel()
        compare_deploy.compare_model()
        compare_deploy.deploy_model()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<')
        obj = CompareAndDownloadModel()
        obj.main()
        logger.info(f'>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<')
    except Exception as e:
        logger.exception(f'Getting Exception {e}')
        raise e
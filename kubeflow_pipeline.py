import kfp
from kfp import dsl
from kfp import kubernetes
from kfp import compiler


@dsl.component(
    base_image="Docker/Image"
)
def download_data():
    from autoencoder.pipeline_components.stage_01_download_and_clean_data import DataDownloadPreprocessPipeline
    downloader = DataDownloadPreprocessPipeline()
    downloader.main()

@dsl.component(
    base_image="Docker/Image"
)
def model_training():
    from autoencoder.pipeline_components.Stage_02_model_training import ModelTraining
    trainer = ModelTraining()
    trainer.main()

@dsl.component(
    base_image="Docker/Image"
)
def model_evaluation():
    from autoencoder.pipeline_components.stage_03_model_evaluation import EvaluationModel
    evaluator = EvaluationModel()
    evaluator.main()


@dsl.component(
    base_image="Docker/Image"
)
def model_evaluation():
    from autoencoder.pipeline_components.stage_03_model_evaluation import EvaluationModel
    evaluator = EvaluationModel()
    evaluator.main()


@dsl.component(
    base_image="Docker/Image"
)
def model_evaluation(mlflow_tracking_uri: str, mlflow_tracking_username: str, mlflow_tracking_password: str):
    from autoencoder.pipeline_components.stage_03_model_evaluation import EvaluationModel
    import os

    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password

    evaluator = EvaluationModel()
    evaluator.main()

@dsl.component(
    base_image="Docker/Image"
)

def model_deployment():
    from autoencoder.pipeline_components.Stage_04_compare_and_deploy_model import CompareAndDownloadModel
    deployer = CompareAndDownloadModel()
    deployer.main()


@dsl.pipeline(
    name="Deblur Image Autoencoder Pipeline",
    description="End to End pipeline for building Autoencoder to Deblur Image"
)

def kfppipeline():
    pvc = kubernetes.CreatePVC(
        pvc_name='my_pvc',
        access_modes=['ReadWriteMany'],
        size='5Gi',
        storage_class_name = 'standard'
    )

    download_task = download_data().set_caching_options(False)
    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/app/artifacts'
    )

    training_task = model_training().set_caching_options(False)
    kubernetes.mount_pvc(
        training_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/app/artifacts'
    )

    evaluation_task = model_evaluation(
        mlflow_tracking_uri='https://dagshub.com/Kamal254/Deblur-Image-autoencoder.mlflow',
        mlflow_tracking_username='Kamal254',
        mlflow_tracking_password='3cbdb442b5873e54a9130d1e83862bb2e7993f55'
    ).set_caching_options(False)
    kubernetes.mount_pvc(
        evaluation_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/app/artifacts'
    )

    deployment_task = model_deployment().set_caching_options(False)
    kubernetes.mount_pvc(
        deployment_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/app/artifacts'
    )


    training_task.after(download_task)
    evaluation_task.after(training_task)
    deployment_task.after(evaluation_task)


if __name__ == '__main__':
    compiler.Compiler().compile(kfppipeline, 'kubeflowpipeline.yaml')
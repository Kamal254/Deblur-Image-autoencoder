import kfp
from kfp import dsl
from kfp import kubernetes
from kfp import compiler
import subprocess



@dsl.component(
    base_image="kamalxs/deblur-image-autoencoder:v1"
)
def download_data():
    import subprocess
    subprocess.run(["pip", "install", "opencv-python"], check=True)
    from autoencoder.pipeline_components.stage_01_download_and_clean_data import DataDownloadPreprocessPipeline
    downloader = DataDownloadPreprocessPipeline()
    downloader.main()

@dsl.component(
    base_image="kamalxs/deblur-image-autoencoder:v1"
)
def model_training():
    from autoencoder.pipeline_components.Stage_02_model_training import ModelTraining
    trainer = ModelTraining()
    trainer.main()

@dsl.component(
    base_image="kamalxs/deblur-image-autoencoder:v1"
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
    base_image="kamalxs/deblur-image-autoencoder:v1"
)

def model_deployment():
    from autoencoder.pipeline_components.Stage_04_compare_and_deploy_model import CompareAndDownloadModel
    deployer = CompareAndDownloadModel()
    deployer.main()


@dsl.component(
    base_image="python:3.9-slim"  # You can use a lightweight image since it's a utility task.
)

def delete_pvc_if_exist():
    import subprocess
    
    try:
        result = subprocess.run(['kubectl', 'get', 'pvc', 'autoencoder-pvc', '-n', 'kubeflow'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode == 0:
            print("PVC 'autoencoder-pvc' already exists.")
            print("Deleting the existing PVC...")
            subprocess.run(['kubectl', 'delete', 'pvc', 'autoencoder-pvc', '-n', 'kubeflow'], check=True)
            print("PVC 'autoencoder-pvc' deleted successfully.")
        else:
            print("PVC 'autoencoder-pvc' does not exist, no need to delete.")

    except subprocess.CalledProcessError as e:
        print(f"Error while checking or deleting PVC: {str(e)}")



@dsl.pipeline(
    name="autoencoder Pipeline",
    description="Kubeflow Pipeline for autoencoder task"
)
def ml_pipeline():

    check_pvc_task = delete_pvc_if_exist().set_caching_options(False)

    pvc1 = kubernetes.CreatePVC(
        pvc_name='autoencoder-pvc',
        access_modes=['ReadWriteMany'],
        size='5Gi',
        storage_class_name='standard'
    )


    download_task = download_data().set_caching_options(False)
    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/app/artifacts',
    )

    training_task = model_training().set_caching_options(False)
    kubernetes.mount_pvc(
        training_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/app/artifacts',
    )

    evaluation_task = model_evaluation(
        mlflow_tracking_uri='https://dagshub.com/Kamal254/Deblur-Image-autoencoder.mlflow',
        mlflow_tracking_username='Kamal254',
        mlflow_tracking_password='3cbdb442b5873e54a9130d1e83862bb2e7993f55'
    ).set_caching_options(False)
    kubernetes.mount_pvc(
        evaluation_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/app/artifacts',
    )

    deployment_task = model_deployment().set_caching_options(False)
    kubernetes.mount_pvc(
        deployment_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/app/artifacts',
    )

    download_task.after(check_pvc_task)
    training_task.after(download_task)
    evaluation_task.after(training_task)
    deployment_task.after(evaluation_task)


if __name__ == '__main__':
    compiler.Compiler().compile(ml_pipeline, 'kubeflowpipeline.yaml')
import mlflow
from mlflow import MlflowClient
import os
import pandas as pd

from autoencoder import logger
from autoencoder.utils.util_functions import create_dir
import mlflow.tensorflow
import tensorflow as tf

os.environ['MLFLOW_TRACKING_USERNAME'] = 'Kamal254'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '3cbdb442b5873e54a9130d1e83862bb2e7993f55'
mlflow.set_tracking_uri("https://dagshub.com/Kamal254/Deblur-Image-autoencoder.mlflow")


class CompareAndDeployModel():
    def __init__(self):
        self.client = MlflowClient()
        self.model_name = "autoencoder"

    def compare_model(self) -> str:
        logger.info("Loading Model Versions and Comparing them")

        # Get current Champion model
        try:
            current_champion_version = self.client.get_model_version_by_alias(
                name=self.model_name,
                alias="Champion"
            )
            current_champion_run = self.client.get_run(current_champion_version.run_id)
            current_champion_accuracy = float(current_champion_run.data.metrics['train_accuracy'])
            current_champion_psnr = float(current_champion_run.data.metrics[' peak signal-to-noise ratio '])
            logger.info(f"Current Champion Model Version: {current_champion_version.version} with Accuracy: {current_champion_accuracy} and PSNR: {current_champion_psnr}")
        except Exception as e:
            logger.warning(f"No current Champion model found. Proceeding without Champion. Error: {e}")
            current_champion_version = None
            current_champion_accuracy = -1  # If no Champion, assign a low value for comparison
            current_champion_psnr = -1

        # Get all model versions
        model_versions = self.client.search_model_versions(f"name='{self.model_name}'")

        model_info = []

        for version in model_versions:
            run_id = version.run_id  # Get the run_id for the version
            run = self.client.get_run(run_id)  # Get the run object
            
            # Check for the required metrics in the run data
            if 'train_accuracy' in run.data.metrics and ' peak signal-to-noise ratio ' in run.data.metrics:
                accuracy = float(run.data.metrics['train_accuracy'])  # Get accuracy
                psnr = float(run.data.metrics[' peak signal-to-noise ratio '])  # Get PSNR

                model_info.append({
                    'version': version.version,
                    'run_id': run_id,
                    'accuracy': accuracy,
                    'psnr': psnr,
                    'stage': version.current_stage
                })

        model_df = pd.DataFrame(model_info)

        # Sort by accuracy first and select the top 3
        top_3_models = model_df.sort_values(by='accuracy', ascending=False).head(3)

        # From the top 3, find the model with the highest PSNR
        best_model = top_3_models.sort_values(by='psnr', ascending=False).iloc[0]

        logger.info(f"Selected New Model Version: {best_model['version']} with PSNR: {best_model['psnr']} and Accuracy: {best_model['accuracy']}")

        # Check if the new model is better than the current champion
        if (best_model['accuracy'] > current_champion_accuracy) or (best_model['psnr'] > current_champion_psnr):
            logger.info(f"New model version {best_model['version']} is better than the current Champion.")

            # If there is a current Champion, tag it as "Old Champion" and remove the "Champion" alias
            if current_champion_version:
                self.client.delete_registered_model_alias(
                    name=self.model_name,
                    alias="Champion"
                )
                self.client.set_registered_model_alias(
                    name=self.model_name,
                    version=current_champion_version.version,
                    alias="Old Champion"
                )
                logger.info(f"Old Champion Model version {current_champion_version.version} has been tagged as 'Old Champion'.")

            # Tag the new model as "Champion"
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=best_model['version'],
                stage="Production"
            )
            self.client.set_registered_model_alias(
                name=self.model_name,
                version=best_model['version'],
                alias="Champion"
            )
            logger.info(f"Model version {best_model['version']} has been moved to Production and tagged as @Champion.")
        else:
            logger.info(f"No changes made. The current Champion model is still the best.")



    def deploy_model(self):
        # Get the champion model version
        champion_model_version = self.client.get_model_version_by_alias(
                                    name=self.model_name,
                                    alias="Champion").version

        if champion_model_version:
            print(f"Champion Model Version: {champion_model_version}")
            
            # Load the TensorFlow model (instead of pyfunc)
            model_uri = f"models:/{self.model_name}/{champion_model_version}"
            loaded_model = mlflow.keras.load_model(model_uri)  # Load the model as TensorFlow

            # Define the folder where you want to save the model
            save_folder = "../../../artifacts/models/best_model"  # e.g., "./saved_model"
            create_dir([save_folder])
            model_path = f'{save_folder}/best_autoencoder.h5'
            loaded_model.save(model_path)
            
            # Save the model using mlflow.tensorflow.save_model
            print(f"Model saved locally in folder: {save_folder}")
        else:
            print("No Champion model found.")
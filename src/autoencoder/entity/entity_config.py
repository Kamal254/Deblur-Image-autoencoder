"""
    This File Holding the Entities(Type of variable used in Components)
    Type of variables.
"""

from dataclasses import dataclass
from pathlib import Path


"""
    Below are the Configuration for Download and Preprocess Data
"""
@dataclass(frozen=True)
class DownloadDataConfig:
    data_directory: Path
    source_gdrive_url: str
    gdrive_api_key : str
    download_image_folder : Path


@dataclass(frozen=True)
class PreprocessDataConfig:
    download_image_folder:Path
    processed_image_folder:Path

@dataclass(frozen=True)
class PrepareBlurImageConfig:
    processed_image_folder:Path
    blur_image_folder:Path

"""
    Below are the Configuration for DataPreparation and Model Training
"""

@dataclass
class PrepareInputOutputData:
    processed_image_folder : Path
    blur_image_folder : Path

@dataclass
class ModelParametersConfig:
    input_shape : tuple
    batch_size : int
    kernel_size : int
    latent_dim : int
    layer_filters : list

@dataclass
class TrainingModelConfig:
    model_dir : Path
    HDFmodel_path : Path
    model_path : Path
    loss:str
    optimizer: str
    metrics:list
    batch_size: int
    epochs:int
    model_history_path :Path




"""
    Below are the Configuration for Model Evaluation and mlflow Experiment Tracking
"""


@dataclass
class ModelEvaluationConfig():
    HDFmodel_path : Path
    model_path : Path
    test_images_path : Path
    test_blur_images_path : Path
    test_clean_images_path : Path
    test_blurimages_source: str
    test_cleanimages_source : str
    loss:str
    optimizer: str
    metrics:list
    batch_size: int
    epochs:int
    model_history_path :Path
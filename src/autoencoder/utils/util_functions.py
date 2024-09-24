import os
import yaml
import json
import joblib
import base64
from box.exceptions import BoxValueError
from box import ConfigBox
from autoencoder import logger
from pathlib import Path
from typing import Any
from ensure import ensure_annotations

# Read yaml Function
@ensure_annotations
def read_yaml(path_of_yaml : Path)-> ConfigBox:
    """Read the Yaml file and return

       Args:
            path_to_yaml (str) : path like input

       Raises:
            ValueError: if yaml file is empty
            e: empty file

        Returns:
            ConfigBox: ConfigBox type

    """

    try:
        with open(path_of_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_of_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

# Create Directories
@ensure_annotations
def create_dir(list_of_directories:list, verbose=True):
    """Create the list of directory in local system

        Args:
            path of directories need to be create
        
        Return:
            Create an Directory at provided location if list_of_directoris not empty
            else return List_of_directories is empty
    """
    if list_of_directories:
        for directory in list_of_directories:
            os.makedirs(directory, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at : {directory}")
    else:
        logger.info("List_of_derectories is empty ")


@ensure_annotations
def save_json(path_to_json: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path_to_json, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path_to_json}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
import json
import pathlib
import torch
from common.constants import *
from common.config import Config
from settings import RUNS_DATA_DIR

def get_model_path(config: Config, base: pathlib.Path = None):
    """Create the path to save the model from config."""
    if base is None: base = RUNS_DATA_DIR
    path = base / config.local_dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_model_or_skip(model, config: Config, filename):
    """save a model with name property to disk."""
    if config.persist != True: return

    path = get_model_path(config)
    torch.save({STATE_DICT: model.state_dict()}, path / f"{filename}.pt")

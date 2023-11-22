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

def save_hparams(config: Config, base = None):
    # Save the object to a file
    # config.model_class = config.model_class.__name__
    base = get_model_path(config, base)
    hparams_path = base / HPARAMS_FILE

    with open(hparams_path, 'w') as f:
        json.dump(config.as_dict(), f, indent=4)

def load_hparams(base: pathlib.Path) -> dict:
    """Returns Config dict if exists, else None."""
    filepath = base / HPARAMS_FILE
    
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def save_model_or_skip(model, config: Config, filename):
    """save a model with name property to disk."""
    if config.persist != True: return

    path = get_model_path(config)
    torch.save({STATE_DICT: model.state_dict()}, path / f"{filename}.pt")

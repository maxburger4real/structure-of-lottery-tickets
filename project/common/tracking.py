
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
import pathlib
import torch
from common import STATE_DICT

HPARAMS_FILE = 'hparams.json'
PROJECT='init-thesis'

ADAM = 'adam'
SGD = 'sgd'
persistance_path = pathlib.Path("runs")

@dataclass
class Config:
    experiment : str
    lr : float
    dataset : str
    training_epochs : int
    pruning_levels : int
    pruning_rate   : float
    prune_weights : bool
    prune_biases : bool
    pruning_strategy : str
    optimizer : str
    momentum : float
    model_shape : list[int]
    model_class : str
    model_seed : int
    data_seed : int
    batch_size : int
    reinit : bool
    persist : bool
    timestamp : str
    device: str
    wandb: bool

    to_dict = asdict

def get_model_path(config: Config, base: pathlib.Path = None):
    """Create the path to save the model from config."""
    if base is None: base = persistance_path
    shape_info = str(config.model_shape).replace(', ','_').replace('[','_').replace(']','')
    name = config.model_class + shape_info
    path = base / name / config.timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_hparams(config: Config, base = None):
    # Save the object to a file
    config.model_class = config.model_class.__name__
    base = get_model_path(config, base)
    hparams_path = base / HPARAMS_FILE
    with open(hparams_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)

def load_hparams(base: pathlib.Path):
    """Returns Config Object if exists, else None."""
    filepath = base / HPARAMS_FILE
    
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        loaded_dict = json.load(f)
        config = Config(**loaded_dict)
    return config

def save_model(model, iteration: int , base: pathlib.Path):
    """save a model with name property to disk."""
    torch.save({STATE_DICT: model.state_dict()}, base / f"{iteration}.pt")

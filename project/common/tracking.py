
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import pathlib
import torch
from common import STATE_DICT


HPARAMS_FILE = 'hparams.json'
PROJECT='init-thesis'

# Optimizers
ADAM = 'Adam'
SGD = 'sgd'
ADAMW = 'AdamW'

# Activations
RELU = 'relu'
SILU = 'silu'

# Loss Functions
MSE = 'mse'
CCE = 'cce'

# Training Pipelines
VANILLA = 'vanilla'
IMP = 'imp'

persistance_path = pathlib.Path("runs")

@dataclass
class Config:
    pipeline : str
    activation : str
    loss_fn : str
    experiment : str
    lr : float
    dataset : str
    training_epochs : int
    model_shape : list[int]
    model_class : str
    model_seed : int
    data_seed : int
    optimizer : str
    persist : bool = True
    timestamp : str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    pruning_levels : int = None
    pruning_rate   : float = None
    prune_weights : bool = None
    prune_biases : bool = None
    pruning_strategy : str = None
    momentum : float = None
    batch_size : int  = None
    reinit : bool  = None
    device: str = 'cpu'

    def as_dict(self):
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}
    

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
    # config.model_class = config.model_class.__name__
    base = get_model_path(config, base)
    hparams_path = base / HPARAMS_FILE
    with open(hparams_path, 'w') as f:
        json.dump(config.as_dict(), f, indent=4)

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

import json
import pathlib
import numpy as np
import torch
from dataclasses import dataclass, asdict
from common.constants import *

persistance_path = pathlib.Path("runs")

@dataclass
class Config:
    model_class : str  # use CLASS.__name__

    dataset : str  # use CONSTANTS
    loss_fn : str  # use CONSTANTS
    pipeline : str   # use CONSTANTS
    activation : str # use CONSTANTS
    optimizer : str  # use CONSTANTS

    lr : float
    training_epochs : int
    model_shape : list[int]
    model_seed : int
    data_seed : int

    # DEFAULTED CONFIGS
    device: str = 'cpu'
    persist : bool = True  # wether to save the model at the checkpoints

    # OPTIONAL CONFIGS
    num_concat_datasets: int = None
    run_name : str = None
    wandb_url : str = None
    run_id : str = None
    local_dir_name: str = None 

    pruning_levels : int = None  # number of times pruning is applied
    pruning_rate   : float = None  # if pruning target is specified, pruning rate is overwritten
    pruning_target : int = None  # if specified, pruning rate is ignored
    params_total  : int = None  # is set when pruning
    params_prunable : int = None  # is set when pruning
    pruning_trajectory : list[int] = None

    prune_weights : bool = None
    prune_biases : bool = None
    reinit : bool  = None  # reinitialize the network after pruning (only IMP)

    l1_lambda : float = None  # lambda for l1 regularisation
    bimt_local : bool = None  # if locality regularisation should be activated
    bimt_swap : int = None  # swap every n-th iteration
    bimt_prune : float = None  # if bimt thresholding in the end is activated.
    momentum : float = None
    batch_size : int  = None  # if None, batch size is dataset size
    description : str = None  # just add information about the idea behind the configuration for documentation purposes.
    pruning_strategy : str = None  # not really in use yet. kindof unnecessary
    early_stopping : bool = False
    early_stop_delta : float = 0.0
    early_stop_patience : int = 1

    def as_dict(self):
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}
    

def get_model_path(config: Config, base: pathlib.Path = None):
    """Create the path to save the model from config."""
    if base is None: base = persistance_path
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

def save_model(model, config: Config, filename):
    """save a model with name property to disk."""
    if config.persist != True: return

    path = get_model_path(config)
    torch.save({STATE_DICT: model.state_dict()}, path / f"{filename}.pt")

def logdict(loss : np.ndarray, prefix):
    """create a loggable dict for wandb"""

    dims = len(loss.shape)
    if dims == 0:
        return {prefix: loss.item()}
    if 0 < dims < 3:
        return {prefix: loss.mean().item()}
    
    if dims == 3:
        metrics = {prefix: loss.mean().item()}

        # assumption: last dimension is task dimension
        all_axis_but_the_last_one = tuple(range(dims-1))
        taskwise_loss = loss.mean(axis=all_axis_but_the_last_one)
        for i, l in enumerate(taskwise_loss):
            metrics[prefix + '_' + str(i)] = l.item()

        return metrics
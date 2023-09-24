
from dataclasses import dataclass, asdict
from typing import List, Optional
PROJECT='init-thesis'

ADAM = 'adam'
SGD = 'sgd'

@dataclass
class Config:
    experiment : str
    lr : float
    dataset : str
    training_epochs : int
    pruning_levels : int
    pruning_rate   : float
    num_layers   : int
    prune_weights : bool
    prune_biases : bool
    pruning_strategy : str
    optimizer : str
    momentum : float
    model_shape : list[int]
    model_seed : int
    data_seed : int
    batch_size : int
    architecture : Optional[str] = None

    to_dict = asdict

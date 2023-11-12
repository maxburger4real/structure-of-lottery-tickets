from dataclasses import dataclass, asdict

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
    pruning_method : str = None  # RANDOM or MAGNITUDE
    early_stopping : bool = False
    early_stop_delta : float = 0.0
    early_stop_patience : int = 1
    loss_cutoff : float = 0.0  # if loss is less than this, stop training early

    # this is whack, but saves a lot of time through being abble to use sweeps
    model_extension: int = 0

    def as_dict(self):
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

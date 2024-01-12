import importlib.util
from dataclasses import dataclass, asdict, fields
from common.constants import *


@dataclass
class Config:
    # MODEL and INITIALIZATION
    model_class : str  # use CLASS.__name__
    dataset : str  # use CONSTANTS
    pipeline : str   # use CONSTANTS
    optimizer : str  # use CONSTANTS

    lr : float
    epochs : int
    model_shape : list[int]

    model_seed : int = 0
    activation : str = RELU # use CONSTANTS
    # DEFAULTED CONFIGS
    scaler: str = None
    factor: float = 0.35
    init_strategy_biases: str = None
    init_strategy_weights: str = None
    init_mean : float = None
    init_std : float = None

    task_description : dict = None
    n_samples : int = 800
    data_seed : int = None
    noise : float = None

    device: str = 'cpu'
    persist : bool = False  # wether to save the model at the checkpoints

    # OPTIONAL CONFIGS
    log_every: int = None
    log_graphs: bool = True
    log_graphs_before_split: bool = False
    stop_on_degradation: bool = True

    num_concat_datasets: int = None
    run_name : str = None
    wandb_url : str = None
    run_id : str = None
    local_dir_name: str = None 

    l1_lambda : float = None  # lambda for l1 regularisation
    bimt_local : bool = None  # if locality regularisation should be activated
    bimt_swap : int = None  # swap every n-th iteration
    bimt_prune : float = None  # if bimt thresholding in the end is activated.
    momentum : float = None
    batch_size : int  = None  # if None, batch size is dataset size
    description : str = None  # just add information about the idea behind the configuration for documentation purposes.

    # Early stopping
    early_stop_delta : float = 0.0
    early_stop_patience : int = None
    loss_cutoff : float = None  # if loss is less than this, stop training early
    
    # Pruning
    pruning_method : str = None  # RANDOM or MAGNITUDE
    pruning_scope : str = None  # GLOBAL or LAYERWISE
    extension_levels: int = 0 # how many 
    pruning_levels : int = None  # number of times pruning is applied
    pruning_rate   : float = None  # if pruning target is specified, pruning rate is overwritten
    pruning_target : int = None  # if specified, pruning rate is ignored
    prune_weights : bool = None
    prune_biases : bool = None

    # Set by program
    params_before_pruning : int = None # params in the beginning param_trajectory[0]
    params_prunable : int = None  # is set when pruning
    params_total  : int = None  # is set when pruning
    pruning_trajectory : list[int] = None
    param_trajectory : list[int] = None
    reinit : bool  = None  # reinitialize the network after pruning (only IMP)

    base_model_shape: str = None # if extenion overrides model shape, this is original

    def as_dict(self):
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    def __post_init__(self):

        if isinstance(self.model_shape, str):
            dims = self.model_shape.split('_')
            hidden_dims = [int(d) for d in dims]
            inputs, outputs = tuple(zip(*self.dataset.value.values()))
            input_dim, output_dim = sum(inputs), sum(outputs)
            self.model_shape = [input_dim] + hidden_dims + [output_dim]
        
        # TODO: change the whole task_description implementation to Ordered Dict.
        self.task_description = [(k, v) for k, v in self.dataset.value.items()]

        for field in fields(self):
            value = getattr(self, field.name)
            
            # transform enums to their names
            if isinstance(value, Enum):
                setattr(self, field.name, value.name)
            
            if isinstance(value, type):
                setattr(self, field.name, value.__name__)

def import_config(filename):
    """Import a file, used for config."""
    if filename is None: return (None, None)

    # Remove the '.py' extension from the filename
    module_name = filename.replace('.py', '')

    # Create a module spec from the filename
    spec = importlib.util.spec_from_file_location(module_name, filename)

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    run_config = getattr(module, 'run_config', None)
    sweep_config = getattr(module, 'sweep_config', None)
    
    return run_config, sweep_config
from common.models import InitMLP
from common.config import Config
from common.constants import *

m = 2

run_config = Config(
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,
    dataset=CONCAT_MOONS,
    num_concat_datasets=m,
    
    model_shape=[2*m, 40, 40, 1*m],
    model_class = InitMLP.__name__,

    # training
    lr=0.1,
    optimizer=ADAM,
    training_epochs=100,
    log_every_n_epochs = 10,  # insert int to log every n epochs

    # seeds
    model_seed=42,
    data_seed=0,

    persist=False,

    # early stop
    early_stopping=True,
    early_stop_patience=20,
    early_stop_delta=0.0,

    extension_levels=2,

    # pruning
    prune_biases=True,
    prune_weights=True,
    pruning_target=80,
    pruning_levels=10,
    pruning_method=MAGNITUDE,
    reinit=True,
)

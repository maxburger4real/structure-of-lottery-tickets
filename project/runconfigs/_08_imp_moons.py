from common.architectures import SimpleMLP
from common.config import Config
from common.constants import *

m = 2

run_config = Config(
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,
    dataset=CONCAT_MOONS,
    num_concat_datasets=m,
    
    model_shape=[m*2, 40, 40, m],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.001,
    optimizer=ADAM,
    training_epochs= 1000, #3000,

    # seeds
    model_seed=2,
    data_seed=0,

    persist=True,

    # early stop
    early_stopping=True,
    early_stop_patience=10,
    early_stop_delta=0.0,

    # pruning
    prune_biases=True,
    prune_weights=True,
    pruning_target=30,
    pruning_levels=30,
    reinit=True
)

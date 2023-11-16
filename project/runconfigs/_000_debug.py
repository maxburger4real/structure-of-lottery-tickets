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
    
    model_shape=[2*m, 40, 40, 1*m],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.01,
    optimizer=ADAM,
    training_epochs=30,

    # seeds
    model_seed=42,
    data_seed=0,

    persist=True,

    # early stop
    early_stopping=True,
    early_stop_patience=20,
    early_stop_delta=0.0,

    extension_levels=2,

    # pruning
    prune_biases=True,
    prune_weights=True,
    pruning_target=30,
    pruning_levels=30,
    pruning_method=MAGNITUDE,
    reinit=True,

)

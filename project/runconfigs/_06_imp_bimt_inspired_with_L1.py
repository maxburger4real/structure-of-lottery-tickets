from common.models import SimpleMLP
from common.datasets.independence import INPUT_DIM, OUTPUT_DIM, DATASET_NAME
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=IMP,
    activation=SILU,
    loss_fn=MSE,
    dataset=DATASET_NAME,
    
    model_shape=[INPUT_DIM, 20, 20, OUTPUT_DIM],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.002,
    optimizer=ADAM,
    training_epochs=3000,
    l1_lambda=0.001,

    # seeds
    model_seed=0,
    data_seed=0,

    # lottery
    # pruning
    pruning_levels=30,
    pruning_rate=0.1,
    pruning_strategy='global',
    prune_weights=True,
    prune_biases=False,
    reinit=True,
)

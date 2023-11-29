from common.models import SimpleMLP
from common.datasets.independence import INPUT_DIM, OUTPUT_DIM, DATASET_NAME
from common.config import Config
from common.constants import *

run_config = Config(
        pipeline=IMP,
        activation=RELU,
        loss_fn=MSE,
        description=f'current best',
        dataset=DATASET_NAME,
        model_shape=[INPUT_DIM, 20, 20, OUTPUT_DIM],
        model_class = SimpleMLP.__name__,

        # pruning
        pruning_levels=30,
        pruning_rate=0.1,
        pruning_strategy='global',
        prune_weights=True,
        prune_biases=False,

        # training
        epochs=500,
        lr=0.0033,
        momentum=0,
        optimizer=ADAM,

        # seeds
        model_seed=0,
        data_seed=0,

        # lottery
        reinit=True,
    )

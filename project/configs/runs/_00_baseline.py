from common.architectures import SimpleMLP
from common.datasets.independence import INPUT_DIM, OUTPUT_DIM, DATASET_NAME
from common.tracking import Config
from common.constants import *

run_config = Config(
        pipeline=IMP,
        activation=RELU,
        loss_fn=MSE,
        description=f'IMP-reinit-with nograd',
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
        training_epochs=1500,
        lr=0.001,
        momentum=0,
        optimizer=ADAM,

        # seeds
        model_seed=4,
        data_seed=2,

        # lottery
        reinit=True,
    )

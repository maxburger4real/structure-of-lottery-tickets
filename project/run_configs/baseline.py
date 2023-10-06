
from datetime import datetime

from common.architectures import SimpleMLP
from common.datasets.independence import INPUT_DIM, OUTPUT_DIM, DATASET_NAME
from common.tracking import Config, SGD, ADAM

config = Config(
    experiment=f'IMP-reinit-with nograd',
    dataset=DATASET_NAME,
    model_shape=[INPUT_DIM, 20, 20, OUTPUT_DIM],
    model_class = SimpleMLP,

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
    batch_size = None,

    # seeds
    model_seed=4,
    data_seed=2,

    # lottery
    reinit=True,

    # storage
    persist=True,
    timestamp=datetime.now().strftime("%Y_%m_%d_%H%M%S"),
    device='cpu',
    wandb=True, # does this even make any sense
)

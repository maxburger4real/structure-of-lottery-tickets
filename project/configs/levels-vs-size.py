"""
goyyst73 seed 0
ui53he8j seed 1, 2, 3
tqrtw8se seed 0, 1, 2, 3
"""

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.models import Init
from training.datasets import Datasets, Scalers
from training.routines import Routines

description = """

"""

seeds = [0]
seeds = [1, 2, 3]
seeds = [0, 1, 2, 3]

model_shapes=[
    [4, 31, 31, 2],
    [4, 38, 38, 2],
    [4, 48, 48, 2],
    [4, 57, 57, 2],
]

pruning_levels = [5, 6, 7, 8, 9, 10]
pruning_levels = [3, 4]
pruning_levels = [1, 2]

sweep_config = {
    "method": "grid",
    "description": description,
    "name": str(__name__),
    "parameters": {
        "model_shape": {"values": model_shapes},
        "pruning_levels": {"values": pruning_levels},
        "model_seed": {"values": seeds},
        "pruning_rate" : {"values": [None]}
    },
}


run_config = Config(
    # sweeped
    model_shape=None,
    pruning_levels=None,
    model_seed=None,
    pruning_rate=None,

    pipeline=Routines.imp,
    dataset=Datasets.CIRCLES_MOONS,
    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,

    # training
    lr=0.001,
    optimizer="adam",
    epochs=3000,
    batch_size=64,
    # seeds
    data_seed=0,
    persist=False,
    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,
    # pruning
    pruning_method="magnitude",
    pruning_scope="global",
    prune_biases=False,
    prune_weights=True,
    pruning_target=112,
    reinit=True,

    # newly added
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
    n_samples=1000,
    noise=0.1,
)

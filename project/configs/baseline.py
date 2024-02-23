"""
sweep_id : wpcowdl5
seeds = [0]

levels 0-19
sweep_id: j9lvvxjg
[1, 2, 3]

levels 20-25
sweep_id: a5kr3muw
[1, 2, 3]
"""

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.models import Init
from training.datasets import Datasets, Scalers
from training.routines import Routines

description = """
"""

seeds = [1, 2, 3]

sweep_config = {
    "method": "grid",
    "description": description,
    "name": str(__name__),
    "parameters": {
        "model_seed": {"values": seeds},
    },
}


run_config = Config(

    model_shape=[4, 8, 8, 2],
    pipeline=Routines.vanilla,
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
    pruning_rate=0.32,
    pruning_levels=0,
    reinit=True,

    # newly added
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
    n_samples=1000,
    noise=0.1,
)

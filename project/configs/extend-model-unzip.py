"""
model shape: 4 4 4 4 2
"""

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.models import Init
from training.datasets import Datasets, Scalers
from training.routines import Routines

description = """
TODO
"""

import os
os.environ["WANDB_SILENT"]="true"

# These are the networks that did split
seeds = [3]
extension_levels = [10, 11, 13, 14, 16, 17, 18]
extension_levels = [10]
extension_levels = [18]
extension_levels = [11, 13, 14, 16, 17]

sweep_config = {
    "method": "grid",
    "description": description,
    "name": str(__name__),
    "parameters": {
        "extension_levels": {"values": extension_levels},
        "model_seed": {"values": seeds},
    },
}

run_config = Config(

    # sweeped
    model_shape=[4, 4, 4, 4, 2],  # is overwritten
    base_model_shape=[4, 4, 4, 4, 2],

    stop_on_seperation = True,

    pipeline=Routines.imp,
    dataset=Datasets.CIRCLES_MOONS,
    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,
    data_seed=0,

    # training
    lr=0.001,
    optimizer="adam",
    epochs=3000,
    batch_size=64,

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

"""
model shape: 4 8 8 2
sweep_id : wpcowdl5
seeds = [0]

levels 0-19
sweep_id: j9lvvxjg
[1, 2, 3]

levels 20-25
sweep_id: a5kr3muw
[1, 2, 3]

--------
model shape: 4 4 4 4 2
with different model shape

123 t1mp5nhi
45 c3gin27l
12345 82wq0l59
vk3ftk2q

try again, but with 4 6 6 6 2 from 0 to 15

-------
model shape: 4, 20, 2
01ayiisq
"""

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.models import Init
from training.datasets import Datasets, Scalers
from training.routines import Routines

description = """
This experiment should show behaviour of splitting when starting with a small network,
namely [4, 8, 8, 2] and extending it up to 20 times with a pruning rate of 0.32

This pruning rate is derived from a well working experiment with a network of size 
[4, 320, 320, 2], which is pruned 20 times to 50 parameters. It yielded good splitting behaviour and had 
this pruning rate.
"""

extension_levels = list(range(20))
extension_levels = list(range(20, 26))
extension_levels = list(range(7, 13))
extension_levels = list(range(7))
extension_levels = list(range(13, 15))
extension_levels = list(range(0, 15))
extension_levels = list(range(15, 17))


seeds = [1]
seeds = [0]
seeds = [1, 2, 3]
seeds = [1, 2, 3, 4, 5]

sweep_config = {
    "method": "grid",
    "description": description,
    "name": str(__name__),
    "parameters": {
        "extension_levels": {"values": extension_levels},
        "model_seed": {"values": seeds},
        #"model_shape" : {"values": [[4, 4, 4, 4, 2]]}
    },
}

run_config = Config(

    # sweeped
    model_shape=[4, 4, 4, 4, 2],  # is overwritten
    base_model_shape=[4, 4, 4, 4, 2],
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

from training.models import Init
from training.datasets import Datasets, Scalers

num_hidden = 2
shapes = [[4] + num_hidden * [h] + [2] for h in [160, 320]]
seeds = [7, 8, 9]

scalers = [Scalers.StandardUnitVariance, None]
datasets = [Datasets.CIRCLES_AND_MOONS.name, Datasets.OLD_MOONS.name]
bias_inits = [Init.zero, Init.DEFAULT.name]

sweep_config = {
    "method": "grid",
    "name": "Checkout if, how and when it splits vs performance with different bias init and pruning.",
    "parameters": {
        "model_shape": {"values": shapes},
        "model_seed": {"values": seeds},
        "scaler": {"values": scalers},
        "dataset": {"values": datasets},
        "prune_biases": {"values": [True, False]},
        "init_strategy_biases": {"values": bias_inits},
    },
}

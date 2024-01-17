from training.models import Init
from training.datasets import Scalers

num_hidden = 2
shapes = [[4] + num_hidden * [h] + [2] for h in [320]]
seeds = [7, 8, 9]
scaler = [Scalers.StandardUnitVariance, None]

sweep_config = {
    "method": "grid",
    "name": "Checkout if, how and when it splits vs performance with different bias init and pruning.",
    "parameters": {
        "model_shape": {"values": shapes},
        "model_seed": {"values": seeds},
        "scaler": {"values": scaler},
        "prune_biases": {"values": [True, False]},
        "init_strategy_biases": {"values": [Init.zero, Init.DEFAULT]},
    },
}

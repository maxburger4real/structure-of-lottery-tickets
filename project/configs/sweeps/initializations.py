"""Simply run the configuration with multiple seeds to see stability."""

from utils.constants import Init

seeds = list(range(10, 15))
seeds = list(range(10))

seeds = list(range(100, 110))

init_strategies = (
    Init.kaiming_normal,
    Init.DEFAULT.name,
)

sweep_config = {
    "method": "grid",
    "name": "Test initialization for weights with new early stopping.",
    "parameters": {
        "init_strategy_weights": {"values": init_strategies},
        "model_seed": {"values": seeds},
    },
}

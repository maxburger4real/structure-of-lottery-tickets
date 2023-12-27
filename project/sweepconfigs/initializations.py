"""Simply run the configuration with multiple seeds to see stability."""

from common.constants import InitializationStrategy
seeds = list(range(10,15))
seeds = list(range(10))

seeds = list(range(100,110))

init_strategies = (
    InitializationStrategy.KAIMING_NORMAL.name,
    InitializationStrategy.DEFAULT.name,
)

sweep_config = {
    'method': 'grid', 
    'name': f'Test initialization for weights with new early stopping.',
    'parameters':{
        "init_strategy_weights":  {"values": init_strategies},
        "model_seed":  { "values": seeds },
        }
    }
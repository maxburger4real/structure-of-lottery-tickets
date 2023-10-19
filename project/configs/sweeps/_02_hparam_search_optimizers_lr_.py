"""Simply run the configuration with multiple seeds to see stability."""
from training_pipelines.imp import BEST_EVAL
from common.tracking import ADAM, ADAMW, SGD

sweep_config = {
    'method': 'bayes', 
    'name': 'compare optimizers w/ different learning rates',
    'metric': {
        'name': BEST_EVAL,
        'goal': 'minimize' 
    },
    'parameters':{
        "lr": {"max": 0.1, "min": 0.001},
        "optimizer" : {"values": [ADAM, ADAMW, SGD]},
        }
    }
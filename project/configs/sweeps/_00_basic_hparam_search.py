from common.tracking import PROJECT
from training_pipelines.imp import BEST_EVAL

config = {
    'project': PROJECT,
    'method': 'random', # random, grid, bayes
    'name': 'my-first-real-sweep',
    'metric': {
        'name': BEST_EVAL,
        'goal': 'minimize' 
    },
    'parameters':{
        "training_epochs": {"values": [500, 1000, 1500, 2000]},
        "lr": {"max": 0.07, "min": 0.0001},
        }
    }
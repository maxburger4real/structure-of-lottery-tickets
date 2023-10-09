from common.tracking import PROJECT
from training_pipelines.imp import BEST_EVAL

config = {
    'project': PROJECT,
    'method': 'grid', # random, grid, bayes
    'name': 'my-first-real-sweep',
    'metric': {
        'name': BEST_EVAL,
        'goal': 'minimize' 
    },
    'parameters':{
        "training_epochs": {"values": [500, 1000, 1500, 2000]},
        "model_seed":  {"values": [0,1,2,42]},
        }
    }
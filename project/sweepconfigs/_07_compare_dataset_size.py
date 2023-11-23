"""72 Runs"""
from common.constants import *

sweep_config = {
    'method': 'grid', 
    'name': 'different dataset sizes',
    'parameters':{
        "n_samples" : {
            "values": [400,800]
        },
        "model_shape": {"values" : [
            [4,20,20,2],    [4,40,40,2],
            [4,60,60,2],    [4,80,80,2],]}
        # "pruning_method":  { "values": [RANDOM, MAGNITUDE]},
        # "prune_biases" : {  'values' : [ True, False ]        }
    }
}
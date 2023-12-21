"""72 Runs"""
from common.constants import *
m=2
sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "extension_levels" : {
            "values": [0,1]
        },
        # "pruning_method":  { "values": [RANDOM, MAGNITUDE]},
        # "prune_biases" : {  'values' : [ True, False ]        }
    }
}
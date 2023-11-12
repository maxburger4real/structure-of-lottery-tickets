"""72 Runs"""
from common.constants import *
m = 2
sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "model_shape" : {
            "values":[
                [m*2, 40, 40, m],
                [m*2, 80, 80, m],
                [m*2, 160, 160, m],
                [m*2, 320, 320, m],
                [m*2, 640, 640, m],
            ]
        },
        "model_seed":  {"values": [7, 9, 11]},
        "pruning_method":  { "values": [RANDOM, MAGNITUDE]},
        #"prune_bias" : { 'values' : [True, False]}
    }
}
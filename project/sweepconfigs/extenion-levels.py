"""72 Runs"""
from common.constants import *

extension_levels = list(range(13,21))
seeds = [0, 1, 2]
seeds = [0]

sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "extension_levels" : { "values": extension_levels },
        "model_seed":  { "values": seeds },
    }
}
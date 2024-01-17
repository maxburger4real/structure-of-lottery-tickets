"""72 Runs"""
from utils.constants import *

extension_levels = list(range(0,5))
extension_levels = list(range(5,10))
seeds = [0, 1, 2]
seeds = [123]

sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "extension_levels" : { "values": extension_levels },
        "model_seed":  { "values": seeds },
    }
}
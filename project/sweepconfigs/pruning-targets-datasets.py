from common.constants import *

seeds = [0, 1, 2, 3]
targets = [20,30,50]
datasets = [MOONS, CIRCLES]
sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "dataset" : { "values": datasets },
        "pruning_target" : { "values": targets },
        "model_seed":  { "values": seeds },
    }
}
from utils.constants import *


shapes = [
    #[4],
    #[8],
    #[16],
    #[32],
    #[64],
    #[4, 4],
    #[8, 8],
    #[16, 16],
    [32, 32],
    #[4, 4, 4],
    #[8, 8, 8],
    [16, 16, 16],
]

shapes = [ [2] + hidden + [1] for hidden in shapes  ]

seeds = [0, 1, 2, 3]
seeds = [0,2]
sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "model_shape" : { "values": shapes },
        "model_seed":  { "values": seeds },
    }
}
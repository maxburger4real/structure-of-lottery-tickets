from common.constants import *

num_hidden = 2
hidden_dims = [320]
shapes = [[4] + num_hidden * [h] + [2] for h in hidden_dims]
seeds = [7, 8, 9]
batch_sizes = [64, 128, 256]

sweep_config = {
    'method': 'grid', 
    'name': 'See if batch size changes a lot',
    'parameters':{
        "model_seed":  {"values": seeds},
        "batch_size":  {"values": batch_sizes}
    }
}
from utils.constants import *

num_hidden = 2
hidden_dims = [320]
shapes = [[4] + num_hidden * [h] + [2] for h in hidden_dims]
seeds = [7, 8, 9]
cutoffs = [0.01, None]

sweep_config = {
    'method': 'grid', 
    'name': 'See if batch size changes a lot',
    'parameters':{
        "model_seed":  {"values": seeds},
        "loss_cutoff":  {"values": cutoffs}
    }
}
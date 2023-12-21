num_hidden = 2
shapes = [[4] + num_hidden * h + [2] for h in  [40, 80, 160, 320, 640]]
seeds = [7]
noises = [0.0, 0.1, 0.2]

sweep_config = {
    'method': 'grid', 
    'name': 'Network Splitting of different sizes with noise 0 and .1',
    'parameters':{
        "model_shape" : { "values": shapes },
        "model_seed":  { "values": seeds },
        "noise": {"values": noises }
    }
}
import numpy as np
from torch import nn
import torch

class BaseModel(nn.Module):
    """A mini mlp for demo purposes."""
    def __init__(self, shape: torch.Size, init_bias_zero=False):
        super().__init__()

        linear = nn.Linear(shape[0], shape[1])
        nn.init.ones_(linear.bias)
        if init_bias_zero:
            nn.init.zeros_(linear.bias)

        nn.init.uniform_(linear.weight)
        modules = [linear]

        for i in range(1, len(shape) - 1):
            modules.append(nn.ReLU())
            in_dim = shape[i]
            out_dim = shape[i+1]
            linear = nn.Linear(in_dim, out_dim)
            nn.init.uniform_(linear.weight)
            nn.init.ones_(linear.bias)
            if init_bias_zero:
                nn.init.zeros_(linear.bias)
            
            modules.append(linear)

        self.modules = modules
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        y = self.model(x)
        return y
    
def print_bias_pruned(model, layer=2):
    l = list(np.array([b.item() for b in model.modules[layer].bias]) == 0)
    true_indices = [index for index, value in enumerate(l) if value]
    print(true_indices)

def disconnected_weights(task_size, num_tasks):
    '''create an identity matrix that is enlarged for the task in and output size.'''
    a = np.identity(num_tasks)
    tin, tout = task_size
    return a.repeat(tin, axis=0).repeat(tout, axis=1)

def make_splitable_model(model, num_tasks, tile_offset=0.1):
    '''tile_offset is 0, the model is already split in the beginning.'''
    with torch.no_grad():
        for m in model.layers:
            if not isinstance(m, torch.nn.Linear): continue
            _out = m.weight.shape[0] / num_tasks
            _in = m.weight.shape[1] / num_tasks

            #tiling = disconnected_weights(task_size, num_tasks).astype(np.float32) + tile_offset
            tiling = disconnected_weights((_out, _in), num_tasks).astype(np.float32) * 0.5 + tile_offset
            m.weight = nn.Parameter(torch.rand_like(m.weight) * tiling)

    return model

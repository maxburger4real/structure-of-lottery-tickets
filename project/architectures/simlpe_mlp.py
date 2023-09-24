import torch
import torch.nn as nn
from architectures import ReproducibleModel
from datasets import symbolic_1

class SimpleMLP(ReproducibleModel):
    """A mini mlp for demo purposes."""
    def __init__(self, weight_shape: torch.Size, Activation=nn.ReLU ,seed=None):
        super(SimpleMLP, self).__init__(seed)
        self.name = SimpleMLP.__name__ + str(weight_shape).replace(', ','_').replace('[','_').replace(']','')

        modules = []
        modules.append(nn.Linear(weight_shape[0], weight_shape[1]))

        for i in range(1, len(weight_shape) - 1):
            modules.append(Activation())
            in_dim = weight_shape[i]
            out_dim = weight_shape[i+1]
            modules.append(nn.Linear(in_dim, out_dim))

        self.modules = modules
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        y = self.model(x)
        return y

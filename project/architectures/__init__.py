"""
This module contains the MLP Architectures for reproducibility.
For reproducibility reasons, the *code* for each model architecture must be
saved. Otherwise it can not be recreated from state_dicts.
"""

import torch.nn as nn
from common import torch_utils

class ReproducibleModel(nn.Module):
    """An abstract class that sets the seed for reproducible initialization."""
    def __init__(self, seed=None):
        super(ReproducibleModel, self).__init__()
        if seed is None: seed = torch_utils.SEED
        torch_utils.set_seed(seed)

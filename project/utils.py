import torch
import numpy as np
import random

SEED = 64

def set_seed(seed):
    """Sets all seeds of randomness sources"""
    # TODO: set MPS seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def check_if_models_equal(model1, model2):
    """Returns True if both models are equal, otherwise False"""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
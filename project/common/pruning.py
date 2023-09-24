import copy
from typing import List

import torch
import torch.nn.utils.prune as prune

from common import torch_utils

def convert_to_pruning_model(modules: List[torch.nn.Module], prune_weights: bool, prune_biases: bool) -> List[torch.nn.Module]:
    """
    Before training, convert the model parameters to the pytorch pruning-style
    parameters with _orig and _mask suffix. This is achieved by running global pruning, but prune
    0 parameters. This transforms the parameter structure.

    returns: parameters to prune
    """
    # filter out non-trainable modules
    modules = [m for m in modules if torch_utils.module_is_trainable(m)]

    parameters_to_prune = []
    if prune_weights:
        parameters_to_prune.extend([(module, 'weight') for module in modules])
    if prune_biases:
        parameters_to_prune.extend([(module, 'bias') for module in modules])

    if len(parameters_to_prune) == 0:
        print("no pruning")

    prune.global_unstructured(
            parameters_to_prune,
            prune.L1Unstructured,
            amount=0,
        )
    
    return parameters_to_prune

def get_model_state_dict(model, drop_masks=True):
    """
    Retrieve a true copy of the model state dict at an arbitrary time point
    with or without masks
    """
    # deepcopy is necessary, otherwise the state_dict just holds a reference to the modules
    state_dict =  copy.deepcopy(model.state_dict())

    if drop_masks:
        # creates a copy
        state_dict = {k: v for k, v in state_dict.items() if "_mask" not in k}
    
    return state_dict

def global_magnitude_pruning(params_to_prune: List[torch.nn.Module], pruning_rate):
    """Perform global pruning by magnitude of provided modules and parameters."""
    prune.global_unstructured(params_to_prune, prune.L1Unstructured, amount=pruning_rate)

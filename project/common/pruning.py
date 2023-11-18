import copy
import torch
import torch.nn.utils.prune as prune
import numpy as np
from common.config import Config
from common.torch_utils import module_is_trainable
from common.constants import MAGNITUDE, RANDOM

def build_reinit_func(model: torch.nn.Module):
    """
    Returns a Closure that reinitializes the model on call.
    """
    # deepcopy to kill off references to the model parameters
    state_dict =  copy.deepcopy(model.state_dict())

    # remove masks, because they shouldnt be reinitialized
    state_dict = {k: v for k, v in state_dict.items() if "_mask" not in k}
    

    def reinitialize(model):
        """Closure that contains the reinit state_dict to reinitialize any model."""
        model.load_state_dict(state_dict, strict=False)

    return reinitialize

def build_pruning_func(model: torch.nn.Module, config: Config):
    """
    Returns a Closure that prunes the model on call.
    """
    if config.pruning_method == MAGNITUDE: pruning_method = prune.L1Unstructured
    elif config.pruning_method == RANDOM: pruning_method = prune.RandomUnstructured
    else: raise ValueError('must specify pruning method')

    # convert to the pruning parameterization
    params = _extract_pruning_params(model, config)
    prune.global_unstructured(params, prune.Identity)
    
    def pruning_func(amount):
        """A Closure that contains the trajectory Generator and the pruning parameters."""

        prune.global_unstructured(
            parameters=params, 
            pruning_method=pruning_method, 
            amount=amount
        )
        tensors = [getattr(module, name) for module, name in params]
        min_magnitude = min(T[T != 0].abs().min() for T in tensors if T[T != 0].numel() > 0).item()

        return min_magnitude
    
    return pruning_func, config.pruning_trajectory

def _extract_pruning_params(model, config: Config):
    """Returns the parameters which are selected for pruning based on the Config."""

    if config.prune_weights is None or config.prune_biases is None : raise ValueError

    # filter out non-trainable modules
    modules = [m for m in model.modules if module_is_trainable(m)]

    params = []
    if config.prune_weights: params.extend([(module, 'weight') for module in modules])
    if config.prune_biases: params.extend([(module, 'bias') for module in modules])
    if len(params) == 0: raise ValueError

    return params

def count_trainable_and_prunable_params(model):
    trainable = _count_trainable_params(model)
    prunable = _count_prunable_params(model)
    return trainable, prunable

def _count_prunable_params(model):
    """Counts the number of weights and biases that are prunable with pytorch, meaning they have _mask """
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name:
                    total += torch.numel(buffer)

                if "bias_mask" in buffer_name:
                    total += torch.numel(buffer)

    return total

def _count_trainable_params(model):
    """Counts the number of weights and biases that require grad, hence are trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
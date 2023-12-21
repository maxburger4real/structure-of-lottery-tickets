import copy
import torch
import torch.nn.utils.prune as prune
from common.config import Config
from common.torch_utils import module_is_trainable
from common.constants import *

# visible
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
    params = _extract_pruning_params(model, config.prune_weights, config.prune_biases)
    prune.global_unstructured(params, prune.Identity)
    
    if config.pruning_scope == LAYERWISE:
        if config.prune_biases is not False: raise NotImplementedError('Cannot prune biases with Layerwise pruning')
        tensors = [getattr(module, name) for module, name in params]
        layerwise_params = torch.Tensor([t.numel() for t in tensors])
        layerwise_percentages = layerwise_params / layerwise_params.sum()
        def layerwise_pruning_func(iteration_amount):
            """A Closure that contains the trajectory Generator and the pruning parameters."""
            amounts = torch.round(layerwise_percentages * iteration_amount).int()
            while sum(amounts) != iteration_amount:
                if sum(amounts) < iteration_amount:
                    amounts[torch.argmax(amounts)] += 1
                elif sum(amounts) > iteration_amount:
                    amounts[torch.argmax(amounts)] -= 1         

            for (module, name), layer_amount in zip(params, amounts.tolist()):
                prune.l1_unstructured(module, name, layer_amount)

            tensors = [getattr(module, name) for module, name in params]
            min_magnitude = min(T[T != 0].abs().min() for T in tensors if T[T != 0].numel() > 0).item()
            return min_magnitude
        
        return layerwise_pruning_func
    
    if config.pruning_scope == GLOBAL:
        def global_pruning_func(amount):
            """A Closure that contains the trajectory Generator and the pruning parameters."""

            prune.global_unstructured(
                parameters=params, 
                pruning_method=pruning_method, 
                amount=amount
            )
            tensors = [getattr(module, name) for module, name in params]
            min_magnitude = min(T[T != 0].abs().min() for T in tensors if T[T != 0].numel() > 0).item()
            return min_magnitude
        
        return global_pruning_func
    
    
    raise ValueError(f'Pruning scope not supported : {config.pruning_scope}')
    

def count_prunable_params(model):
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


# helpers
def _extract_pruning_params(model, prune_weights: bool, prune_biases: bool):
    """Returns the parameters which are selected for pruning based on the Config."""

    if prune_weights is None or prune_biases is None : raise ValueError

    # filter out non-trainable modules
    modules = [m for m in model.modules if module_is_trainable(m)]

    params = []
    if prune_weights: params.extend([(module, 'weight') for module in modules])
    if prune_biases: params.extend([(module, 'bias') for module in modules])
    if len(params) == 0: raise ValueError

    return params

import copy
import torch
import torch.nn.utils.prune as prune
import numpy as np
from common.tracking import Config
from common.torch_utils import module_is_trainable

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
    config.params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # convert to the pruning parameterization
    params = _extract_pruning_params(model, config)
    prune.global_unstructured(params, prune.Identity)
    config.params_prunable = _count_prunable_params(model)

    trajectory = _build_prune_trajectory(config)
    trajectory_generator = (y for y in trajectory)
    
    def pruning_func():
        """A Closure that contains the trajectory Generator and the pruning parameters."""

        amount = next(trajectory_generator)
        prune.global_unstructured(
            parameters=params, 
            pruning_method=prune.L1Unstructured, 
            amount=amount
        )
        return amount

    return pruning_func

def _build_prune_trajectory(config: Config) -> np.ndarray:
    """
    Return a ndarray of length pruning_levels with the 
    number of parameters to prune every level.
    """
    T = config.pruning_levels
    if T is None: raise ValueError('Must specify pruning_levels')
    
    # override pruning rate if target is specified
    if config.pruning_target is not None:
        N0 = config.params_prunable
        NT = config.pruning_target
        config.pruning_rate = 1 - (NT / N0) ** (1 / T)

    # if not overridden and not specified
    if config.pruning_rate is None:
        raise ValueError('Pruning not specified')
    
    pr = config.pruning_rate
    t = torch.arange(0, T+1)
    param_trajectory = np.rint((N0 * (1 - pr) ** t).numpy()).astype(int)
    prune_trajectory = -np.diff(param_trajectory)

    return prune_trajectory

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

def _count_prunable_params(model):
    
    total = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name:
                    total += torch.numel(buffer)

                if "bias_mask" in buffer_name:
                    total += torch.numel(buffer)

    return total
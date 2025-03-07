"""
Pruning functions.
TODO:
[ ] remove the config dependency.
[ ] relocate 'update_pruning_config'
"""

import copy
import torch
import torch.nn.utils.prune as prune
import numpy as np
from sympy import symbols, Eq, solve
from typing import List, Tuple

from training.config import Config
from utils.torch_utils import module_is_trainable
from training.routines import Routines


# visible
def build_reinit_func(model: torch.nn.Module, config: Config):
    """
    Returns a Closure that reinitializes the model on call.
    """
    # deepcopy to kill off references to the model parameters
    state_dict = copy.deepcopy(model.state_dict())

    # remove masks, because they shouldnt be reinitialized
    state_dict = {k: v for k, v in state_dict.items() if "_mask" not in k}

    def reinitialize(model):
        """Closure that contains the reinit state_dict to reinitialize any model."""
        if config.reinit:
            model.load_state_dict(state_dict, strict=False)

    return reinitialize


def prunify_model(model, prune_weights: bool, prune_biases: bool):
    params = _extract_pruning_params(model, prune_weights, prune_biases)
    prune.global_unstructured(params, prune.Identity)
    return params


def count_prunable_params(model):
    """Counts the number of weights and biases that are prunable with pytorch, meaning they have _mask"""
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name:
                    total += torch.numel(buffer)

                if "bias_mask" in buffer_name:
                    total += torch.numel(buffer)

    return total


def update_pruning_config(config: Config):
    """
    this function updates
     * model_shape  - > for building the model
     * pruning_rate  -> just fyi
     * pruning_trajectory  -> single point of truth
    """
    if Routines[config.pipeline] != Routines.imp:
        return config

    pparams = __count_parameters(
        config.base_model_shape, config.prune_weights, config.prune_biases
    )

    pr = __calculate_prunung_rate(
        pparams, config.pruning_rate, config.pruning_target, config.pruning_levels
    )

    param_trajectory = __make_parameter_trajectory(pparams, config.pruning_levels, pr)

    extended_shapes, extended_pparams = __make_extension_trajectory(
        pparams,
        pr,
        config.extension_levels,
        config.base_model_shape,
        config.prune_weights,
        config.prune_biases,
    )

    param_trajectory = list(reversed(extended_pparams)) + param_trajectory
    pruning_trajectory = list(-np.diff(param_trajectory))

    config.pruning_rate = pr
    config.pparams = pparams
    config.param_trajectory = param_trajectory
    config.pruning_trajectory = pruning_trajectory

    if extended_shapes:
        # add extension before parma trajectory (order decreasing)
        config.model_shape = extended_shapes[-1]

    return config


def make_global_pruner(parameters, pruning_method):
    # pruner function CLOSURE
    def pruner(amount):
        prune.global_unstructured(
            parameters=parameters, pruning_method=pruning_method, amount=amount
        )
        tensors = [getattr(module, name) for module, name in parameters]
        metrics = dict(
            min_pruning_magnitude=min(
                T[T != 0].abs().min() for T in tensors if T[T != 0].numel() > 0
            ).item()
        )

        return metrics

    return pruner


def get_pruning_method(method_name):
    if method_name == "magnitude":
        return prune.L1Unstructured
    elif method_name == "random":
        return prune.RandomUnstructured
    else:
        raise ValueError("must specify pruning method")


# not visible
def __count_parameters(shape, prune_weights=True, prune_biases=True):
    """Return the number of prunable parameters based on model shape."""
    if not prune_weights and not prune_biases:
        return 0

    params = 0
    for i in range(len(shape) - 1):
        if prune_weights:
            w = shape[i] * shape[i + 1]
            params += w
        if prune_biases:
            b = shape[i + 1]
            params += b

    return params


def __calculate_prunung_rate(
    pparams: int, pruning_rate: float, pruning_target: int, pruning_levels: int
):
    if pruning_rate is not None:
        return pruning_rate

    if pruning_levels == 0 or pruning_target is None:
        raise ValueError(
            "if pruning rate is calculated implicitly, must have prunint levels and pruning target"
        )

    """Calculate the pruning."""

    x = pruning_target / pparams

    if pruning_levels == 0:
        return 0

    t = 1 / pruning_levels

    pruning_rate = 1 - x**t

    return pruning_rate


def __make_parameter_trajectory(
    params_before_pruning: int,
    pruning_levels: int,
    pruning_rate: float,
) -> List[int]:
    t = np.arange(0, pruning_levels + 1)

    # rate of parameters remaining at each iteration
    relative_trajectory = (1 - pruning_rate) ** t

    # virtual number of parameters remaining (floating point, not discrete)
    absolute_trajectory = relative_trajectory * params_before_pruning

    # number of parameters remaining (discrete, nice)
    param_trajectory = np.rint(absolute_trajectory).astype(int)

    return list(param_trajectory)


def __solve_hidden_dim(
    model_shape,
    pruning_rate: float,
    pruning_level: int,
    params_before_pruning: int,
    prune_weights: bool = True,
    prune_biases: bool = True,
) -> int:
    """
    This is dumb! one can simply use the PQ formula and delete sympy.
    Solve for the hidden_size of
    a network after extending for any number of iterations.
    """
    if not prune_weights:
        raise NotImplementedError("makes no sense")

    # parse out from shape
    in_dim, *h, out_dim = model_shape
    num_hidden_layers = len(h)

    # inverse pruning rate / growing rate such that
    gr = 1 / (1 - pruning_rate)

    X, Y, H, P, n = symbols("X Y H P n", positive=True)
    if not prune_biases:
        equation = Eq(X * H + (n - 1) * H**2 + H * Y, P)
    else:
        equation = Eq(X * H + (n - 1) * H**2 + H * Y + Y + n * H, P)

    # set the values for the equation
    growth_factor = gr**pruning_level
    extended_params = growth_factor * params_before_pruning

    equation_with_values = equation.subs(
        {
            X: in_dim,  # number of input features
            Y: out_dim,  # number of output features
            P: extended_params,  # number of parameters
            n: num_hidden_layers,  # number of hidden dimensions
        }
    )

    # solve for hidden dimension
    solutions = solve(equation_with_values, H)

    # float solution
    _h = float(solutions[0].evalf())

    # round it to int
    h = np.rint(_h).astype(int)

    return h


def __make_extension_trajectory(
    params_before_pruning: int,
    pruning_rate: float,
    extension_levels: int,
    model_shape: list[int],
    prune_weights: bool = True,
    prune_biases: bool = True,
) -> Tuple[List[int], List[int]]:
    if not prune_weights and not prune_biases:
        return model_shape

    in_dim, *h, out_dim = model_shape
    num_hidden = len(h)

    extended_shapes = []
    extended_params = []
    for i in range(1, 1 + extension_levels):
        hidden_size = __solve_hidden_dim(
            model_shape,
            pruning_rate,
            i,
            params_before_pruning,
            prune_weights,
            prune_biases,
        )

        ext_model_shape = [in_dim] + num_hidden * [hidden_size] + [out_dim]
        extended_shapes.append(ext_model_shape)

        pparams = __count_parameters(ext_model_shape, prune_weights, prune_biases)
        extended_params.append(pparams)

    return extended_shapes, extended_params


# helpers
def _extract_pruning_params(model, prune_weights: bool, prune_biases: bool):
    """Returns the parameters which are selected for pruning based on the Config."""

    if prune_weights is None or prune_biases is None:
        raise ValueError

    # filter out non-trainable modules
    modules = [m for m in model.modules if module_is_trainable(m)]

    params = []
    if prune_weights:
        params.extend([(module, "weight") for module in modules])
    if prune_biases:
        params.extend([(module, "bias") for module in modules])
    if len(params) == 0:
        raise ValueError

    return params

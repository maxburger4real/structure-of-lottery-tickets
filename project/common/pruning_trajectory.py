import numpy as np
from sympy import symbols, Eq, solve
from common.config import Config

def calc_params_from_shape(
    shape, 
    prune_weights=True, 
    prune_biases=True
):
    if not prune_weights and not prune_biases:
        return 0
        
    params = 0
    for i in range(len(shape) - 1):
        
        if prune_weights: 
            w = shape[i] * shape[i+1]
            params += w
        if prune_biases:
            b = shape[i+1]
            params += b

    return params

def calc_extended_hidden_dim(
    pr, 
    P0, 
    in_dim,
    out_dim,
    num_hidden,
    extend_by=1,
    ignore_bias=False
):

    p = P0
    gr = 1 / (1 - pr)

    X, Y, H, P, n = symbols('X Y H P n', positive=True)
    if ignore_bias:
        equation = Eq(X*H + (n-1)*H**2 + H*Y, P)
    else:
        equation = Eq(X*H + (n-1)*H**2 + n*H + H*Y + Y, P)

    # set the values for the equation
    equation_with_values = equation.subs({
        X: in_dim,  # number of input features
        Y: out_dim,  # number of output features
        P:p * gr**extend_by,  # number of parameters
        n:num_hidden  # number of hidden dimensions
        }
    )
    
    # solve equation and get numeric value of the hidden dimension
    solutions = solve(equation_with_values, H)
    _h = float(solutions[0].evalf())

    # round to int
    h = np.rint(_h).astype(int)

    return h

def calc_extension_trajectory(
    pr, 
    P0, 
    in_dim,
    out_dim,
    num_hidden,
    iterations=1,
    ignore_bias=False
):

    p = P0
    gr = 1 / (1 - pr)

    X, Y, H, P, n = symbols('X Y H P n', positive=True)
    if ignore_bias:
        equation = Eq(X*H + (n-1)*H**2 + H*Y, P)
    else:
        equation = Eq(X*H + (n-1)*H**2 + n*H + H*Y + Y, P)

    param_trajectory, hidden_dims = [], []
    for i in range(1, 1+ iterations):

        # set the values for the equation
        equation_with_values = equation.subs({
            X: in_dim,  # number of input features
            Y: out_dim,  # number of output features
            P:p * gr**i,  # number of parameters
            n:num_hidden  # number of hidden dimensions
            }
        )
        
        # solve equation and get numeric value of the hidden dimension
        solutions = solve(equation_with_values, H)
        _h = float(solutions[0].evalf())

        # round to int
        h = np.rint(_h).astype(int)

        if h in hidden_dims: continue

        # calculate the number of parameters this network would have
        _p = calc_params(in_dim, out_dim,  h, num_hidden)
        param_trajectory.append(_p)
        hidden_dims.append(h)
        
    return hidden_dims

def calc_pruning_rate(
    params_before_pruning: int,
    params_after_pruning : int,
    pruning_levels: int
):
    """Calculate the pruning."""

    x = params_after_pruning / params_before_pruning

    if pruning_levels == 0: return 0


    t = 1 / pruning_levels

    pruning_rate = 1 - x ** t

    return pruning_rate

def build_param_trajectory(
    params_before_pruning: int, 
    pruning_levels: int,
    pruning_rate: float, 
):
    t = np.arange(0, pruning_levels+1)

    # rate of parameters remaining at each iteration
    relative_trajectory = (1 - pruning_rate) ** t
    
    # virtual number of parameters remaining (floating point, not discrete)
    absolute_trajectory = relative_trajectory * params_before_pruning

    # number of parameters remaining (discrete, nice)
    param_trajectory = np.rint(absolute_trajectory).astype(int)

    return param_trajectory

def solve_for_hidden_size(
    model_shape,
    pruning_rate: float,
    pruning_level: int,
    params_before_pruning: int,
    prune_weights: bool = True,
    prune_biases: bool = True,
):
    """Solve for the hidden_size of 
    a network after extending for any number of iterations.
    """
    if not prune_weights:
        raise NotImplemented('makes no sense')

    # parse out from shape
    in_dim, *h, out_dim = model_shape
    num_hidden_layers = len(h)

    # inverse pruning rate / growing rate such that 
    gr = 1 / (1 - pruning_rate)

    X, Y, H, P, n = symbols('X Y H P n', positive=True)
    if not prune_biases:
        equation = Eq(X*H + (n-1)*H**2 + H*Y, P)
    else:
        equation = Eq(X*H + (n-1)*H**2 + H*Y + Y + n*H , P)

    # set the values for the equation
    growth_factor = gr**pruning_level
    extended_params = growth_factor * params_before_pruning

    equation_with_values = equation.subs({
        X: in_dim,  # number of input features
        Y: out_dim,  # number of output features
        P: extended_params,  # number of parameters
        n: num_hidden_layers  # number of hidden dimensions
        }
    )

    # solve for hidden dimension
    solutions = solve(equation_with_values, H)

    # float solution
    _h = float(solutions[0].evalf())

    # round it to int
    h = np.rint(_h).astype(int)

    return h

def calc_extended_model_shapes(
    params_before_pruning: int, 
    pruning_rate: float,
    extension_levels: int,
    model_shape: list[int],
    prune_weights: bool = True,
    prune_biases: bool = True,
) -> list[list[int]]:
    
    if not prune_weights and not prune_biases:
        return model_shape

    in_dim, *h, out_dim = model_shape
    num_hidden = len(h)

    extended_shapes = []
    for i in range(1, 1+ extension_levels):

        hidden_size = solve_for_hidden_size(
            model_shape,
            pruning_rate,
            i,
            params_before_pruning,
            prune_weights,
            prune_biases,
        )

        ext_model_shape = [in_dim] + num_hidden * [hidden_size] + [out_dim]
        extended_shapes.append(ext_model_shape)

    return extended_shapes

def update_pruning_config(config: Config):
    '''        
    this function updates
     * model_shape  - > for building the model
     * pruning_rate  -> just fyi 
     * pruning_trajectory  -> single point of truth
    '''
    update_dict = {}
    params_before_pruning = calc_params_from_shape(
        config.model_shape, 
        config.prune_weights, 
        config.prune_biases
    )

    pruning_rate = calc_pruning_rate(
        params_before_pruning, 
        config.pruning_target,
        config.pruning_levels
    )

    param_trajectory = build_param_trajectory(
        params_before_pruning, 
        config.pruning_levels, 
        pruning_rate
    )

    if config.extension_levels > 0:
        extended_shapes = calc_extended_model_shapes(
            params_before_pruning,
            pruning_rate,
            config.extension_levels,
            config.model_shape,
            config.prune_weights,
            config.prune_biases
        )

        extended_trajectory = []
        for shape in reversed(extended_shapes):
            params = calc_params_from_shape(
                shape, 
                config.prune_weights, 
                config.prune_biases
            )
            extended_trajectory.append(params)
        
        # add extension before parma trajectory (order decreasing)
        param_trajectory = np.concatenate([np.array(extended_trajectory), param_trajectory])
        updated_model_shape = extended_shapes[-1]
        update_dict['model_shape'] = updated_model_shape
        update_dict['base_model_shape'] = config.model_shape

    pruning_trajectory = -np.diff(param_trajectory)
    update_dict['pruning_rate'] = pruning_rate
    update_dict['params_before_pruning'] = params_before_pruning
    update_dict['param_trajectory'] = list(param_trajectory)
    update_dict['pruning_trajectory'] = list(pruning_trajectory)

    config.update(update_dict, allow_val_change=True)


if __name__ == '__main__':
    assert 520 == calc_params_from_shape([4,20,20,2], prune_biases=False)
    assert 42 == calc_params_from_shape([4,20,20,2], prune_weights=False)
    assert 520+42 == calc_params_from_shape([4,20,20,2])

    assert np.isclose(calc_pruning_rate(100,90,1), 0.1)
    assert np.isclose(calc_pruning_rate(100,12.5,3), 0.5)
    assert np.isclose(calc_pruning_rate(100,100,1), 0.)

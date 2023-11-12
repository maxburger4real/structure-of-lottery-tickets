import numpy as np
import wandb
from sympy import symbols, Eq, solve

def calc_params(X, Y, H, n, ignore_bias=False):
    if ignore_bias:
        return X*H + (n-1)*H**2 + H*Y
    else:
        return X*H + (n-1)*H**2 + n*H + H*Y + Y

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

def get_pruning_trajectory(config):
    T = config.pruning_levels
    if T is None: raise ValueError('Must specify pruning_levels')
    
    x, *h, y = config.model_shape
    n = len(h)
    
    ignore_bias = not config.prune_biases
    P0 = calc_params(x, y, h[0], n, ignore_bias)
    PT = config.pruning_target

    if PT is not None:
        pr = 1 - (PT / P0) ** (1 / T)

    t = np.arange(0, T+1)
    param_trajectory = np.rint((P0 * (1 - pr) ** t)).astype(int)

    return pr, param_trajectory

def update_pruning_config(config):
    '''        
    # this function updates
    #  * model_shape  - > for building the model
    #  * pruning_rate  -> just fyi 
    #  * pruning_trajectory  -> single point of truth
    '''
    UPDATE_REQUIRED = True
    if not UPDATE_REQUIRED: return 

    in_dim, *h, out_dim = config.model_shape
    
    pr, param_trajectory = get_pruning_trajectory(config)
    pruning_trajectory = -np.diff(param_trajectory)


    updated_hidden_dim = calc_extended_hidden_dim(
        pr, param_trajectory[0], 
        in_dim, out_dim, len(h), 
        extend_by=1
    )

    if update_pruning_config == h[0]:
        print("what to do here if it is equal. OR less?")

    # extend pruning_trajectory

    assert False: 'Need to update pruning trajectory'

    config.update({
        'pruning_rate': pr,
        'model_shape' : [in_dim] + [updated_hidden_dim] * len(h) + [out_dim],
        'pruning_trajectory' : pruning_trajectory,
        'param_trajectory' : [4,3,2,1],  # is this needed? already implemented on the fly.
    }, allow_val_change=True)



if __name__ == '__main__':
    print(calc_extension_trajectory(
        pr = 0.2, 
        P0 = 50, 
        in_dim = 4,
        out_dim = 2,
        num_hidden = 2,
        iterations=1,
        ignore_bias=False
    )[-1])




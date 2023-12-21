import wandb
import numpy as np
from common.nx_utils import neuron_analysis, subnet_analysis, _get_w_and_b
from common.config import Config
from common.constants import *

# available as log.loss() , log.subnetworks(). looks pretty
def descriptive_statistics(model, at_init=False, prefix='descriptive'):
    weights, biases = _get_w_and_b(model)

    for l, (w, b) in enumerate(zip(weights, biases, strict=True)):
        params = dict(w=w.numpy(), b=b.numpy())

        for name, p in params.items():
            num_params = p.size
            num_zeros = np.sum(p == 0)
            num_nonzeros = np.sum(p != 0)
            abs_nonzero_params = np.abs(p[p != 0])

            if len(abs_nonzero_params) > 0:
                median = np.median(abs_nonzero_params)
                mean = np.mean(abs_nonzero_params)
            else:
                median, mean = 0,0
            
            assert num_params == num_nonzeros + num_zeros

            rate_remaining = 1. if num_zeros == 0 else (num_nonzeros / num_params)

            if at_init and rate_remaining != 1.0:
                rate_remaining = 1.0

            wandb.log({
                f'{prefix} L{l}-rate-remaining({name})' : rate_remaining,
                f'{prefix} L{l}-median(abs({name}))' : median,
                f'{prefix} L{l}-mean(abs({name}))' : mean,
            }, commit=False)

def scalar_metric(value, prefix, commit=False):
    '''Log a singular scalar value.'''

    if isinstance(value, np.ndarray):
        value = value.item()

    wandb.log({prefix:value}, commit=commit)

def taskwise_metric(metric : np.ndarray, prefix, commit=False):
    """log loss that takes care of logging the tasks sepertely."""
    
    batch_size, num_tasks = metric.shape
    batch_metric = metric.mean(axis=0)
    d = {prefix : batch_metric.mean()}

    if num_tasks < 2:
        wandb.log(d, commit=commit)
        return

    for i, task_metric in enumerate(batch_metric, start=1):
        key = f'{prefix}-{i}'
        d[key] = task_metric.item()

    wandb.log(d, commit=commit)

def zombies_and_comatose(G, config: Config):
    """
    TODO:
    - problem: there are too many metrics that are tracked
    - it is unnecessary to track a metric for every single comatose or zombie neuron.
    - i dont care about it
    - what do i car about?
    - how many comatose/zombies are there in total
    - min-max-mean-median of their bias
    - lifespan of them
    - how long do they survive? -> histogram
    - how long do they survive when they have a positive bias vs 0 bias
    - how long do they survive depending on their remaining inputs or outputs

    Histogram of zombies/comatose by number of in/out connections
    Histogram of zombies based on how long they already exist
    Histogram
    """
    zombies, comatose = neuron_analysis(G, config)
    wandb.log({
        'zombies' : len(zombies), 
        'comatose' : len(comatose)
        }, commit=False)
    
    for i, data in G.subgraph(zombies).nodes(data=True):
        wandb.log({
            f'zombie-neuron-{i}': i,
            f'zombie-bias-{i}' : data[BIAS],
            f'zombie-out-{i}' : data['out']
        }, commit=False)

    for i, data in G.subgraph(comatose).nodes(data=True):
        wandb.log({
            f'comatose-neuron-{i}': i,
            f'comatose-bias-{i}' : data[BIAS],
            f'comatose-in-{i}' : data['in']
        }, commit=False)

def subnetworks(G, config: Config, ignore_fragments=True, log_detailed=True):
    """Log everything about subnetworks."""
    subnet_report = subnet_analysis(G, config)

    complete_subnetworks = 0
    partial_subnetworks = 0
    fragment_subnetworks = 0
    zombie_subnetworks = 0

    for net in subnet_report:
        ic = net['input']['complete']
        ip = net['input']['incomplete']
        oc = net['output']['complete']
        op = net['output']['incomplete']
        
        # no outputs --> fragment
        if not any([oc, op]): fragment_subnetworks += 1
        # no inputs but outputs --> zombie
        elif not any([ic, ip]): zombie_subnetworks += 1
        # complete inputs and outputs and they are the same --> complete
        elif all([ic, oc]) and ic == oc: complete_subnetworks += 1
        # has outputs and inputs, but not complete ones
        else:  partial_subnetworks += 1

        if not log_detailed: continue

        # make name such that it looks like this
        # c-12-12-c for complete inputs complete outputs for tasks 1 and 2
        # problems arise when more than 9 tasks.
        name = ''
        for prefix, features in [('c' ,ic),('p' ,ip)]:
            if len(features) == 0: continue
            name += prefix + '-' +  ''.join(map(str,features))
        name += "-"
        for prefix, features in [('c' ,oc),('p' ,op)]:
            if len(features) == 0: continue
            name += ''.join(map(str,features)) + '-' + prefix

        is_fragment = (name == "-")

        if is_fragment and ignore_fragments: continue

        name = 'fragment' if is_fragment else name 
        metric_name = 'sub_' + name
        wandb.log({metric_name : net['num_weights']}, commit=False)

    wandb.log({    
        'complete_subnetworks' : complete_subnetworks,
        'partial_subnetworks' : partial_subnetworks,
        'fragment_subnetworks' : fragment_subnetworks,
        'zombie_subnetworks' : zombie_subnetworks,
    }, commit=False)

def returns_true_every_nth_time(n, and_at_0=False):

    if n is None or n <= 0:
        return lambda : False

    N = range(n)

    def generator():
        if and_at_0:
            yield True
        while True:
            for _ in N:
                yield False
            yield True
    g = generator()

    def closure():
        return next(g)

    return closure
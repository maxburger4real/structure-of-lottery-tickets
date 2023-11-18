import wandb
import numpy as np
from common.nx_utils import neuron_analysis, subnet_analysis
from common.config import Config
from common.constants import *

def logdict(loss : np.ndarray, prefix):
    """create a loggable dict for wandb"""

    dims = len(loss.shape)
    if dims == 0:
        return {prefix: loss.item()}
    if 0 < dims < 3:
        return {prefix: loss.mean().item()}
    if dims == 3:
        metrics = {prefix: loss.mean().item()}

        # assumption: last dimension is task dimension
        all_axis_but_the_last_one = tuple(range(dims-1))
        taskwise_loss = loss.mean(axis=all_axis_but_the_last_one)
        for i, l in enumerate(taskwise_loss):
            metrics[prefix + '_' + str(i)] = l.item()

        return metrics

 
def log_param_aggregate_statistics(G, config: Config, commit=False):
    """Log statistics of weights and bias matrices.
    """

    # go over every layer in the nn
    for l in range(len(config.model_shape)):
        parameters = {}

        w = []
        for u, v, data in G.edges(data=True):
            input_node = G.nodes()[u]
            if input_node[LAYER]==l:
                w.append(data[WEIGHT])
        parameters['w'] = w

        if l != 0:
            b = []
            for _, data in G.nodes(data=True):
                if data[LAYER] == l:
                    b.append(data[BIAS])
            parameters['b'] = b

        for name, p in parameters.items():
            zeros = p[p == 0].numel().item()
            total = p.numel().item()
            wandb.log({
                f'L{l}-size({name}<0)' : p[p < 0].numel().item(),
                f'L{l}-size({name}>0)' : p[p > 0].numel().item(),
                f'L{l}-size({name}==0)' : zeros,
                f'L{l}-mean({name})' : np.mean(p).item(),
                f'L{l}-mean(abs({name}))' : np.mean(np.abs(p)).item(),
                f'L{l}-spratio({name})' : zeros / total
            }, commit=commit)


def log_zombies_and_comatose(G, config):
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


def log_subnet_analysis(G, config, ignore_fragments=True, log_detailed=True):
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
import wandb
import numpy as np
from common.nx_utils import neuron_analysis, _get_w_and_b
from common.config import Config
from common.nxutils import GraphManager
from common.constants import *

class Logger():
    '''This class handles logging with Wandb and is strict with overriding. It raises an Exception.'''
    def __init__(self):
        self.log_graphs = False
        self.logdict = {}

    def commit(self):
        '''Commit to wandb, but only if there is something to commit.'''
        if not self.logdict: return
        wandb.log(self.logdict)
        self.logdict = {}

    def summary(self, gm):
        wandb.run.summary['split-iteration'] = gm.split_iteration
        wandb.run.summary['degradation-iteration'] = gm.degradation_iteration
        # if gm.split_iteration is not None
        wandb.log({'split-iteration' : gm.split_iteration})
        wandb.log({'degradation-iteration' : gm.degradation_iteration})

    def splitting(self, gm: GraphManager):
        if gm is None: return
        self.__strict_insert('untapped-potential', gm.untapped_potential)
    
    def graphs(self, gm: GraphManager):
        if gm is None: return
        if len(gm.catalogue) > 1: self.log_graphs =  True  # only log 2 or more.
        if not self.log_graphs: return
        for name, g in gm.catalogue.items():
            self.__strict_insert(name, gm.make_plotly(g))

    def metrics(self, values: dict, prefix='', only_if_true=True):
        if not only_if_true: return
        for key, x in values.items():
            self.__metric(x, prefix+key)
    
    def feature_categorization(self, gm: GraphManager):
        if gm is None: return
        total = len(gm.lifecycles)
        values = {
            'num-alive' : len(gm.alive_params_list),
            'num-zombie' : len(gm.zombie_params_list),
            'num-audience' : len(gm.audience_params_list),
            'num-unproductive' : len(gm.unproductive_params_list),
        }

        for name, value in values.items():
            self.__strict_insert(name + '-abs', value)
            self.__strict_insert(name + '-rel', value / total)
    
    def __metric(self, x : np.ndarray, prefix: str):

        if not isinstance(x, np.ndarray):
            self.__strict_insert(prefix, x)
            return 
        
        # single value in array
        if x.shape == (1,):
            self.__strict_insert(prefix, x.item())
            return 
        
        # single dimensional. assuming batch
        if len(x.shape) == 1:
            raise ValueError(f'what is this {x.shape, x}')

        # batch and task
        if len(x.shape) == 2:
            batch_size, num_tasks = x.shape
            batch_metric = x.mean(axis=0)

            self.__strict_insert(prefix, batch_metric.mean())

            if num_tasks < 2: return 
                
            for i, task_metric in enumerate(batch_metric, start=1):
                key = f'{prefix}-{i}'
                self.__strict_insert(key, task_metric.item())
            return 
        
        raise ValueError('This is not planned.')

    def __strict_insert(self, key, value):
        if key in self.logdict:
            raise ValueError('Cannot Override Key in strict logdict.')
        self.logdict[key] = value


def lifetime(gm: GraphManager):
    raise

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
import wandb
import numpy as np
from common.nxutils import GraphManager
from common.constants import *

class Logger():
    '''This class handles logging with Wandb and is strict with overriding. It raises an Exception.'''
    def __init__(self, task_description, step=None, start=0):
        self.task_description = task_description
        self.logdict = {}

    def commit(self):
        '''Commit to wandb, but only if there is something to commit.'''
        if not self.logdict: return
        wandb.log(self.logdict)
        self.logdict = {}

    def summary(self):
        raise
        wandb.run.summary['split-iteration'] = self.gm.split_iteration
        wandb.run.summary['degradation-iteration'] = self.gm.degradation_iteration
        self.__strict_insert('split-iteration', self.gm.split_iteration)
        self.__strict_insert('degradation-iteration', self.gm.degradation_iteration)

    def splitting(self):
        raise
        self.__strict_insert('untapped-potential', self.gm.untapped_potential)
    
    def graphs(self):
        raise
        for name, g in self.gm.catalogue.items():
            self.__strict_insert(name, self.gm.fig(g))

    def metrics(self, values: dict, prefix=''):
        for key, x in values.items():
            self.__metric(x, prefix+key)
    
    def feature_categorization(self):
        raise
        total_nodes = sum([value for key, value in self.gm.node_statistics.items() if key != ParamState.pruned])
        total_edges = sum([value for key, value in self.gm.edge_statistics.items() if key != ParamState.pruned])
        
        for state in ParamState:
            num_nodes = self.gm.node_statistics[state]
            num_edges = self.gm.edge_statistics[state]
            self.__strict_insert(state.name + '-features' + '-abs', num_nodes)
            self.__strict_insert(state.name + '-features' + '-rel', num_nodes / total_nodes)

            self.__strict_insert(state.name + '-weights' + '-abs', num_edges)
            self.__strict_insert(state.name + '-weights' + '-rel', num_edges / total_edges)

            self.__strict_insert(state.name + '-rel', (num_edges + num_nodes) / (total_edges + total_nodes))
            self.__strict_insert(state.name + '-abs', num_edges + num_nodes)

    def __metric(self, x : np.ndarray, prefix: str):

        if not isinstance(x, np.ndarray):
            self.__strict_insert(prefix, x)
            return 
        
        # single value in array
        if x.shape == (1,) or x.shape == (1,1):
            self.__strict_insert(prefix, x.item())
            return 
        
        # single dimensional. assuming batch
        if len(x.shape) == 1:
            self.__strict_insert(prefix, x.mean().item())
            return

        # batch and task
        if len(x.shape) == 2:
            batch_size, num_tasks = x.shape

            if num_tasks != len(self.task_description):
                raise ValueError('SOMETHING IS WRONG')

            batch_metric = x.mean(axis=0)

            self.__strict_insert(prefix, batch_metric.mean())

            if num_tasks < 2: return 
                
            for (name, (_,__)), task_metric in zip(self.task_description, batch_metric):
                key = f'{name}-{prefix}'
                self.__strict_insert(key, task_metric.item())
            return 
        
        raise ValueError('This is not planned.')

    def __strict_insert(self, key, value):
        if key in self.logdict:
            raise ValueError('Cannot Override Key in strict logdict.')
        self.logdict[key] = value

import wandb
import numpy as np

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

    def metrics(self, *metrics: dict, prefix='', **keyword_metrics):
        for _metrics in metrics:
            for key, x in _metrics.items():
                self.__metric(x, prefix+key)

        for key, x in keyword_metrics.items():
            self.__metric(x, prefix+key)

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
                key = f'{prefix}taskwise/{name}-'
                self.__strict_insert(key, task_metric.item())
            return 
        
        raise ValueError('This is not planned.')

    def __strict_insert(self, key, value):
        if key in self.logdict:
            raise ValueError('Cannot Override Key in strict logdict.')
        self.logdict[key] = value

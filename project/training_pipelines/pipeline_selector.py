from common.constants import *
from common.config import Config
from training_pipelines import vanilla, imp, bimt

def run(model, train_loader, test_loader, loss_fn, config: Config):
    
    pipeline = config.pipeline
    if pipeline == VANILLA:
        return vanilla.run(model, train_loader, test_loader, loss_fn, config)

    if pipeline == IMP:
        return imp.run(model, train_loader, test_loader, loss_fn, config)
    
    if pipeline == BIMT:
        return bimt.run(model, train_loader, test_loader, loss_fn, config)
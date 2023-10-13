from common.tracking import Config, IMP, VANILLA
from training_pipelines import vanilla, imp

def run(model, train_loader, test_loader, optim, loss_fn, config: Config):
    
    pipeline = config.pipeline
    if pipeline == VANILLA:
        return vanilla.run(model, train_loader, test_loader, optim, loss_fn, config)

    if pipeline == IMP:
        return imp.run(model, train_loader, test_loader, optim, loss_fn, config)
from tqdm import tqdm
from common.config import Config
from common.log import Logger
from common.persistance import save_model_or_skip
from common.training import build_early_stopper, build_optimizer, evaluate, update
from common.constants import *

def run(model, train_loader, test_loader, config: Config):
    
    log = Logger(None, config.task_description)
    optim = build_optimizer(model, config)
    stop = build_early_stopper(config)

    # log initial performance
    init_loss, init_accuracy = evaluate(model, test_loader, config.device)
    log.metrics({VAL_LOSS:init_loss, ACCURACY:init_accuracy})
    log.commit()  # LOG BEFORE TRAINING

    # train and evaluate
    epochs = config.epochs
    for epoch in tqdm(range(epochs), 'Training', epochs):
        
        # update
        train_loss = update(model, train_loader, optim, config.device, config.l1_lambda)

        # evaluate
        val_loss, val_acc = evaluate(model, test_loader, config.device)
        log.metrics({TRAIN_LOSS : train_loss, VAL_LOSS : val_loss, ACCURACY : val_acc})
        log.commit()
        if stop(val_loss.mean().item()): 
            break

        #if val_acc.mean().item() == 1: 
            #print('Perfection reached.')
            #break

    # store model
    save_model_or_skip(model, config, config.epochs)

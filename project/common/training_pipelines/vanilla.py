import wandb
from tqdm import tqdm
from common.config import Config
from common.log import log_loss
from common.persistance import save_model_or_skip
from common.training import build_early_stopper, build_optimizer, evaluate, update
from common.constants import *

def run(model, train_loader, test_loader, loss_fn, config: Config):

    optim = build_optimizer(model, config)
    stop = build_early_stopper(config)

    # log initial performance
    loss_init = evaluate(model, test_loader, loss_fn, config.device)
    log_loss(loss_init, VAL_LOSS, commit=True)

    # train and evaluate
    epochs = config.training_epochs

    for epoch in tqdm(range(epochs), 'Training', epochs):
        
        # update
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        log_loss(loss_train, TRAIN_LOSS)

        # evaluate
        loss_eval = evaluate(model, test_loader, loss_fn, config.device)
        log_loss(loss_eval, VAL_LOSS, commit=True)

        if stop(loss_eval.mean().item()): break

    wandb.log({'stop': epoch})

    # store model
    save_model_or_skip(model, config, config.training_epochs)

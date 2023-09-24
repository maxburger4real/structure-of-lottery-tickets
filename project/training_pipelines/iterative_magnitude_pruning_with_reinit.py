import wandb
from common import pruning

from common.tracking import Config
from common.torch_utils import measure_global_sparsity
from common.training import train, evaluate

def run(model, train_loader, test_loader, optim, loss_fn, config: Config):
    # preparing for pruning
    params_to_prune = pruning.convert_to_pruning_model(model.modules, prune_weights=True, prune_biases=True)
    reinit_model_state_dict = pruning.get_model_state_dict(model, drop_masks=True)

    # evaluate before anything
    loss_eval = evaluate(model, test_loader, loss_fn)
    wandb.log({'loss/eval' : loss_eval.mean().item()})

    for step in range(config.pruning_levels):
        
        # train for training_epochs
        loss_train = train(model, train_loader, optim, loss_fn, epochs=config.training_epochs)

        # evaluate the model on test set
        loss_eval = evaluate(model, test_loader, loss_fn)

        # prune by global magnitude
        pruning.global_magnitude_pruning(params_to_prune, config.pruning_rate)

        num_zeros, num_elements, sparsity = measure_global_sparsity(model, use_mask=True)

        wandb.log({
            'loss/eval' : loss_eval.mean().item(),
            'loss/train' : loss_train.mean().item(),
            'pruning_level' : step,
            'remaining_weights' : num_elements - num_zeros,
        })

        # reset to the original weights
        model.load_state_dict(reinit_model_state_dict, strict=False)

    return model


from common import torch_utils, pruning

def run(model, pruning_levels, pruning_rate, train_loader, test_loader, optim, loss_fn, training_epochs=1):

    # preparing for pruning
    params_to_prune = pruning.convert_to_pruning_model(model.modules, prune_weights=True, prune_biases=True)
    reinit_model_state_dict = pruning.get_model_state_dict(model, drop_masks=True)

    # evaluate before anything
    eval_losses = [torch_utils.evaluate(model, test_loader, loss_fn)]
    train_losses = []
    for step in range(pruning_levels):
        
        # train for training_epochs
        train_loss = torch_utils.train(model, train_loader, optim, loss_fn, epochs=training_epochs)
        train_losses.append(train_loss)
        
        # evaluate the model on test set
        eval_loss = torch_utils.evaluate(model, test_loader, loss_fn)
        eval_losses.append(eval_loss)

        # prune by global magnitude
        pruning.global_magnitude_pruning(params_to_prune, pruning_rate)

        # reset to the original weights
        model.load_state_dict(reinit_model_state_dict, strict=False)

    return train_losses, eval_losses

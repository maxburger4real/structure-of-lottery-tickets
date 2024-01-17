import torch
import numpy as np
import random


def get_pytorch_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed):
    """Sets all seeds of randomness sources"""
    # TODO: set MPS seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def check_if_models_equal(model1, model2):
    """Returns True if both models are equal, otherwise False"""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def module_is_trainable(module: torch.nn.Module):
    return any(param.requires_grad for param in module.parameters())


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    """from legendary overarchiever lei mao https://leimao.github.io/blog/PyTorch-Pruning/"""

    num_zeros = 0
    num_elements = 0

    if use_mask:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight is True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias is True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight is True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias is True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model, weight=True, bias=False, use_mask=False):
    """from legendary overarchiever lei mao https://leimao.github.io/blog/PyTorch-Pruning/"""
    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=use_mask
            )
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

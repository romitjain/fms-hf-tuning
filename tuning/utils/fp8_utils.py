import torch
from functools import partial

def filter_linear_layers(module, fqn, first_layer_name=None, last_layer_name=None):
    if not isinstance(module, torch.nn.Linear):
        return False

    if fqn in (first_layer_name, last_layer_name):
        return False
    if fqn.endswith("lm_head") or "embed" in fqn:
        return False

    w = module.weight
    if w.ndim != 2:
        return False

    # Weight dims constraint (still useful)
    if module.in_features % 16 != 0 or module.out_features % 16 != 0:
        return False

    return True

def get_fp8_filter_func(model):
    first_linear = last_linear = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            first_linear = name if first_linear is None else first_linear
            last_linear = name
    return partial(filter_linear_layers, first_layer_name=first_linear, last_layer_name=last_linear)

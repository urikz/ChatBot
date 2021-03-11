import torch


def clip_grad_norm(parameters, max_norm):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    total_norm = 0
    for p in parameters:
        if p.grad.data.is_sparse:
            param_norm = p.grad.data.coalesce()._values().norm()
        else:
            param_norm = p.grad.data.norm()
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6).item()
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

import torch

def get_grad_norm(model: torch.nn.Module) -> float:
    square_sum = 0.
    for param in model.parameters():
        if param.grad is not None:
            square_sum += param.grad.detach().data.norm(2).item() ** 2
    return square_sum ** 0.5
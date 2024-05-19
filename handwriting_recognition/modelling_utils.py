from functools import partial, cache

from torch import nn, optim
from timm import create_model
import torch


@cache
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_image_model(model_name: str) -> nn.Module:
    return create_model(model_name=model_name, pretrained=True, in_chans=1)


def get_optimizer(
    model: nn.Module,
    optim_type: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> optim.Optimizer:

    if optim_type.lower() == "sgd":
        optimizer = partial(optim.SGD, nesterov=True, momentum=momentum)
    elif optim_type.lower() == "adam":
        optimizer = optim.Adam
    else:
        raise NotImplementedError("Must be one of Adam / SGD")

    return optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

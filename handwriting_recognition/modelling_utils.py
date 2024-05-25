from functools import partial, cache

from torch import nn, optim
from timm import create_model
import torch
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from typing import Any


@cache
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_image_model(model_name: str) -> nn.Module:
    return create_model(model_name=model_name, pretrained=False, in_chans=1)


def get_optimizer(model: nn.Module, optim_config: dict[str, Any]) -> optim.Optimizer:
    if optim_config["optim_type"].lower() == "sgd":
        optimizer = partial(optim.SGD, nesterov=True, momentum=optim_config["momentum"])
    elif optim_config["optim_type"].lower() == "adam":
        optimizer = partial(optim.Adam, betas=(optim_config["beta1"], optim_config["beta2"]))

    else:
        raise NotImplementedError("Must be one of Adam / SGD")

    return optimizer(
        params=model.parameters(), lr=optim_config["learning_rate"], weight_decay=optim_config["weight_decay"]
    )


def get_scheduler(optimizer: optim.Optimizer, scheduler_config: dict[str, Any]) -> LRScheduler:
    if scheduler_config["scheduler_type"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, **scheduler_config["params"])
    else:
        raise NotImplementedError(f"Unknown scheduler: {scheduler_config['scheduler_type']}")

    return scheduler

from functools import partial

from PIL import Image
from torch import nn, optim
from transformers import ViTImageProcessor, ViTModel


def get_image_model_and_processor(model_name: str, processor_name: str | None) -> tuple[ViTModel, ViTImageProcessor]:
    if processor_name is None:
        processor_name = model_name

    processor = ViTImageProcessor.from_pretrained(processor_name)
    model = ViTModel.from_pretrained(model_name)

    return model, processor


def get_optimizer(
    model: nn.Module,
    optim_type: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> optim.Optimizer:
    if optim_type == "SGD":
        optimizer = partial(optim.SGD, nesterov=True, momentum=momentum)
    elif optim_type == "Adam":
        optimizer = optim.Adam
    else:
        raise NotImplementedError("Must be one of Adam / SGD")

    return optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

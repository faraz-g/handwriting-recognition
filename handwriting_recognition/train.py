import argparse
import json
import os

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from handwriting_recognition.model import HandwritingRecognitionModel
from handwriting_recognition.modelling_utils import get_image_model_and_processor, get_optimizer
from handwriting_recognition.utils import TrainingConfig, get_config

# from handwriting_recognition.data_loader import HandWritingDataset # TODO

torch.backends.cudnn.benchmark = True


class LossTracker:
    def __init__(self) -> None:
        self.initialise()

    def initialise(self):
        self.loss_value = 0
        self.average = 0
        self.sum = 0
        self.count_iters = 0

    def update(self, loss_value: float, iterations: int):
        self.loss_value = loss_value
        self.sum += loss_value * iterations
        self.count_iters += iterations
        self.average = self.sum / self.count_iters


def train(run_prefix: str, config_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    config = TrainingConfig.from_path(config_path=config_path)

    print("Training with config: \n", json.dumps(config.to_dict(), indent=1))

    image_model, image_processor = get_image_model_and_processor(
        model_name=config.feature_extractor_config.hf_model_name,
        processor_name=config.feature_extractor_config.hf_pre_processor_name,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HandwritingRecognitionModel(image_feature_extractor=image_model)

    model = model.to(device)
    loss_function = CrossEntropyLoss().to(device)
    optimizer = get_optimizer(
        model=model,
        optim_type=config.optim_config.optim_type,
        lr=config.optim_config.learning_rate,
        momentum=config.optim_config.momentum,
        weight_decay=config.optim_config.weight_decay,
    )

    # TODO
    # train_loader = DataLoader(
    #     data_train,
    #     batch_size=config.batch_size,
    #     pin_memory=False,
    #     shuffle=True,
    #     drop_last=True,
    # )

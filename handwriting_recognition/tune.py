import argparse
from handwriting_recognition.train import train_handwriting_recognition
import argparse
import os

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import random

# from handwriting_recognition.label_converter import LabelConverter
# from handwriting_recognition.model.model import HandwritingRecognitionModel
# from handwriting_recognition.modelling_utils import get_image_model, get_optimizer, get_scheduler
from handwriting_recognition.utils import TrainingConfig, get_dataset_folder_path

# from handwriting_recognition.dataset import HandWritingDataset
from pathlib import Path

# from handwriting_recognition.modelling_utils import get_device
# from handwriting_recognition.eval import cer, wer

import ray
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from filelock import FileLock
import tempfile
from typing import Any
import shutil

CONFIG_SEARCH_SPACE = {
    "batch_size": tune.choice([16, 32, 64, 128, 192, 256]),
    "lstm_hidden_size": tune.choice([64, 128, 256, 512]),
    "attention_hidden_size": tune.choice([64, 128, 256, 512]),
    "batches_per_epoch": tune.choice([500, 5000, 10000]),
}

OPTIM_CONFIG_SEARCH_SPACE = {
    "optim_type": "adam",
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.loguniform(1e-4, 1e-2),
    "beta1": tune.uniform(0.7, 1),
    "beta2": tune.uniform(0.7, 1),
}

SCHEDULER_CONFIG_SEARCH_SPACE = {
    "scheduler_type": "cosine",
    "params": {"T_max": tune.randint(10, 100), "eta_min": tune.loguniform(1e-4, 1e-1)},
}


def tune_handwriting_recognition_model(config_name: str, out_dir: str, num_samples: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_folder = Path(out_dir) / config_name

    os.makedirs(out_folder, exist_ok=True)

    config_path = Path(__file__).parent.joinpath("configs", config_name).with_suffix(".json")
    config = TrainingConfig.from_path(config_path=config_path)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    config_dict = config.model_dump()

    config_dict.update(CONFIG_SEARCH_SPACE)
    config_dict["optim_config"].update(OPTIM_CONFIG_SEARCH_SPACE)
    config_dict["scheduler_config"].update(SCHEDULER_CONFIG_SEARCH_SPACE)

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=config_dict["max_epochs"],
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        train_handwriting_recognition,
        resources_per_trial={"cpu": os.cpu_count() - 1, "gpu": 1},
        config=config_dict,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial(metric="val_loss", mode="min", scope="last-5-avg")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    print(f"Best trial final character error rate: {best_trial.last_result['character_error_rate']}")

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="val_loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "checkpoint.pt"
        print(f"Copied best checkpoint to {os.path.abspath(out_folder)}")

        shutil.copy(data_path, out_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="default_config", type=str)
    parser.add_argument("--out_dir", default="model_outputs")
    parser.add_argument("--num_samples", type=int, default=25)

    args = parser.parse_args()

    tune_handwriting_recognition_model(**vars(args))

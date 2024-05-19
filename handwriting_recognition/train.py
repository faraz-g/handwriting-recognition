import argparse
import os

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import random

from handwriting_recognition.label_converter import LabelConverter
from handwriting_recognition.model.model import HandwritingRecognitionModel
from handwriting_recognition.modelling_utils import get_image_model, get_optimizer
from handwriting_recognition.utils import TrainingConfig, get_dataset_folder_path
from handwriting_recognition.dataset import HandWritingDataset
from pathlib import Path

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


def _single_epoch(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: CrossEntropyLoss,
    train_loader: DataLoader,
    config: TrainingConfig,
    converter: LabelConverter,
    device: torch.device,
):
    max_batches = config.batches_per_epoch
    model = model.train()
    loss_tracker = LossTracker()

    with tqdm(total=(max_batches), desc=f"Training Epoch: {epoch}", ncols=0) as pbar:
        for i, data in enumerate(train_loader):
            images = data[0]
            labels = data[1]

            text, length = converter.encode(labels, batch_max_length=config.max_text_length)

            text = text.to(device)
            length = length.to(device)
            images = images.to(device)

            preds = model(x=images, y=text[:, :-1], is_train=True)
            target = text[:, 1:]

            loss = loss_function(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            loss_tracker.update(loss.item(), images.size(0))

            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix_str(f"LR: {lr:.4f} Avg. Loss: {loss_tracker.average:.4f}")
            pbar.update()

            if i + 1 == max_batches:
                break


def _evaluate(
    epoch: int,
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
):
    # TODO
    return

    model = model.eval()

    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Validating Epoch: {epoch}", ncols=0)
        for data in pbar:
            images = data[0]
            labels = data[1]


def train(config_name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    config_path = Path(__file__).parent.joinpath("configs", config_name).with_suffix(".json")
    config = TrainingConfig.from_path(config_path=config_path)
    print("Training with config: \n", config.model_dump_json(indent=1))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    image_model = get_image_model(model_name=config.feature_extractor_config.model_name)

    data_train = HandWritingDataset(
        data_path=get_dataset_folder_path() / "pre_processed" / "train.csv",
        img_size=config.feature_extractor_config.input_size,
    )
    data_val = HandWritingDataset(
        data_path=get_dataset_folder_path() / "pre_processed" / "validation.csv",
        img_size=config.feature_extractor_config.input_size,
    )

    config.max_text_length = data_train.max_length
    config.num_classes = len(data_train.char_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HandwritingRecognitionModel(image_feature_extractor=image_model, training_config=config)

    model = model.to(device)
    loss_function = CrossEntropyLoss(ignore_index=0).to(device)

    optimizer = get_optimizer(
        model=model,
        optim_type=config.optim_config.optim_type,
        lr=config.optim_config.learning_rate,
        momentum=config.optim_config.momentum,
        weight_decay=config.optim_config.weight_decay,
    )

    train_loader = DataLoader(
        data_train,
        batch_size=config.batch_size,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        data_val,
        batch_size=config.batch_size,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    converter = LabelConverter(character_set=data_train.char_set)

    out_folder = Path(out_dir) / config_name
    os.makedirs(out_folder, exist_ok=True)
    epochs_since_best_loss = 0

    for epoch in range(config.max_epochs):
        _single_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            train_loader=train_loader,
            config=config,
            converter=converter,
            device=device,
        )

        if epoch % config.evaluation_frequency == 0:
            validation_loss = _evaluate(
                epoch=epoch,
                model=model,
                val_loader=val_loader,
                device=device,
            )
            if validation_loss < best_val_loss:
                epochs_since_best_loss = 0
                best_val_loss = validation_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "state": model.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(out_folder, "best"),
                )
            else:
                epochs_since_best_loss += 1

            print(f"Validation Loss at epoch {epoch}: {validation_loss}. Best Loss: {best_val_loss}")

        torch.save(
            {
                "epoch": epoch,
                "state": model.state_dict(),
                "best_val_loss": best_val_loss,
            },
            os.path.join(out_folder, f"{epoch}"),
        )
        torch.save(
            {
                "epoch": epoch,
                "state": model.state_dict(),
                "best_val_loss": best_val_loss,
            },
            os.path.join(out_folder, "last"),
        )
        if epochs_since_best_loss > config.early_stopping_threshold:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="default_config", type=str)
    parser.add_argument("--out_dir", default="model_outputs")

    args = parser.parse_args()

    train(**vars(args))

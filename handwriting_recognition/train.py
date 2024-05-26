import argparse
import os

import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from handwriting_recognition.label_converter import LabelConverter
from handwriting_recognition.model.model import HandwritingRecognitionModel
from handwriting_recognition.modelling_utils import get_optimizer, get_scheduler
from handwriting_recognition.utils import get_dataset_folder_path
from handwriting_recognition.dataset import HandWritingDataset
from handwriting_recognition.modelling_utils import get_device
from handwriting_recognition.eval import cer, wer

from ray import train
from ray.train import Checkpoint, get_checkpoint
import tempfile
from typing import Any
import json

torch.backends.cudnn.benchmark = True


class LossTracker:
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def _single_epoch(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_function: CrossEntropyLoss,
    train_loader: DataLoader,
    config: dict[str, Any],
    converter: LabelConverter,
):
    max_batches = config["batches_per_epoch"]
    model = model.train()
    loss_tracker = LossTracker()

    with tqdm(total=(max_batches), desc=f"Training Epoch: {epoch}", ncols=0) as pbar:
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            images = data[0]
            labels = data[1]

            text, length = converter.encode(labels)

            text = text.to(get_device())
            length = length.to(get_device())
            images = images.to(get_device())

            preds = model(x=images, y=text[:, :-1], is_train=True)

            target = text[:, 1:]

            loss = loss_function(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            loss_tracker.add(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix_str(f"LR: {lr:.4f} Avg. Loss: {loss_tracker.val():.4f}")
            pbar.update()

            if i + 1 == max_batches:
                break

    return loss_tracker.val().item()


def _evaluate(
    epoch: int,
    model: torch.nn.Module,
    data_loader: DataLoader,
    converter: LabelConverter,
    loss_function: CrossEntropyLoss,
    max_iter: int | None = None,
):
    model = model.eval()

    all_preds = []
    all_ground_truths = []
    loss_tracker = LossTracker()

    with torch.inference_mode():
        pbar = tqdm(data_loader, desc=f"Validating Epoch: {epoch}", ncols=0)
        for i, data in enumerate(pbar):
            images = data[0]
            labels = data[1]

            text, length = converter.encode(labels)

            text = text.to(get_device())
            length = length.to(get_device())
            images = images.to(get_device())

            preds = model(x=images, y=text[:, :-1], is_train=False)
            target = text[:, 1:]

            loss = loss_function(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            loss_tracker.add(loss)

            predicted_classes = preds.argmax(dim=-1)

            ground_truth = converter.decode(target, length)
            predicted = converter.decode(predicted_classes, length)

            all_ground_truths.extend(ground_truth)
            all_preds.extend(predicted)

            if max_iter is not None and i + 1 == max_iter:
                break

    all_preds = [x[: x.find("[s]")] for x in all_preds]
    all_ground_truths = [x[: x.find("[s]")] for x in all_ground_truths]

    character_error_rate = cer(all_preds, all_ground_truths)
    word_error_rate = wer(all_preds, all_ground_truths)

    return loss_tracker.val().item(), character_error_rate.item(), word_error_rate.item(), all_preds, all_ground_truths


def train_handwriting_recognition(config: dict[str, Any]) -> None:
    data_train = HandWritingDataset(
        data_path=get_dataset_folder_path() / "pre_processed" / "train.csv",
        img_size=config["feature_extractor_config"]["input_size"],
    )
    data_val = HandWritingDataset(
        data_path=get_dataset_folder_path() / "pre_processed" / "validation.csv",
        img_size=config["feature_extractor_config"]["input_size"],
    )

    converter = LabelConverter(character_set=data_train.char_set, max_text_length=data_train.max_length)

    model = HandwritingRecognitionModel(
        image_feature_extractor_name=config["feature_extractor_config"]["model_name"],
        num_classes=len(converter.characters),
        max_text_length=data_train.max_length,
        lstm_hidden_size=config["lstm_hidden_size"],
        attention_hidden_size=config["attention_hidden_size"],
    )

    optimizer = get_optimizer(model=model, optim_config=config["optim_config"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            saved_model = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

            start_epoch = saved_model["epoch"] + 1
            model.load_state_dict(saved_model["state"])
            optimizer.load_state_dict(saved_model["optim_state"])

            converter = LabelConverter(
                character_set=saved_model["character_set"], max_text_length=saved_model["max_text_length"]
            )
            config = saved_model["config"]
            best_val_loss = saved_model["best_val_loss"]

            all_val_loss = saved_model["all_val_loss"]
            all_train_loss = saved_model["all_train_loss"]
            all_wer = saved_model["all_wer"]
            all_cer = saved_model["all_cer"]
    else:
        start_epoch = 1
        best_val_loss = float("inf")
        all_val_loss = []
        all_train_loss = []
        all_wer = []
        all_cer = []

    print("Training with config: \n", json.dumps(config))

    model = model.to(get_device())
    loss_function = CrossEntropyLoss(ignore_index=0).to(get_device())

    scheduler = get_scheduler(optimizer=optimizer, scheduler_config=config["scheduler_config"])

    train_loader = DataLoader(
        data_train,
        batch_size=config["batch_size"],
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        data_val,
        batch_size=config["batch_size"],
        pin_memory=False,
        shuffle=True,
        drop_last=False,
    )

    epochs_since_best_loss = 0
    for epoch in range(start_epoch, config["max_epochs"]):
        current_train_loss = _single_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            train_loader=train_loader,
            config=config,
            converter=converter,
        )
        all_train_loss.append(current_train_loss)

        validation_loss, character_error_rate, word_error_rate, _, _ = _evaluate(
            epoch=epoch, model=model, data_loader=val_loader, converter=converter, loss_function=loss_function
        )

        all_val_loss.append(validation_loss)
        all_cer.append(character_error_rate)
        all_wer.append(word_error_rate)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_since_best_loss = 0
        else:
            epochs_since_best_loss += 1

        print(
            f"Validation Loss at epoch {epoch}: {validation_loss:.4f}, WER: {word_error_rate}, CER: {character_error_rate}. Best Loss so far: {best_val_loss}, Epochs Since Best Loss: {epochs_since_best_loss}"
        )

        checkpoint_data = {
            "epoch": epoch,
            "state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "current_val_loss": validation_loss,
            "config": config,
            "character_set": converter.original_character_set,
            "max_text_length": converter.max_text_length,
            "current_train_loss": current_train_loss,
            "val_character_error_rate": character_error_rate,
            "val_word_error_rate": word_error_rate,
            "best_val_loss": best_val_loss,
            "all_train_loss": all_train_loss,
            "all_val_loss": all_val_loss,
            "all_cer": all_cer,
            "all_wer": all_wer,
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            out_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint_data, out_path)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"val_loss": validation_loss, "character_error_rate": character_error_rate}, checkpoint=checkpoint
            )

        if epochs_since_best_loss > config["early_stopping_threshold"]:
            print(f"Stopping training at epoch {epoch} because of early stopping")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="default_config", type=str)
    parser.add_argument("--out_dir", default="model_outputs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    train(**vars(args))

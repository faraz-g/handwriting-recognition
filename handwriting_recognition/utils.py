import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel


def get_dataset_folder_path():
    return Path(__file__).parent.parent / "dataset"


class OptimizerConfig(BaseModel):
    optim_type: str
    learning_rate: float
    momentum: float | None
    weight_decay: float | None
    beta1: float | None
    beta2: float | None


class SchedulerConfig(BaseModel):
    scheduler_type: str
    params: dict[str, Any]


class FeatureExtractorConfig(BaseModel):
    model_name: str
    input_size: int


class TrainingConfig(BaseModel):
    seed: int
    batch_size: int
    batches_per_epoch: int
    max_epochs: int
    early_stopping_threshold: int
    optim_config: OptimizerConfig
    scheduler_config: SchedulerConfig | None
    feature_extractor_config: FeatureExtractorConfig
    lstm_hidden_size: int
    max_text_length: int | None = None
    num_classes: int | None = None

    @classmethod
    def from_path(cls, config_path: str) -> "TrainingConfig":
        with open(config_path) as f:
            config_dict = json.load(f)

        return cls(**config_dict)

import inspect
import json
from abc import ABC
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def get_dataset_folder_path():
    return Path(__file__).parent.parent / "dataset"


@dataclass
class BaseConfig(ABC):
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, config_path: str) -> "BaseConfig":
        with open(config_path) as f:
            config_dict = json.load(f)

        # TODO
        # all_dataclass_instances = []
        # for attr in inspect.getmembers(cls, lambda x: not inspect.isroutine(x)):
        #     attr_name = attr[0]
        #     if attr_name.startswith("__") and attr_name.endswith("__"):
        #         continue

        #     if is_dataclass(getattr(cls, attr_name)):
        #         all_dataclass_instances.append(attr_name)

        return cls(**config_dict)


class OptimizerConfig(BaseConfig):
    optim_type: str
    learning_rate: float
    momentum: float
    weight_decay: float


class SchedulerConfig(BaseConfig):
    scheduler_type: str
    params: dict[str, Any]


class FeatureExtractorConfig(BaseConfig):
    hf_model_name: str
    hf_pre_processor_name: str


class TrainingConfig(BaseConfig):
    seed: int
    batch_size: int
    batches_per_epoch: int
    max_epochs: int
    evaluation_frequency: int
    early_stopping_threshold: int
    optim_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    feature_extractor_config: FeatureExtractorConfig

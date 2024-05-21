import torch
from torch import nn
from handwriting_recognition.utils import TrainingConfig
from handwriting_recognition.model.bidirectional_lstm import BidirectionalLSTM
from handwriting_recognition.model.attention import Attention

LSTM_INPUT_SIZE_MAP = {"resnet34": 49, "vit_base_patch16_224.augreg2_in21k_ft_in1k": 768}


class HandwritingRecognitionModel(torch.nn.Module):
    def __init__(self, image_feature_extractor: nn.Module, training_config: TrainingConfig):
        super(HandwritingRecognitionModel, self).__init__()

        self.training_config = training_config

        # Image Feature Extraction
        self.image_feature_extractor = image_feature_extractor

        # Sequence Modelling
        self.sequence_modelling = nn.Sequential(
            BidirectionalLSTM(
                LSTM_INPUT_SIZE_MAP[training_config.feature_extractor_config.model_name],
                self.training_config.lstm_hidden_size,
                self.training_config.lstm_hidden_size,
            ),
            BidirectionalLSTM(
                self.training_config.lstm_hidden_size,
                self.training_config.lstm_hidden_size,
                self.training_config.lstm_hidden_size,
            ),
        )
        self.sequence_modelling_out_size = self.training_config.lstm_hidden_size

        # Prediction
        self.prediction = Attention(
            input_size=self.sequence_modelling_out_size,
            hidden_size=self.sequence_modelling_out_size,
            num_classes=training_config.num_classes,
        )

    def forward(self, x, y, is_train):
        image_features = self.image_feature_extractor.forward_features(x)

        if self.training_config.feature_extractor_config.model_name == "resnet34":
            image_features = image_features.view(image_features.shape[0], image_features.shape[1], -1)

        lstm_features = self.sequence_modelling(image_features)

        prediction = self.prediction(
            lstm_features.contiguous(), y, is_train, batch_max_length=self.training_config.max_text_length
        )

        return prediction

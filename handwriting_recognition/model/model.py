import torch
from torch import nn
from handwriting_recognition.utils import TrainingConfig
from handwriting_recognition.model.bidirectional_lstm import BidirectionalLSTM
from handwriting_recognition.model.attention import Attention


class HandwritingRecognitionModel(torch.nn.Module):
    def __init__(self, image_feature_extractor: nn.Module, training_config: TrainingConfig):
        super(HandwritingRecognitionModel, self).__init__()

        self.training_config = training_config

        # Image Feature Extraction
        self.image_feature_extractor = image_feature_extractor
        self.img_hidden_size = self.image_feature_extractor.embed_dim

        # Sequence Modelling
        self.sequence_modelling = nn.Sequential(
            BidirectionalLSTM(
                self.img_hidden_size, self.training_config.lstm_hidden_size, self.training_config.lstm_hidden_size
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
            self.sequence_modelling_out_size,
            self.training_config.lstm_hidden_size,
            num_classes=training_config.max_text_length,
        )

    def forward(self, x, y, is_train):
        image_features = self.image_feature_extractor.forward_features(x)

        lstm_features = self.sequence_modelling(image_features)

        prediction = self.prediction(
            lstm_features.contiguous(), y, is_train, batch_max_length=self.training_config.max_text_length
        )

        return prediction

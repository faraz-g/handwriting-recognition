import torch
from transformers import ViTImageProcessor, ViTModel
from torch import nn
from handwriting_recognition.utils import TrainingConfig
from handwriting_recognition.model.bidirectional_lstm import BidirectionalLSTM
from handwriting_recognition.model.attention import Attention


class HandwritingRecognitionModel(torch.nn.Module):
    def __init__(self, image_feature_extractor: ViTModel, training_config: TrainingConfig):
        super(HandwritingRecognitionModel, self).__init__()

        self.training_config = training_config

        # Image Feature Extraction
        self.image_feature_extractor = image_feature_extractor
        # self.image_processor = image_processor

        self.img_hidden_size = image_feature_extractor.config.hidden_size

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
        outputs = self.image_feature_extractor(pixel_values=x)
        image_embeddings = outputs.last_hidden_state[:, 0]

        lstm_features = self.sequence_modelling(image_embeddings)

        prediction = self.prediction(
            lstm_features.contiguous(), y, is_train, batch_max_length=self.training_config.max_text_length
        )

        return prediction

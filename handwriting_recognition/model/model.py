import torch
from torch import nn
from handwriting_recognition.model.bidirectional_lstm import BidirectionalLSTM
from handwriting_recognition.model.attention import Attention
from handwriting_recognition.modelling_utils import get_image_model

LSTM_INPUT_SIZE_MAP = {"resnet34": 49, "vit_base_patch16_224.augreg2_in21k_ft_in1k": 768}


class HandwritingRecognitionModel(torch.nn.Module):
    def __init__(
        self,
        image_feature_extractor_name: str,
        num_classes: int,
        max_text_length: int,
        lstm_hidden_size: int = 256,
        attention_hidden_size: int = 256,
    ):
        super(HandwritingRecognitionModel, self).__init__()

        self.image_feature_extractor_name = image_feature_extractor_name
        self.num_classes = num_classes
        self.max_text_length = max_text_length

        # Image Feature Extraction
        self.image_feature_extractor = get_image_model(model_name=image_feature_extractor_name)

        # Sequence Modelling
        self.sequence_modelling = nn.Sequential(
            BidirectionalLSTM(
                LSTM_INPUT_SIZE_MAP[image_feature_extractor_name],
                lstm_hidden_size,
                lstm_hidden_size,
            ),
            BidirectionalLSTM(
                lstm_hidden_size,
                lstm_hidden_size,
                lstm_hidden_size,
            ),
        )
        self.sequence_modelling_out_size = lstm_hidden_size

        # Prediction
        self.prediction = Attention(
            input_size=self.sequence_modelling_out_size,
            hidden_size=attention_hidden_size,
            num_classes=self.num_classes,
        )

    def forward(self, x, y, is_train):
        image_features = self.image_feature_extractor.forward_features(x)

        if self.image_feature_extractor_name == "resnet34":
            image_features = image_features.view(image_features.shape[0], image_features.shape[1], -1)

        lstm_features = self.sequence_modelling(image_features)

        prediction = self.prediction(lstm_features.contiguous(), y, is_train, batch_max_length=self.max_text_length)

        return prediction

import torch
from transformers import ViTModel


# TODO
class HandwritingRecognitionModel(torch.nn.Module):
    def __init__(self, image_feature_extractor: ViTModel):
        super(HandwritingRecognitionModel, self).__init__()
        self.image_feature_extractor = image_feature_extractor

    def forward(self, x, y=None):
        self.last_hidden_states = self.image_feature_extractor(**x)

        # TODO

        return self.last_hidden_states  # This is temporary. DELTE ME

    def predict(self, x):
        self.last_hidden_states = self.image_feature_extractor(**x)

        # TODO

        return self.last_hidden_states  # This is temporary. DELTE ME

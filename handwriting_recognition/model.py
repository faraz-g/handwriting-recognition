import torch
from transformers import ViTModel


# TODO
class HandwritingRecognitionModel(torch.nn.Module):
    def __init__(self, image_feature_extractor: ViTModel):
        self.image_feature_extractor = image_feature_extractor
        pass

    def forward(self):
        raise NotImplementedError

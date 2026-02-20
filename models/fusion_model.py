import torch
import torch.nn as nn
from torchvision import models


class FusionModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(FusionModel, self).__init__()

        self.cnn = models.resnet18(weights=None)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)
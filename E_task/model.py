import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class OxfordPetsModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
import torch
import torch.nn as nn
from torchvision import models

class CustomEffnet(nn.Module):
    def __init__(self, pretrained=True, num_classes=4) -> None:
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=4, bias=True)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
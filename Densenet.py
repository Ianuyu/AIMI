import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DenseNet121(nn.Module):

    def __init__(self, num_classes: int = 2, in_ch: int = 3,
                 dropout: float = 0.0, pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.DenseNet121_Weights.DEFAULT
            except Exception:
                weights = None  

        backbone = models.densenet121(weights=weights, drop_rate=0.0)
        if in_ch != 3:
            old = backbone.features.conv0
            backbone.features.conv0 = nn.Conv2d(
                in_ch, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False
            )

        self.features = backbone.features
        self.num_features = backbone.classifier.in_features
        self.dropout_p = float(dropout)
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.classifier(x)
        return x

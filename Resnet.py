import torch
import torch.nn as nn

def DownSample(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class BasicBlock(nn.Module):
    extend = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out

class Bottleneck(nn.Module):
    extend = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.extend, 1, bias=False),
            nn.BatchNorm2d(out_channels * self.extend),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet(nn.Module):
    def __init__(self, Block, layers, num_classes, num_channels=3, dropout=0.3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(num_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(Block, layers[0], base=64,  stride=1)
        self.layer2 = self._make_layer(Block, layers[1], base=128, stride=2)
        self.layer3 = self._make_layer(Block, layers[2], base=256, stride=2)
        self.layer4 = self._make_layer(Block, layers[3], base=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 512 * Block.extend
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(50, num_classes)
        )

    def _make_layer(self, Block, blocks, base, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != base * Block.extend:
            downsample = DownSample(self.in_channels, base * Block.extend, stride)
        layers.append(Block(self.in_channels, base, downsample=downsample, stride=stride))
        self.in_channels = base * Block.extend
        for _ in range(blocks - 1):
            layers.append(Block(self.in_channels, base))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(num_classes: int, in_ch: int = 3, dropout: float = 0.09, pretrained: bool = False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_channels=in_ch, dropout=dropout)

def ResNet50(num_classes: int, in_ch: int = 3, dropout: float = 0.09, pretrained: bool = False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_channels=in_ch, dropout=dropout)
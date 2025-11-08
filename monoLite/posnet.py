import torch
import torch.nn as nn


# First, define the BasicBlock and ResNet18 feature extractor (no final fc)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18Feat(nn.Module):
    def __init__(self, in_channels=6):  # input: concatenated img pairs
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # Feature map [B,512,H',W']


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResNet18Feat(in_channels=6)
        # Global average pooling followed by FC layers for pose
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)  # 3 for rotation, 3 for translation
        )

    def forward(self, img_pair):
        # img_pair: concatenated [B,6,H,W] frame pair
        assert not torch.isinf(img_pair).any(), 'InF value in forward PoseNet'
        assert not torch.isNaN(img_pair).any(), 'NaN value in forward PoseNet'
        feat = self.feature_extractor(img_pair)
        assert not torch.isinf(feat).any(), 'InF value in feat PoseNet'
        assert not torch.isNaN(feat).any(), 'NaN value in feat PoseNet'
        pooled = self.avgpool(feat)
        assert not torch.isinf(pooled).any(), 'InF value in pooled PoseNet'
        assert not torch.isNaN(pooled).any(), 'NaN value in pooled PoseNet'
        out = self.pose_head(pooled)
        assert not torch.isinf(out).any(), 'InF value in out PoseNet'
        assert not torch.isNaN(out).any(), 'NaN value in out PoseNet'
        # Optionally, scale translation for stability as in Monodepth2
        out[:, :3] = torch.tanh(out[:, :3])  # rotation
        out[:, 3:] = out[:, 3:] * 0.01
        return out  # [B,6]

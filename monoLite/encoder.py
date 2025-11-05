from torch import nn
import torch
import torch.functional as F
from cdc import CDCBlock
from lgfi import LGFIBlock


class LiteMonoEncoder(nn.Module):
    def __init__(self, variant="base"):  # Choose variant: 'tiny', 'small', 'base', '8M'
        super().__init__()

        # Define architectures per Table 1 (channels and block counts)
        settings = {
            "tiny":  {"channels": [32, 32, 64, 128], "blocks": [3, 3, 6, 6]},
            "small": {"channels": [48, 48, 96, 128], "blocks": [3, 3, 6, 6]},
            "base":  {"channels": [48, 48, 128, 128], "blocks": [3, 3, 9, 9]},
            "8M":    {"channels": [64, 64, 224, 224], "blocks": [3, 3, 9, 9]}
        }
        c = settings[variant]["channels"]
        n = settings[variant]["blocks"]

        # Stem: initial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.GELU(),
            nn.Conv2d(c[0], c[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.GELU(),
            nn.Conv2d(c[0], c[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.GELU()
        )

        # Stage 1 (downsampling)
        self.down1 = nn.Conv2d(c[0], c[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage1 = nn.Sequential(*[
            CDCBlock(in_channels=c[1], out_channels=c[1], dilation_rates=[1, 2, 3]) for _ in range(n[0])],
            LGFIBlock(channels=c[1]))

        # Stage 2 (downsampling)
        self.down2 = nn.Conv2d(c[1], c[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(*[
            CDCBlock(in_channels=c[2], out_channels=c[2], dilation_rates=[1, 2, 3, 2, 4, 6]) for _ in range(n[1])],
            LGFIBlock(channels=c[2]))

        # Stage 3 (downsampling)
        self.down3 = nn.Conv2d(c[2], c[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(*[
            CDCBlock(in_channels=c[3], out_channels=c[3], dilation_rates=[1, 2, 3, 2, 4, 6]) for _ in range(n[2])],
            LGFIBlock(channels=c[3]))

        # Optionally more stages depending on variant
        # Add pooled concatenation and cross-stage connections as needed

    def forward(self, x):
        # Inside your LiteMonoEncoder.forward(x):
        feats = []
        img = x

        # Feature map at 1/2 res
        x1 = self.stem(x)
        img_p1 = F.adaptive_avg_pool2d(img, (x1.shape[2], x1.shape[3]))
        # Concatenate pooled input image to feature map
        x1_cat = torch.cat([x1, img_p1], dim=1)
        # Downsample to 1/4 res
        x1_down = self.down1(x1_cat)
        x2 = self.stage1(x1_down)
        feats.append(x2)

        img_p2 = F.adaptive_avg_pool2d(img, (x2.shape[2], x2.shape[3]))
        x2_cat = torch.cat([x2, img_p2], dim=1)
        x2_down = self.down2(x2_cat)
        x3 = self.stage2(x2_down)
        feats.append(x3)

        img_p3 = F.adaptive_avg_pool2d(img, (x3.shape[2], x3.shape[3]))
        x3_cat = torch.cat([x3, img_p3], dim=1)
        x3_down = self.down3(x3_cat)
        x4 = self.stage3(x3_down)
        feats.append(x4)

        return feats

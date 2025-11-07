import torch
from torch import nn
import torch.nn.functional as F
from .cdc import CDCBlock
from .lgfi import LGFIBlock


class LiteMonoEncoder(nn.Module):
    def __init__(self, variant="base"):  # 'tiny', 'small', 'base', '8M'
        super().__init__()
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

        # STAGE 1
        self.down1 = nn.Conv2d(c[0]+3, c[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage1 = nn.Sequential(
            *[CDCBlock(channels=c[1], dilation_rates=[1, 2, 3]) for _ in range(n[0])],
            LGFIBlock(channels=c[1])
        )

        # STAGE 2
        self.down2 = nn.Conv2d(c[1]+3, c[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            *[CDCBlock(channels=c[2], dilation_rates=[1, 2, 3, 2, 4, 6]) for _ in range(n[1])],
            LGFIBlock(channels=c[2])
        )

        # STAGE 3
        self.down3 = nn.Conv2d(c[2]+3, c[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            *[CDCBlock(channels=c[3], dilation_rates=[1, 2, 3, 2, 4, 6]) for _ in range(n[2])],
            LGFIBlock(channels=c[3])
        )

    def forward(self, x):
        feats = []
        img = x  # [B, 3, H, W]
        # 1. Stem
        x1 = self.stem(x)  # [B, c[0], H/2, W/2]
        img_p1 = F.adaptive_avg_pool2d(img, (x1.shape[2], x1.shape[3]))
        x1_cat = torch.cat([x1, img_p1], dim=1)  # [B, c[0]+3, H/2, W/2]

        # 2. Stage 1
        x2_down = self.down1(x1_cat)  # [B, c[1], H/4, W/4]
        x2 = self.stage1(x2_down)     # [B, c[1], H/4, W/4]
        feats.append(x2)

        img_p2 = F.adaptive_avg_pool2d(img, (x2.shape[2], x2.shape[3]))
        x2_cat = torch.cat([x2, img_p2], dim=1)  # [B, c[1]+3, H/4, W/4]

        # 3. Stage 2
        x3_down = self.down2(x2_cat)  # [B, c[2], H/8, W/8]
        x3 = self.stage2(x3_down)     # [B, c[2], H/8, W/8]
        feats.append(x3)

        img_p3 = F.adaptive_avg_pool2d(img, (x3.shape[2], x3.shape[3]))
        x3_cat = torch.cat([x3, img_p3], dim=1)  # [B, c[2]+3, H/8, W/8]

        # 4. Stage 3
        x4_down = self.down3(x3_cat)  # [B, c[3], H/16, W/16]
        x4 = self.stage3(x4_down)     # [B, c[3], H/16, W/16]
        feats.append(x4)
        feats = feats[::-1]

        return feats

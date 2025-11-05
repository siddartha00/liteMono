import torch
from torch import nn
import torch.nn.functional as F


# --- Upsampling Block ---
class UpSampling(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


# --- Prediction Head ---
class PredHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        # Inverse depth maps are predicted with a sigmoid activation

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        x = torch.sigmoid(x)  # Output values in (0,1)
        return x


# --- LiteMono Decoder ---
class LiteMonoDecoder(nn.Module):
    def __init__(self, encoder_channels):
        """
        encoder_channels: list of int, number of channels from encoder's deepest, mid, shallow stages.
        Typically ordered deepest to shallowest, e.g. [C3, C2, C1] (lowest res to highest).
        """
        super().__init__()
        # Set upsampling block channels (these should match your encoder output sizes)
        self.up1 = UpSampling(encoder_channels[0], encoder_channels[1], encoder_channels[1])
        self.head1 = PredHead(encoder_channels[1])  # 1/4 resolution

        self.up2 = UpSampling(encoder_channels[1], encoder_channels[2], encoder_channels[2])
        self.head2 = PredHead(encoder_channels[2])  # 1/2 resolution

        self.up3 = UpSampling(encoder_channels[2], 0, encoder_channels[2] // 2)  # to full size
        self.head3 = PredHead(encoder_channels[2] // 2)  # full resolution

    def forward(self, encoder_feats):
        """
        encoder_feats: list of 3 feature tensors, from low-resolution to high-resolution (deep to shallow)
        """
        x3, x2, x1 = encoder_feats  # e.g., [B, C3, H/8, W/8], [B, C2, H/4, W/4], [B, C1, H/2, W/2]

        x = self.up1(x3, x2)  # Upsample x3 and concat with x2
        pred1 = self.head1(x)  # 1/4 resolution

        x = self.up2(x, x1)  # Upsample and concat with x1
        pred2 = self.head2(x)  # 1/2 resolution

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Up to full
        x = self.up3(x, None) if hasattr(self.up3, 'forward') and self.up3 is not None else x
        pred3 = self.head3(x)  # Full resolution estimate

        # Return all scales for loss computation as in the paper
        return [pred3, pred2, pred1]  # [full, 1/2, 1/4 resolution]

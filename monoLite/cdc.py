import torch
import torch.nn as nn


class CDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3]):
        super(CDCBlock, self).__init__()
        self.layers = nn.ModuleList()
        for dil in dilation_rates:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),  # pointwise conv
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=dil,
                              dilation=dil,
                              groups=out_channels,
                              bias=False),  # depthwise dilated conv
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()
                )
            )
            # Optionally update in_channels for next block if you want output stacking/concatenation

    def forward(self, x):
        # Each branch processes the same input and outputs its own feature map
        outputs = [branch(x) for branch in self.layers]
        # Concatenate along the channel dimension
        stacked = torch.cat(outputs, dim=1)  # Shape: [B, out_channels * len(dilation_rates), H, W]
        return stacked

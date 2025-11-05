from torch import nn


class CDCBlock(nn.Module):
    def __init__(self, channels, dilation_rates=[1, 2, 3]):
        super().__init__()
        self.layers = nn.ModuleList()
        for dil in dilation_rates:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dil,
                              dilation=dil, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU()
                )
            )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) + out  # Residual addition (not concatenation)
        return out

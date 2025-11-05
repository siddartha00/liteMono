import torch
import torch.nn as nn
import torch.nn.functional as F


class LGFIBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Local path
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        # Global path: channel-wise self-attention (cross-covariance attention)
        self.channel_query = nn.Linear(channels, channels, bias=False)
        self.channel_key = nn.Linear(channels, channels, bias=False)
        self.channel_value = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Local features path
        local_feat = self.local_conv(x)

        # Global features path (cross-channel attention)
        # Flatten spatial dimensions
        y = x.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        # Linear projection for Q, K, V
        Q = self.channel_query(y)
        K = self.channel_key(y)
        V = self.channel_value(y)
        # Split into heads
        Q = Q.view(B,
                   -1,
                   self.num_heads,
                   C // self.num_heads).transpose(1, 2)  # (B, heads, HW, C//heads)
        K = K.view(B,
                   -1,
                   self.num_heads,
                   C // self.num_heads).transpose(1, 2)
        V = V.view(B,
                   -1,
                   self.num_heads,
                   C // self.num_heads).transpose(1, 2)
        # Channel-wise attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C // self.num_heads) ** 0.5  # (B, heads, HW, HW)
        attn = F.softmax(attn_scores, dim=-1)
        global_feat = torch.matmul(attn, V)  # (B, heads, HW, C//heads)
        global_feat = global_feat.transpose(1, 2).contiguous().view(B, -1, C)  # (B, HW, C)
        # Reshape back to spatial
        global_feat = global_feat.transpose(1, 2).view(B, C, H, W)
        global_feat = self.norm(global_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LN over channels

        # Fuse paths (add or concatenate; here, add)
        out = local_feat + global_feat
        return out

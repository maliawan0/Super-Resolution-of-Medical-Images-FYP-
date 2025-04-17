import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        t = t.float().unsqueeze(-1) / 1000
        return self.embed(t)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        # Time embeddings for decoder & bottleneck
        self.temb_mid = TimeEmbedding(base_channels * 8)
        self.temb_dec3 = TimeEmbedding(base_channels * 4)
        self.temb_dec2 = TimeEmbedding(base_channels * 2)
        self.temb_dec1 = TimeEmbedding(base_channels)

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.middle = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        B = x.size(0)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        mid = self.middle(self.pool(enc3))

        # Add time embedding only at decoder/mid
        mid += self.temb_mid(t).view(B, -1, 1, 1)

        # Decoder
        # Decoder
        mid = self.middle(self.pool(enc3))
        mid = mid + self.temb_mid(t).view(B, -1, 1, 1)

        dec3 = self.up3(mid)
        dec3 = dec3 + self.temb_dec3(t).view(B, -1, 1, 1)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)
        dec2 = dec2 + self.temb_dec2(t).view(B, -1, 1, 1)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = dec1 + self.temb_dec1(t).view(B, -1, 1, 1)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))


        return self.out(dec1)

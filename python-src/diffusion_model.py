import math
import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class UNet2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Timestep embedding dimension matches bottleneck channels
        self.time_embed_dim = 512
        self.time_embedding = SinusoidalPosEmb(self.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.ReLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Encoder blocks - takes in 4 channels (eps, mu, src, diffusion)
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Decoder blocks with skip connections
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 512 = 256 (upconv) + 256 (skip)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 = 128 (upconv) + 128 (skip)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 = 64 (upconv) + 64 (skip)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Output single channel for diffusion image
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, eps, mu, src, diffusion, t, omega):
        # Combine inputs into 4 channels: [B,1,H,W] each -> [B,4,H,W]
        x = torch.cat([eps, mu, src, diffusion], dim=1)  # [B,4,H,W]

        # Encoder path with skip connections
        enc1 = self.enc1(x)  # [B,64,H,W]
        pool1 = self.pool1(enc1)  # [B,64,H/2,W/2]

        enc2 = self.enc2(pool1)  # [B,128,H/2,W/2]
        pool2 = self.pool2(enc2)  # [B,128,H/4,W/4]

        enc3 = self.enc3(pool2)  # [B,256,H/4,W/4]
        pool3 = self.pool3(enc3)  # [B,256,H/8,W/8]

        # Bottleneck
        bottleneck = self.bottleneck(pool3)  # [B,512,H/8,W/8]

        # Add timestep embedding to bottleneck
        t_emb = self.time_embedding(t)  # [B,512]
        t_emb = self.time_mlp(t_emb)  # [B,512]
        # Reshape t_emb to match bottleneck spatial dimensions
        t_emb = t_emb.view(-1, self.time_embed_dim, 1, 1)  # [B,512,1,1]
        t_emb = t_emb.expand(
            -1, -1, bottleneck.shape[2], bottleneck.shape[3]
        )  # [B,512,H/8,W/8]
        bottleneck = bottleneck + t_emb  # [B,512,H/8,W/8]

        # Decoder path with skip connections
        up3 = self.upconv3(bottleneck)  # [B,256,H/4,W/4]
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))  # [B,256,H/4,W/4]

        up2 = self.upconv2(dec3)  # [B,128,H/2,W/2]
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))  # [B,128,H/2,W/2]

        up1 = self.upconv1(dec2)  # [B,64,H,W]
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))  # [B,64,H,W]

        # Output diffusion image
        out = self.final(dec1)  # [B,1,H,W]

        # Omega is accepted as an input but not used internally
        return out

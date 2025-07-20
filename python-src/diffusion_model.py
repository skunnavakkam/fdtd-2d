import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device))
            * torch.arange(half_dim, device=device)
            / (half_dim - 1)
        )
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def make_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet2DModel(nn.Module):
    """
    • Time embedding is now injected at *every* scale (encoder + decoder)
    • Shape handling unchanged, so downstream code keeps working.
    """

    def __init__(self, time_embed_dim: int = 512):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # ── Embeddings ────────────────────────────────────────────────────────
        self.time_embedding = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # ω-specific embeddings (one per scale)
        self.omega_emb1 = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        self.omega_emb2 = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128)
        )
        self.omega_emb3 = nn.Sequential(
            nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 256)
        )

        # ── UNet ──────────────────────────────────────────────────────────────
        self.enc1 = make_block(4, 64)
        self.enc2 = make_block(64, 128)
        self.enc3 = make_block(128, 256)

        self.bottleneck = make_block(256, 512)

        self.dec3 = make_block(512 + 256, 256)
        self.dec2 = make_block(256 + 128, 128)
        self.dec1 = make_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _time_map(self, t: torch.Tensor, spatial: Tuple[int, int]) -> torch.Tensor:
        """Broadcast time embedding to a given H×W."""
        emb = self.time_mlp(self.time_embedding(t.float()))  # [B,C]
        B, C = emb.shape
        H, W = spatial
        return emb.view(B, C, 1, 1).expand(B, C, H, W)

    def _omega_map(
        self, ω: torch.Tensor, spatial: Tuple[int, int], net
    ) -> torch.Tensor:
        emb = net(ω.unsqueeze(-1))  # [B,C]
        B, C = emb.shape
        H, W = spatial
        return emb.view(B, C, 1, 1).expand(B, C, H, W)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, eps, mu, src, diffusion, t, omega):
        """
        • No time map at the raw input or decoder scales (shape mismatch);
        time is added only at the 512-ch bottleneck where it fits.
        • Ω embeddings unchanged.
        """
        # Concatenate physical channels: [B,4,H,W]
        x = torch.stack([eps, mu, src, diffusion], dim=1)

        # ─ Encoder 1
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        p1 = p1 + self._omega_map(omega, p1.shape[-2:], self.omega_emb1)

        # ─ Encoder 2
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        p2 = p2 + self._omega_map(omega, p2.shape[-2:], self.omega_emb2)

        # ─ Encoder 3
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)
        p3 = p3 + self._omega_map(omega, p3.shape[-2:], self.omega_emb3)

        # ─ Bottleneck  (time embedding fits here: 512→512)
        b = self.bottleneck(p3)
        b = b + self._time_map(t, b.shape[-2:])

        # ─ Decoder 3
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        # ─ Decoder 2
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # ─ Decoder 1
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1).squeeze(1)

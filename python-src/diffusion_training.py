import torch
from typing import Tuple
from fdfd import make_A
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


"""
This is inspired by DiffusionPDE. We solve FDFD and use that for denoising diffusion training.

We model FDFD as a function f(z, b) where z is the permittivity array and b is the source array.

f(z, b) |-> sol

We then train a model to predict sol, given z and b.

and z, b given sol. 
"""


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
    def __init__(self, time_embed_dim: int = 512):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        # time embedding
        self.time_embedding = SinusoidalPosEmb(self.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        # omega embedding (scalar -> vector)
        self.omega_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # encoder
        self.enc1 = make_block(4, 64)
        self.enc2 = make_block(64, 128)
        self.enc3 = make_block(128, 256)
        # bottleneck
        self.bottleneck = make_block(256, 512)
        # decoder
        self.dec3 = make_block(512 + 256, 256)
        self.dec2 = make_block(256 + 128, 128)
        self.dec1 = make_block(128 + 64, 64)
        # final output
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def _make_time_emb(
        self, t: torch.Tensor, spatial_size: Tuple[int, int]
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        B, C = t_emb.shape
        H, W = spatial_size
        return t_emb.view(B, C, 1, 1).expand(B, C, H, W)

    def _make_omega_emb(
        self, omega: torch.Tensor, spatial_size: Tuple[int, int]
    ) -> torch.Tensor:
        # omega: [B]
        omega = omega.unsqueeze(-1)  # [B,1]
        omega_emb = self.omega_mlp(omega)  # [B, time_embed_dim]
        B, C = omega_emb.shape
        H, W = spatial_size
        return omega_emb.view(B, C, 1, 1).expand(B, C, H, W)

    def forward(
        self,
        eps: torch.Tensor,
        mu: torch.Tensor,
        src: torch.Tensor,
        diffusion: torch.Tensor,
        t: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        # concat inputs -> [B,4,H,W]
        x = torch.cat([eps, mu, src, diffusion], dim=1)
        # encoder
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)

        # bottleneck + time & omega embeddings
        b = self.bottleneck(p3)
        b = b + self._make_time_emb(t.float(), p3.shape[-2:])
        b = b + self._make_omega_emb(omega.float(), p3.shape[-2:])

        # decoder
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


def do_inference(model, eps, mu, src, omega, device):
    """
    Run the full inference pipeline using a correct reverse-diffusion sampler.

    Args:
        model:     Trained UNet model
        eps:       Permittivity tensor [H,W]
        mu:        Permeability tensor [H,W]
        src:       Source tensor [H,W]
        omega:     Angular frequency scalar
        device:    torch.device
    Returns:
        Predicted Ez field [H,W]
    """
    model.eval()

    # Prepare inputs
    eps = eps.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    mu = mu.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    src = src.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    omega = torch.tensor([omega], device=device)  # [1]

    B, _, H, W = eps.shape
    num_steps = 8  # match training schedule

    # Build noise schedule (we only need alphas)
    _, alphas = generate_diffusion_data(
        torch.zeros((H, W), device=device), num_steps, device=device
    )  # returns (noisy, alphas)

    with torch.no_grad():
        # Initialize with pure noise
        x_t = torch.randn((B, 1, H, W), device=device)

        # Reverse diffusion: t = T-1 ... 1
        for t in reversed(range(1, num_steps)):
            t_tensor = torch.tensor([t], device=device).float()

            # Predict x0
            x0_pred = model(eps, mu, src, x_t, t_tensor, omega)

            # Sample noise for this step
            noise = torch.randn_like(x_t)

            # Compute previous alpha
            alpha_prev = alphas[t - 1]

            # Update x_{t-1}
            x_t = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise

        # Final step (t=0): directly predict clean field
        x0 = model(eps, mu, src, x_t, torch.tensor([0], device=device).float(), omega)

    return x0.squeeze()  # [H, W]


def run_fdfd(eps: torch.Tensor, mu: torch.Tensor, source: torch.Tensor, dx, omega):
    assert eps.shape == mu.shape == source.shape, (
        f"Shape mismatch: {eps.shape} != {mu.shape} != {source.shape}"
    )
    # Convert torch tensors to numpy arrays
    eps_np = eps.cpu().numpy()
    mu_np = mu.cpu().numpy()
    source_np = source.cpu().numpy()

    # Run FDFD in numpy
    A = make_A(eps_np, mu_np, dx, dx, *eps_np.shape, omega)
    b = -1j * source_np.flatten() * omega

    # Solve and convert back to torch tensor
    solution_np = scipy.sparse.linalg.spsolve(A, b).reshape(eps_np.shape).real
    return torch.from_numpy(solution_np).to(torch.float32).to(eps.device)


def generate_random_permittivity(
    dimension: Tuple[int, int],
    device: torch.device = torch.device("mps"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a permittivity field (ε) with diverse patterns including:
    - Random binary regions (ε₀ or 5·ε₀)
    - Smooth gradients
    - Periodic structures
    - Random shapes

    Features are generated at lower frequencies to create larger-scale patterns.
    """
    # Physical constants
    eps_0 = 8.85418782e-12  # vacuum permittivity (F/m)
    eps_max = 5 * eps_0
    mu_0 = 1.25663706e-6  # vacuum permeability (H/m)

    # Binary regions with larger Gaussian correlation
    eps_binary = torch.rand(dimension, dtype=dtype, device=device)
    kernel_size = 15  # Increased kernel size
    sigma = torch.rand(1).item() * 4.0 + 2.0  # Larger sigma range
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size // 2)
    xg, yg = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xg**2 + yg**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    weight = kernel.unsqueeze(0).unsqueeze(0)

    # Add padding to maintain output size
    padding = kernel_size // 2
    eps_blurred = F.conv2d(
        eps_binary.unsqueeze(0).unsqueeze(0), weight, padding=padding
    )[0, 0]
    eps = (eps_blurred > 0.5).float() * (eps_max - eps_0) + eps_0

    # Uniform permeability field
    mu = torch.full(dimension, mu_0, dtype=dtype, device=device)

    return eps, mu


def generate_random_source(
    dimension: Tuple[int, int], dtype=torch.float32, device=torch.device("mps")
) -> torch.Tensor:
    """
    Generate a random source configuration with either line or point sources.
    Returns a tensor of shape dimension with sources placed randomly.
    Line sources are less than 10% of the dimension length.
    Sources are placed outside the outer 5 pixels and within the middle 80%.
    """
    source = torch.zeros(dimension, dtype=dtype, device=device)

    # Calculate valid regions (outside 5 pixels, within middle 80%)
    margin = 5
    start_x = margin
    end_x = dimension[0] - margin
    start_y = margin
    end_y = dimension[1] - margin

    # Further restrict to middle 80%
    mid_margin_x = int(dimension[0] * 0.1)  # 10% from each side
    mid_margin_y = int(dimension[1] * 0.1)
    start_x = max(start_x, mid_margin_x)
    end_x = min(end_x, dimension[0] - mid_margin_x)
    start_y = max(start_y, mid_margin_y)
    end_y = min(end_y, dimension[1] - mid_margin_y)

    # Calculate max line length (10% of valid region)
    max_line_length = min(end_x - start_x, end_y - start_y) // 10

    # Randomly choose between line or point source
    if torch.rand(1).item() < 0.5:  # 50% chance for each type
        # Line source
        # Randomly choose horizontal or vertical line
        if torch.rand(1).item() < 0.5:
            # Horizontal line
            row = torch.randint(start_x, end_x, (1,)).item()
            start = torch.randint(start_y, end_y - max_line_length, (1,)).item()
            source[row, start : start + max_line_length] = 1.0
        else:
            # Vertical line
            col = torch.randint(start_y, end_y, (1,)).item()
            start = torch.randint(start_x, end_x - max_line_length, (1,)).item()
            source[start : start + max_line_length, col] = 1.0
    else:
        # Point source
        # Random position within valid region
        row = torch.randint(start_x, end_x, (1,)).item()
        col = torch.randint(start_y, end_y, (1,)).item()
        source[row, col] = 1.0

    return source


def generate_data(num_samples: int, dimension: Tuple[int, int]):
    """Generate training data samples.

    Args:
        num_samples: Number of samples to generate
        dimension: Tuple of (height, width) dimensions

    Returns:
        List of (((permittivity, mu), source, omega), Ez) tuples
    """
    eps_samples = []
    mu_samples = []
    src_samples = []
    omega_samples = []
    Ez_samples = []

    for _ in trange(num_samples):
        # Generate random permittivity and mu
        eps, mu = generate_random_permittivity(dimension)

        # Generate random source configuration
        src = generate_random_source(dimension)

        # Generate random frequency between 9-30 GHz
        omega = torch.rand(1, dtype=torch.float32).item() * (30e9 - 18e9) + 18e9

        Ez = run_fdfd(eps, mu, src, 1e-3, omega)

        eps_samples.append(eps)
        mu_samples.append(mu)
        src_samples.append(src)
        omega_samples.append(omega)
        Ez_samples.append(Ez)

    return (
        torch.stack(eps_samples),
        torch.stack(mu_samples),
        torch.stack(src_samples),
        torch.tensor(omega_samples, dtype=torch.float32),
        torch.stack(Ez_samples),
    )


def plot_example():
    dim = (60, 60)
    eps, mu = generate_random_permittivity(dim)
    src = generate_random_source(dim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(eps)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("Permittivity")

    im2 = ax2.imshow(src)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("Source")

    plt.tight_layout()
    plt.show()


def generate_diffusion_data(
    Ez_samples: torch.Tensor,
    num_steps: int,
    device: torch.device = torch.device("mps"),
    schedule: str = "cosine",
    noise_scale: float = 0.1,
    noise_mean: float = 0.0,
):
    """
    Generate a full diffusion trajectory with:
      • scalar indexing of αₜ (no shape mismatches)
      • warp_factor to delay noise
      • noise_mean & noise_scale knobs
      • automatic batching for [H,W] or [B,H,W] inputs

    Args:
        Ez_samples:  [H,W] or [B,H,W] clean Ez field
        num_steps:   number of diffusion steps
        device:      torch device
        schedule:    "cosine" or "linear"
        warp_factor: >1 stretches out early low-noise region
        noise_scale: multiplies noise std
        noise_mean:  shifts noise mean
    Returns:
        noisy:  [T,H,W] if input was [H,W], else [B,T,H,W]
        alphas: [T] cumulative α schedule
    """
    Ez = Ez_samples.to(device).to(torch.float32)
    orig_ndim = Ez.ndim

    # ensure batch dim
    if Ez.ndim == 2:
        Ez = Ez.unsqueeze(0)  # [1,H,W]
    elif Ez.ndim != 3:
        raise ValueError(f"Expected 2D or 3D Ez_samples, got {Ez.shape}")
    B, H, W = Ez.shape

    # 1) build raw α on [0,1]
    if schedule == "cosine":
        s = 0.5
        t_lin = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32)
        raw = torch.cos(((t_lin + s) / (1 + s)) * (torch.pi / 2)).pow(2)
    elif schedule == "linear":
        betas = torch.linspace(0.0, 0.01, num_steps, device=device, dtype=torch.float32)
        raw = torch.cumprod(1 - betas, dim=0)
    else:
        raise ValueError(f"Unknown schedule {schedule!r}")

    alphas = raw

    # 3) diffusion loop with scalar αₜ
    noisy = torch.zeros((B, num_steps, H, W), device=device, dtype=torch.float32)
    for t in range(num_steps):
        if t == 0:
            noisy[:, t] = Ez
        else:
            α_t = alphas[t]  # zero‐dim tensor
            noise = noise_scale * torch.randn_like(Ez, dtype=torch.float32) + noise_mean
            noisy[:, t] = α_t.sqrt() * Ez + (1 - α_t).sqrt() * noise

    # drop batch dim if input was 2D
    if orig_ndim == 2:
        noisy = noisy.squeeze(0)  # [T,H,W]

    return noisy, alphas


def plot_noisy_sample(noisy_sample: torch.Tensor):
    """
    Plot a noisy sample across all timesteps in a grid.

    Args:
        noisy_sample: Tensor of shape [T, H, W] containing noisy samples at each timestep
    """
    num_timesteps = noisy_sample.shape[0]
    rows = 1
    cols = num_timesteps

    plt.figure(figsize=(16, 6))

    for t in range(num_timesteps):
        plt.subplot(rows, cols, t + 1)
        plt.imshow(noisy_sample[t].cpu().numpy(), cmap="bwr", vmin=-0.5, vmax=0.5)
        plt.title(f"t={t}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def train_model(
    model, optimizer, dataloader, num_steps, num_epochs, device, weight_decay=1e-4
):
    """
    Training loop for the diffusion-based PDE solver.
    Args:
        model:          UNet2DModel instance
        optimizer:      torch optimizer
        dataloader:     DataLoader yielding (eps, mu, src, Ez, omega)
        num_steps:      Number of diffusion timesteps
        num_epochs:     Number of epochs to train
        device:         torch device
    """
    criterion = torch.nn.MSELoss()
    model.train()

    # Get a sample from dataloader for inference
    sample_eps, sample_mu, sample_src, sample_Ez, sample_omega = next(iter(dataloader))
    sample_eps = sample_eps[0].to(device)  # Take first sample
    sample_mu = sample_mu[0].to(device)
    sample_src = sample_src[0].to(device)
    sample_Ez = sample_Ez[0].to(device)
    sample_omega = sample_omega[0].to(device)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for eps, mu, src, Ez, omega in dataloader:
            # Move tensors to device
            eps = eps.to(device).to(torch.float32)
            mu = mu.to(device).to(torch.float32)
            src = src.to(device).to(torch.float32)
            Ez = Ez.to(device).to(torch.float32)
            omega = omega.to(device).to(torch.float32)

            # Generate diffusion trajectory for clean Ez
            noisy_traj, alphas = generate_diffusion_data(Ez, num_steps, device=device)

            # Batch size and device info
            if noisy_traj.ndim == 4:
                B = noisy_traj.shape[0]
            else:
                noisy_traj = noisy_traj.unsqueeze(0)
                B = 1

            # Sample random timesteps t for each sample
            t = torch.randint(0, num_steps, (B,), device=device)

            # Extract noisy input at sampled t
            x_t = noisy_traj[torch.arange(B), t]  # [B, H, W]
            diffusion = x_t.unsqueeze(1)  # add channel dim -> [B, 1, H, W]

            # Forward pass: model expects eps, mu, src, diffusion, t, omega
            pred = model(
                eps.unsqueeze(1),  # [B,1,H,W]
                mu.unsqueeze(1),  # [B,1,H,W]
                src.unsqueeze(1),  # [B,1,H,W]
                diffusion,  # [B,1,H,W]
                t.float(),  # [B]
                omega,  # [B]
            )  # returns [B,1,H,W]

            # Compute loss: predict clean Ez
            target = Ez.unsqueeze(1)  # [B,1,H,W]
            loss = criterion(pred, target)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.6f}")

        # Run inference after each epoch
        model.eval()
        with torch.no_grad():
            inferred_Ez = do_inference(
                model, sample_eps, sample_mu, sample_src, sample_omega, device
            )

            # Calculate reference solution using FDFD
            reference_Ez = run_fdfd(
                sample_eps,
                sample_mu,
                sample_src,
                dx=1e-3,
                omega=sample_omega.item(),
            )

            # Plot both inference and reference
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Plot inference result
            im1 = ax1.imshow(inferred_Ez.cpu().numpy(), cmap="bwr")
            ax1.set_title("Model Inference")
            plt.colorbar(im1, ax=ax1)

            # Plot reference solution
            im2 = ax2.imshow(reference_Ez.cpu().numpy(), cmap="bwr")
            ax2.set_title("FDFD Reference")
            plt.colorbar(im2, ax=ax2)

            plt.suptitle(f"Epoch {epoch}")
            plt.savefig(f"assets/logs/comparison_epoch_{epoch}.png")
            plt.close()

        model.train()


if __name__ == "__main__":
    # Hyperparameters
    num_samples = 1000
    dimension = (250, 250)
    num_steps = 8
    num_epochs = 20
    batch_size = 16
    device = torch.device("mps")

    # Generate dataset: returns eps, mu, src, omega, Ez
    eps_samples, mu_samples, src_samples, omega_samples, Ez_samples = generate_data(
        num_samples, dimension
    )

    noise_sample, alphas = generate_diffusion_data(Ez_samples, num_steps, device=device)
    plot_noisy_sample(noise_sample[0])

    # Create a DataLoader: (eps, mu, src, Ez, omega)
    dataset = TensorDataset(
        eps_samples, mu_samples, src_samples, Ez_samples, omega_samples
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = UNet2DModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Begin training
    train_model(model, optimizer, train_loader, num_steps, num_epochs, device)

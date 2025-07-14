import torch
from typing import Tuple
from fdfd import make_A
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import trange
from diffusion_model import UNet2DModel

"""
This is inspired by DiffusionPDE. We solve FDFD and use that for denoising diffusion training.

We model FDFD as a function f(z, b) where z is the permittivity array and b is the source array.

f(z, b) |-> sol

We then train a model to predict sol, given z and b.

and z, b given sol. 
"""


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
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
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
    dimension: Tuple[int, int], dtype=torch.float64, device=torch.device("cpu")
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
        omega = torch.rand(1).item() * (30e9 - 18e9) + 18e9

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
        torch.tensor(omega_samples),
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
        t_lin = torch.linspace(0, 1, num_steps, device=device)
        raw = torch.cos(((t_lin + s) / (1 + s)) * (torch.pi / 2)).pow(2)
    elif schedule == "linear":
        betas = torch.linspace(0.0, 0.01, num_steps, device=device)
        raw = torch.cumprod(1 - betas, dim=0)
    else:
        raise ValueError(f"Unknown schedule {schedule!r}")

    alphas = raw

    # 3) diffusion loop with scalar αₜ
    noisy = torch.zeros((B, num_steps, H, W), device=device)
    for t in range(num_steps):
        if t == 0:
            noisy[:, t] = Ez
        else:
            α_t = alphas[t]  # zero‐dim tensor
            print(α_t.shape)
            noise = noise_scale * torch.randn_like(Ez) + noise_mean
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


if __name__ == "__main__":
    eps_samples, mu_samples, src_samples, omega_samples, Ez_samples = generate_data(
        1000, (250, 250)
    )
    print(Ez_samples.shape)
    noisy_samples, alphas_cumprod = generate_diffusion_data(Ez_samples, 8)

    print(noisy_samples.shape)
    print(alphas_cumprod.shape)

import torch
from typing import Tuple, Optional

import torch
from torch import nn

from fdfd import make_A
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import math
from diffusion_model import UNet2DModel
from diffusers import DDIMScheduler


"""
This is inspired by DiffusionPDE. We solve FDFD and use that for denoising diffusion training.

We model FDFD as a function f(z, b) where z is the permittivity array and b is the source array.

f(z, b) |-> sol

We then train a model to predict sol, given z and b.

and z, b given sol. 
"""


device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
    device: torch.device = device,
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
    dimension: Tuple[int, int], dtype=torch.float32, device=device
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


def generate_data(
    num_samples: int,
    dimension: Tuple[int, int],
    device: torch.device = device,
):
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
        torch.stack(eps_samples).to(device),
        torch.stack(mu_samples).to(device),
        torch.stack(src_samples).to(device),
        torch.tensor(omega_samples, dtype=torch.float32).to(device),
        torch.stack(Ez_samples).to(device),
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


def plot_ref_v_inference(ref, inference, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convert from frequency to time domain
    ref_time = torch.fft.ifft2(ref, dim=(1, 2)).real
    inference_time = torch.fft.ifft2(inference, dim=(1, 2)).real

    maximum = torch.max(
        torch.max(torch.abs(ref_time)), torch.max(torch.abs(inference_time))
    )

    print("ref shape:", ref.shape)
    print("inference shape:", inference.shape)

    inference_time = inference_time.squeeze(0)

    # Plot predicted field
    im1 = ax1.imshow(
        inference_time.cpu().numpy(), cmap="seismic", vmin=-maximum, vmax=maximum
    )
    ax1.set_title("Predicted Ez (Time Domain)")

    # Plot true field
    im2 = ax2.imshow(
        ref_time.cpu().numpy(), cmap="seismic", vmin=-maximum, vmax=maximum
    )
    ax2.set_title("True Ez (Time Domain)")

    # Add colorbar
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    plt.savefig(path)
    plt.close()


def inference(model, eps, mu, src, omega, scheduler, num_inference_steps=50):
    """
    Perform inference using the trained diffusion model.

    Args:
        model: Trained UNet model
        eps: Permittivity tensor [B, H, W]
        mu: Permeability tensor [B, H, W]
        src: Source tensor [B, H, W]
        omega: Frequency tensor [B]
        scheduler: Diffusion scheduler
        num_inference_steps: Number of denoising steps

    Returns:
        Predicted Ez field
    """
    # Set eval mode
    model.eval()

    # Start from random noise
    noise = torch.randn_like(eps)

    # Set the scheduler timesteps
    scheduler.set_timesteps(num_inference_steps)

    # Initialize sample to pure noise
    sample = noise

    # Gradually denoise
    for t in scheduler.timesteps:
        # Expand timestep tensor to match batch dimension
        timesteps = t.expand(eps.shape[0])

        # Get model prediction
        with torch.no_grad():
            noise_pred = model(eps, mu, src, sample, timesteps.to(device), omega)

        # Update sample with scheduler
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample


if __name__ == "__main__":
    model = UNet2DModel().to(device)
    scheduler = DDIMScheduler()  # using default values

    eps_samples, mu_samples, src_samples, omega_samples, Ez_samples = generate_data(
        1000, (250, 250)
    )

    Ez_samples = torch.fft.fft2(Ez_samples, dim=(1, 2))

    # Create dataset and dataloader
    dataset = TensorDataset(
        eps_samples, mu_samples, src_samples, omega_samples, Ez_samples
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for eps, mu, src, omega, Ez in tqdm(dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            # Sample noise to add to Ez
            noise = torch.randn_like(Ez)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (Ez.shape[0],), device=Ez.device
            )

            # Add noise to Ez according to the schedule
            noisy_Ez = scheduler.add_noise(Ez, noise, timesteps)

            # Get model prediction
            pred = model(eps, mu, src, noisy_Ez, timesteps, omega)

            # Calculate loss (predict the noise)
            loss = F.mse_loss(pred, noise)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")

        # Pick a random index for visualization
        idx = torch.randint(0, len(eps_samples), (1,)).item()
        inference_result = inference(
            model,
            eps_samples[idx].unsqueeze(0),
            mu_samples[idx].unsqueeze(0),
            src_samples[idx].unsqueeze(0),
            omega_samples[idx].unsqueeze(0),
            scheduler,
        )

        plot_ref_v_inference(
            Ez_samples[idx], inference_result, f"eval/comparison_epoch_{epoch}.png"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"checkpoints/checkpoint_epoch_{epoch}.pt",
            )

import numpy as np
import matplotlib.pyplot as plt
import torch


def sparse_solve(A, b, numerical: bool = True): ...


def _sp_solve_numerical(A, b): ...


def _sp_solve_analytic(A, b): ...


def plot_Ez(Ez, eps, source, path, vmax=20, vmin=-20):
    # Normalize to [-1, 1]
    normed = np.clip(Ez, vmin, vmax)

    # Create background grayscale image from eps
    eps_min = 8.85418e-12  # Vacuum permittivity
    eps_max = np.max(eps)
    if eps_max == eps_min:
        eps_gray = np.full_like(eps, 255, dtype=np.uint8)  # White if uniform
    else:
        eps_normed = (eps - eps_min) / (eps_max - eps_min)
        # Scale to 128-255 (gray to white) - high permittivity is gray
        eps_gray = ((1 - eps_normed) * 127 + 128).astype(np.uint8)

    # Create RGB array with eps as background
    background = np.stack([eps_gray] * 3, axis=-1)

    # Map Ez through colormap with alpha=0.7
    cmap = plt.cm.get_cmap("seismic")  # blue-white-red
    rgba = cmap((normed - vmin) / (vmax - vmin))
    rgba[..., 3] = 0.7  # Set alpha to 0.7

    # Convert to RGB with alpha blending
    rgb_float = rgba[..., :3] * rgba[..., 3:] + (background / 255) * (1 - rgba[..., 3:])
    final = (rgb_float * 255).astype(np.uint8)

    plt.imsave(path, final)


def snr_gamma_weight(
    timesteps: torch.Tensor, scheduler, gamma: float = 5.0
) -> torch.Tensor:
    """
    Compute w(t) = SNR(t)^gamma / (SNR(t)^gamma + 1)
    for each timestep in `timesteps` (shape [B]).

    Works with any scheduler that exposes `alphas_cumprod`.
    """
    # scheduler.alphas_cumprod is [num_train_timesteps]
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    # gather the ᾱ_t for each sample’s t
    alphas_t = alphas_cumprod[timesteps]  # shape [B]
    snr_t = alphas_t / (1.0 - alphas_t)  # ᾱ_t / (1−ᾱ_t)
    w_t = (snr_t**gamma) / (snr_t**gamma + 1.0)
    return w_t

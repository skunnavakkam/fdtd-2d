import numpy as np
import jaxtyping
import matplotlib.pyplot as plt


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
    # Add source points in yellow
    source_points = np.nonzero(source)
    eps_gray[source_points] = 255  # Make source points bright
    yellow = np.array([255, 255, 0], dtype=np.uint8)  # Yellow color
    for x, y in zip(*source_points):
        eps_gray[
            max(0, x - 1) : min(x + 2, eps_gray.shape[0]),
            max(0, y - 1) : min(y + 2, eps_gray.shape[1]),
        ] = 255

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

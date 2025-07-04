from fdfd import run_fdfd
from noise import pnoise2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import plot_Ez


def generate_random_structure(Nx=500, Ny=500, eps0=8.85418e-12, mu0=1.25663e-6):
    """
    Generate random waveguide structure with input/output waveguides and random middle section.

    Args:
        Nx, Ny: Grid dimensions
        eps0: Vacuum permittivity
        mu0: Vacuum permeability

    Returns:
        eps: Relative permittivity array
        mu: Permeability array
        source: Source array
    """
    # Initialize background
    eps = np.ones((Nx, Ny)) * eps0
    mu = np.ones((Nx, Ny)) * mu0

    # Add input waveguide on left
    eps[200:300, :100] = 2 * eps0

    # Add output waveguide on right
    eps[200:300, 400:] = 2 * eps0

    # Add random structure in middle
    # Generate Perlin noise for more natural-looking random structures

    # Generate and threshold the noise
    # Generate Perlin noise
    scale = 75.0  # Increased scale for lower frequency noise
    octaves = 4  # Reduced octaves for less fine detail
    persistence = 0.5  # How much each octave contributes
    lacunarity = 2.0  # How much detail is added at each octave

    structure_region = np.zeros((300, 300))
    # Add random offsets to get different patterns each time
    x_offset = np.random.rand() * 10000000
    y_offset = np.random.rand() * 10000000

    for i in range(300):
        for j in range(300):
            structure_region[i][j] = pnoise2(
                (i + x_offset) / scale,
                (j + y_offset) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )

    # Normalize to [0,1] and threshold
    structure_region = (structure_region - structure_region.min()) / (
        structure_region.max() - structure_region.min()
    )
    structure_region = structure_region > 0.5  # Convert to binary mask
    eps[100:400, 100:400] = np.where(structure_region, 2 * eps0, eps[100:400, 100:400])

    # Create source at input waveguide
    source = np.zeros((Nx, Ny))
    source[225:275, 50] = 1  # Line source at x=50

    return eps, mu, source


def preview_eps_source(eps, source):
    """
    Plot the permittivity and source distributions for visualization.

    Args:
        eps: Relative permittivity array
        source: Source array
    """

    plt.figure(figsize=(12, 6))

    # Plot permittivity
    plt.subplot(121)
    plt.imshow(eps, cmap="viridis")
    plt.colorbar(label="Relative Permittivity")
    plt.title("Permittivity Distribution")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot source
    plt.subplot(122)
    plt.imshow(source, cmap="RdBu")
    plt.colorbar(label="Source Amplitude")
    plt.title("Source Distribution")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()

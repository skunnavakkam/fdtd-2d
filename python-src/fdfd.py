import numpy as np
import scipy.sparse as sparse
import scipy.sparse as sp
from scipy.linalg import solve_banded
from main import material_init
from utils import plot_Ez
import matplotlib.pyplot as plt
import time
import pyamg
from collections import deque


def solve_linear(A, b):
    return sp.linalg.spsolve(A, b)


def make_A(eps, mu, dx, dy, Nx, Ny, omega, pml_thickness=40, sigma_max=2, m=3):
    # define the 1d profiles for the pml first
    sigma_x_1d = np.zeros(Nx)
    sigma_x_1d[0:pml_thickness] = (
        sigma_max * (np.flip(np.arange(pml_thickness), axis=0) / (pml_thickness)) ** m
    )
    sigma_x_1d[Nx - pml_thickness :] = (
        sigma_max * (np.arange(pml_thickness) / (pml_thickness)) ** m
    )

    sigma_y_1d = np.zeros(Ny)
    sigma_y_1d[0:pml_thickness] = (
        sigma_max * (np.flip(np.arange(pml_thickness), axis=0) / (pml_thickness)) ** m
    )
    sigma_y_1d[Ny - pml_thickness :] = (
        sigma_max * (np.arange(pml_thickness) / (pml_thickness)) ** m
    )

    # lift to 2d
    sigma_x_2d_extrapolated = np.tile(sigma_x_1d[None, :], (Ny, 1))
    sigma_y_2d_extrapolated = np.tile(sigma_y_1d[:, None], (1, Nx))

    # form stretching array
    s_x = 1 + 1j * sigma_x_2d_extrapolated / (omega * 8.85418e-12)
    s_y = 1 + 1j * sigma_y_2d_extrapolated / (omega * 8.85418e-12)

    # Construct Dx array
    Dx = sp.diags([-1, 1], [-1, 1], shape=(Nx, Nx)) / (2 * dx)
    Dy = sp.diags([-1, 1], [-1, 1], shape=(Ny, Ny)) / (2 * dy)

    I_N_x = sp.eye(Nx)
    I_N_y = sp.eye(Ny)

    C_x = sp.kron(I_N_y, Dx)
    C_y = sp.kron(Dy, I_N_x)

    C_x = sp.diags(1 / s_x.flatten(), 0, shape=(Nx * Ny, Nx * Ny)) @ C_x
    C_y = sp.diags(1 / s_y.flatten(), 0, shape=(Nx * Ny, Nx * Ny)) @ C_y

    eps_flat = eps.flatten()
    mu_flat = mu.flatten()

    M_eps = sp.diags(eps_flat, 0, shape=(Nx * Ny, Nx * Ny))
    M_mu = sp.diags(1 / mu_flat, 0, shape=(Nx * Ny, Nx * Ny))

    A = C_x @ M_mu @ C_x.T + C_y @ M_mu @ C_y.T - omega**2 * M_eps

    return A


def plot_nonzero(A):
    """Plot the sparsity pattern of the first 5000x5000 elements of matrix A.
    Red indicates nonzero elements, blue indicates zero elements."""

    plt.figure(figsize=(12, 12))
    A_dense = A[
        :5000, :5000
    ].toarray()  # Convert top-left 5000x5000 corner to dense array
    plt.imshow(A_dense != 0, cmap="RdBu")  # Red for nonzero, blue for zero
    plt.colorbar()
    plt.title("Sparsity pattern of matrix A (first 5000x5000 elements)")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.savefig("matrix_pattern_5000.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_fdfd(eps, mu, dx, dy, omega, source, patch_size=100):
    patches = []
    # Split the source into 100x100 patches
    Nx, Ny = source.shape

    # Calculate number of patches in each dimension
    n_patches_x = (Nx + patch_size - 1) // patch_size  # Ceiling division
    n_patches_y = (Ny + patch_size - 1) // patch_size

    # Split into patches
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            # Calculate patch boundaries
            x_start = i * patch_size
            x_end = min((i + 1) * patch_size, Nx)
            y_start = j * patch_size
            y_end = min((j + 1) * patch_size, Ny)

            # Extract patch
            patch = source[x_start:x_end, y_start:y_end]
            patches.append(patch)

    # Find patches with nonzero source
    nonzero_patches = []
    patch_locations = []
    for idx, patch in enumerate(patches):
        if np.any(patch != 0):
            nonzero_patches.append(patch)
            # Calculate patch location from index
            patch_i = idx // n_patches_y  # Row index
            patch_j = idx % n_patches_y  # Column index
            patch_locations.append((patch_i, patch_j))

    # Plot the nonzero patches to visualize their locations
    plt.figure(figsize=(10, 10))
    patch_map = np.zeros((n_patches_x, n_patches_y))
    for i, j in patch_locations:
        patch_map[i, j] = 1
    plt.imshow(patch_map, cmap="binary")
    plt.colorbar()
    plt.title("Nonzero Source Patch Locations")
    plt.xlabel("Patch Column Index")
    plt.ylabel("Patch Row Index")
    plt.savefig("nonzero_patches.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    Nx = 1001
    Ny = 1001
    dx = dy = 1e-3
    omega = 17e9

    source = np.zeros((Nx, Ny))
    source[197:199, 190] = 10

    eps, mu = material_init("example_structure.png", Nx, Ny, 3)

    c_medium = 1 / np.sqrt(eps * mu)
    c_min = np.min(c_medium)
    lambda_min = c_min / omega

    # resolution check
    if dx > lambda_min / 10:
        raise ValueError(
            "dx must be less than lambda_min / 10, current dx: {}, lambda_min / 10: {}".format(
                dx, lambda_min / 10
            )
        )

    if dx < lambda_min / 20:
        raise ValueError("dx too small, you're throwing away compute")

    A = make_A(eps, mu, dx, dy, Nx, Ny, omega)
    b = -1j * omega * source.flatten()

    # Plot the top left corner of the matrix A
    # plt.figure(figsize=(10, 10))
    # A_dense = A[:100, :100].toarray()  # Convert top-left corner to dense array
    # plt.imshow(A_dense != 0, cmap="RdBu")  # Red for nonzero, blue for zero
    # plt.colorbar()
    # plt.title("Sparsity pattern of top-left corner of matrix A")
    # plt.savefig("matrix_pattern.png")
    # plt.close()
    # print(b.shape)
    # graph = sp.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    # permuted_A = A[graph, :][:, graph]
    # print(sp.linalg.spbandwidth(A))
    # print(sp.linalg.spbandwidth(permuted_A))

    Ez_new = run_fdfd(eps, mu, dx, dy, omega, source)

import numpy as np
import scipy.sparse as sp
from main import material_init
from utils import plot_Ez
import matplotlib.pyplot as plt
from fdfd_jax import _make_A as make_A_jax
from fdfd_jax import run_FDFD_jax
from scipy.sparse import csr_matrix
import jax.numpy as jnp
from fdfd_jax import _make_A_scipy_jax as make_A_scipy_jax


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


if __name__ == "__main__":
    Nx = 60
    Ny = 60
    dx = dy = 1e-3
    omega = 17e9

    source = np.zeros((Nx, Ny))
    source = np.array(source)

    eps, mu = material_init("assets/example_structure.png", Nx, Ny, 3)

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

    print("starting FDFD")

    A_jax, A_jax_scipy = make_A_scipy_jax(eps, mu, dx, dy, Nx, Ny, omega)

import numpy as np
import scipy.sparse as sp
from main import material_init
from utils import plot_Ez
import matplotlib.pyplot as plt


def make_A(eps, mu, dx, dy, Nx, Ny, omega, pml_thickness=40, sigma_max=2, m=3):
    print("Starting make_A function...")
    print(f"Grid size: {Nx}x{Ny}, dx={dx}, dy={dy}")
    print(f"PML parameters: thickness={pml_thickness}, sigma_max={sigma_max}, m={m}")

    # define the 1d profiles for the pml first
    print("Calculating PML profiles...")
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
    print("Extending PML profiles to 2D...")
    sigma_x_2d_extrapolated = np.tile(sigma_x_1d[None, :], (Ny, 1))
    sigma_y_2d_extrapolated = np.tile(sigma_y_1d[:, None], (1, Nx))

    # form stretching array
    print("Calculating stretching arrays...")
    s_x = 1 + 1j * sigma_x_2d_extrapolated / (omega * 8.85418e-12)
    s_y = 1 + 1j * sigma_y_2d_extrapolated / (omega * 8.85418e-12)

    # Construct Dx array
    print("Constructing derivative operators...")
    Dx = sp.diags([-1, 1], [-1, 1], shape=(Nx, Nx)) / (2 * dx)
    Dy = sp.diags([-1, 1], [-1, 1], shape=(Ny, Ny)) / (2 * dy)

    I_N_x = sp.eye(Nx)
    I_N_y = sp.eye(Ny)

    print("Building curl operators...")
    C_x = sp.kron(I_N_y, Dx)
    C_y = sp.kron(Dy, I_N_x)

    C_x = sp.diags(1 / s_x.flatten(), 0, shape=(Nx * Ny, Nx * Ny)) @ C_x
    C_y = sp.diags(1 / s_y.flatten(), 0, shape=(Nx * Ny, Nx * Ny)) @ C_y

    print("Creating material matrices...")
    eps_flat = eps.flatten()
    mu_flat = mu.flatten()

    M_eps = sp.diags(eps_flat, 0, shape=(Nx * Ny, Nx * Ny))
    M_mu = sp.diags(1 / mu_flat, 0, shape=(Nx * Ny, Nx * Ny))

    print("Assembling final matrix A...")
    A = C_x @ M_mu @ C_x.T + C_y @ M_mu @ C_y.T - omega**2 * M_eps

    print("Matrix A construction complete.")
    return A


if __name__ == "__main__":
    Nx = 1000
    Ny = 1000
    dx = dy = 1e-3
    omega = 9e9

    source = np.zeros((Nx, Ny))
    source[Nx // 2, Ny // 2] = 10

    eps, mu = material_init("box.png", Nx, Ny, 3)

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
    plt.figure(figsize=(10, 10))
    A_dense = A[:100, :100].toarray()  # Convert top-left corner to dense array
    plt.imshow(A_dense != 0, cmap="RdBu")  # Red for nonzero, blue for zero
    plt.colorbar()
    plt.title("Sparsity pattern of top-left corner of matrix A")
    plt.savefig("matrix_pattern.png")
    plt.close()
    print(b.shape)

    # Ez_new = sp.linalg.spsolve(A, b)

    # Ez = Ez_new.reshape(Nx, Ny)

    # Ez = np.real(Ez)
    # print(np.max(Ez), np.min(Ez))

    # plot_Ez(Ez, eps, source, "Ez.png", np.max(np.abs(Ez)), -np.max(np.abs(Ez)))

import numpy as np
import scipy.sparse as sparse
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy
from scipy.linalg import solve_banded
from main import material_init
from utils import plot_Ez
import matplotlib.pyplot as plt
import time
import pyamg
from collections import deque
from collections import namedtuple


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


def _solve_patch(eps, mu, dx, dy, omega, source, pml_thickness=5):
    """Direct FDFD solve on one patch *including* its local PML."""

    assert eps.shape == mu.shape == source.shape, "shape mismatch"

    A = make_A(
        eps,
        mu,
        dx,
        dy,
        *eps.shape,
        omega,
        pml_thickness=pml_thickness,
    )
    b = -1j * omega * source.ravel()
    Ez = scipy.sparse.linalg.spsolve(A, b).reshape(eps.shape)
    return Ez


Patch = namedtuple("Patch", ["top_left", "bottom_right"])


def run_fdfd(
    eps,
    mu,
    dx,
    dy,
    omega,
    source,
    *,
    patch_size: int = 100,
    padding: int = 10,
    pml_thickness: int = 5,
):
    """
    Solve the 2‑D FDFD problem by tiling the domain into partially
    overlapping patches (each including its own PML halo).

    * Every patch is clamped to the simulation window – no negative
      indices are produced.
    * Patches that would collapse to zero (smaller than twice the halo
      thickness in either dimension) are skipped.
    * When the local solution is written back only the interior
      (everything outside the PML halo) is copied, so the public halo
      width naturally follows ``pml_thickness``.
    """

    Nx, Ny = eps.shape
    assert eps.shape == mu.shape == source.shape, "shape mismatch"

    # ------------------------------------------------------------------
    # 1. Generate a regular grid of patch centres.
    # ------------------------------------------------------------------
    x_centers = range(patch_size // 2, Nx, patch_size)
    y_centers = range(patch_size // 2, Ny, patch_size)

    patches = []  # type: list[Patch]
    for cx in x_centers:
        for cy in y_centers:
            x_min = max(0, cx - patch_size // 2 - padding)
            x_max = min(Nx, cx + patch_size // 2 + padding)
            y_min = max(0, cy - patch_size // 2 - padding)
            y_max = min(Ny, cy + patch_size // 2 + padding)

            # Discard degenerate patches (need room for two halos).
            if (x_max - x_min) <= 2 * pml_thickness or (
                y_max - y_min
            ) <= 2 * pml_thickness:
                continue

            patches.append(Patch((x_min, y_min), (x_max, y_max)))

    # ------------------------------------------------------------------
    # 2. Label patches by topological distance from any that contain the
    #    *non‑zero* part of the source in their *interior* region.
    # ------------------------------------------------------------------
    source_patches = set()
    patches_with_dist = []

    halo = pml_thickness  # shorthand

    for idx, patch in enumerate(patches):
        (x_min, y_min), (x_max, y_max) = patch
        # interior region excludes the padding / halo
        ix_min, ix_max = x_min + halo, x_max - halo
        iy_min, iy_max = y_min + halo, y_max - halo

        # Clamp to valid domain before indexing.
        xs = slice(ix_min, ix_max)
        ys = slice(iy_min, iy_max)

        if np.any(source[xs, ys] != 0):
            source_patches.add(idx)
            patches_with_dist.append((patch, 0))

    # Breadth‑first search to propagate outward.
    visited = set(source_patches)
    frontier = set(source_patches)
    current_d = 0
    while frontier and len(visited) < len(patches):
        current_d += 1
        next_frontier = set()

        for i in frontier:
            (x1_min, y1_min), (x1_max, y1_max) = patches[i]

            for j, p2 in enumerate(patches):
                if j in visited:
                    continue
                (x2_min, y2_min), (x2_max, y2_max) = p2

                # axis‑aligned bounding boxes overlap → adjacency
                if (
                    x1_min <= x2_max
                    and x2_min <= x1_max
                    and y1_min <= y2_max
                    and y2_min <= y1_max
                ):
                    visited.add(j)
                    next_frontier.add(j)
                    patches_with_dist.append((p2, current_d))

        frontier = next_frontier

    # Ensure distance‑sorted order.
    patches_with_dist.sort(key=lambda t: t[1])

    # ------------------------------------------------------------------
    # 3. Patch‑wise solves, writing back only the inner region.
    # ------------------------------------------------------------------
    solution = source.copy()

    inner_src = slice(halo, -halo or None)

    for patch_idx, (patch, _dist) in enumerate(patches_with_dist):
        (x_min, y_min), (x_max, y_max) = patch

        # Plot current patch being solved
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(np.abs(solution), cmap="viridis")
        plt.colorbar(label="|Ez|")
        plt.title(f"Solution before patch {patch_idx}")

        # Highlight current patch
        rect = plt.Rectangle(
            (y_min, x_min),
            y_max - y_min,
            x_max - x_min,
            fill=False,
            color="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)

        patch_eps = eps[x_min:x_max, y_min:y_max]
        patch_mu = mu[x_min:x_max, y_min:y_max]
        patch_source = solution[x_min:x_max, y_min:y_max]

        patch_sol = _solve_patch(
            patch_eps, patch_mu, dx, dy, omega, patch_source, pml_thickness=halo
        )

        solution[
            x_min + halo : x_max - halo,
            y_min + halo : y_max - halo,
        ] = patch_sol[inner_src, inner_src]

        # Plot solution after solving this patch
        plt.subplot(122)
        plt.imshow(np.abs(solution), cmap="viridis")
        plt.colorbar(label="|Ez|")
        plt.title(f"Solution after patch {patch_idx}")

        # Highlight current patch
        rect = plt.Rectangle(
            (y_min, x_min),
            y_max - y_min,
            x_max - x_min,
            fill=False,
            color="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)

        plt.tight_layout()
        plt.show()

    return solution


if __name__ == "__main__":
    Nx = 1000
    Ny = 1000
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
    plot_Ez(
        Ez_new,
        eps,
        source,
        "Ez_tiled.png",
        np.max(np.abs(Ez_new)),
        -np.max(np.abs(Ez_new)),
    )

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


def _extract_dirichlet_bc(sol_patch: np.ndarray, halo: int):
    """Return (top, bottom, left, right) field arrays from the outermost ring."""
    top = sol_patch[halo, halo:-halo].copy()
    bottom = sol_patch[-halo - 1, halo:-halo].copy()
    left = sol_patch[halo:-halo, halo].copy()
    right = sol_patch[halo:-halo, -halo - 1].copy()
    return top, bottom, left, right


def _solve_patch(
    eps,
    mu,
    dx,
    dy,
    omega,
    source,
    *,
    pml_thickness: int = 5,
    dirichlet_bc=None,  # NEW – pass (top, bottom, left, right) or None
):
    """
    Direct FDFD solve on one patch *including* its local PML.

    Parameters
    ----------
    dirichlet_bc : None | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Fixed Ez values on the outermost cell ring (top, bottom, left, right).
        Each array is 1-D and already matches the length of that edge.
        If None (default) no extra Dirichlet rows are imposed.
    """
    # ------------------------------------------------------------------
    # 0. basic checks
    # ------------------------------------------------------------------
    assert eps.shape == mu.shape == source.shape, "shape mismatch"
    Nx, Ny = eps.shape
    halo = pml_thickness

    # ------------------------------------------------------------------
    # 1. Assemble the standard Helmholtz matrix for this patch
    # ------------------------------------------------------------------
    A = make_A(
        eps,
        mu,
        dx,
        dy,
        Nx,
        Ny,
        omega,
        pml_thickness=pml_thickness,
    )
    b = (-1j * omega * source).ravel()  # column vector

    # ------------------------------------------------------------------
    # 2. Impose Dirichlet rows if requested
    # ------------------------------------------------------------------
    if dirichlet_bc is not None:
        top, bottom, left, right = dirichlet_bc
        A = A.tolil()  # easier to edit rows
        idx = lambda i, j: i * Ny + j  # 2-D → 1-D mapping

        # --- top edge (y = halo) ---
        i = halo
        for j, val in zip(range(halo, Ny - halo), top):
            k = idx(i, j)
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = val

        # --- bottom edge (y = Ny-halo-1) ---
        i = Nx - halo - 1
        for j, val in zip(range(halo, Ny - halo), bottom):
            k = idx(i, j)
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = val

        # --- left edge (x = halo) ---
        j = halo
        for i, val in zip(range(halo, Nx - halo), left):
            k = idx(i, j)
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = val

        # --- right edge (x = Ny-halo-1) ---
        j = Ny - halo - 1
        for i, val in zip(range(halo, Nx - halo), right):
            k = idx(i, j)
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = val

        A = A.tocsc()  # back to efficient format

    # ------------------------------------------------------------------
    # 3. Direct solve
    # ------------------------------------------------------------------
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
    padding: int = 30,
    pml_thickness: int = 10,  # ↑ a bit thicker for better BCs
):
    """
    One-pass tiled FDFD solve with correct RHS and Dirichlet halo exchange.
    Everything else (patch layout, ordering) matches your original draft.
    """
    Nx, Ny = eps.shape
    assert eps.shape == mu.shape == source.shape, "shape mismatch"

    halo = pml_thickness
    inner = slice(halo, -halo or None)  # interior slice within a patch
    orig_source = source.copy()  # fixed RHS
    solution = np.zeros_like(source)  # start from zero field

    # ------------------------------------------------------------------
    # 1. Generate patches exactly as before
    # ------------------------------------------------------------------
    x_centers = range(patch_size // 2, Nx, patch_size)
    y_centers = range(patch_size // 2, Ny, patch_size)

    patches = []
    for cx in x_centers:
        for cy in y_centers:
            x_min = max(0, cx - patch_size // 2 - padding)
            x_max = min(Nx, cx + patch_size // 2 + padding)
            y_min = max(0, cy - patch_size // 2 - padding)
            y_max = min(Ny, cy + patch_size // 2 + padding)

            if (x_max - x_min) <= 2 * halo or (y_max - y_min) <= 2 * halo:
                continue
            patches.append(((x_min, y_min), (x_max, y_max)))

    # ------------------------------------------------------------------
    # 2. Distance-sort outward from any patch whose *interior* touches RHS
    # ------------------------------------------------------------------
    source_bool = orig_source != 0
    patches_with_dist, frontier, visited = [], set(), set()

    for idx, patch in enumerate(patches):
        (a0, b0), (a1, b1) = patch
        if np.any(source_bool[a0 + halo : a1 - halo, b0 + halo : b1 - halo]):
            patches_with_dist.append((patch, 0))
            frontier.add(idx)
            visited.add(idx)

    d = 0
    while frontier and len(visited) < len(patches):
        d += 1
        nxt = set()
        for i in frontier:
            (x0, y0), (x1, y1) = patches[i]
            for j, p2 in enumerate(patches):
                if j in visited:  # already labelled
                    continue
                (u0, v0), (u1, v1) = p2
                if x0 <= u1 and u0 <= x1 and y0 <= v1 and v0 <= y1:
                    visited.add(j)
                    nxt.add(j)
                    patches_with_dist.append((p2, d))
        frontier = nxt

    patches_with_dist.sort(key=lambda t: t[1])  # ensure outward order

    # ------------------------------------------------------------------
    # 3. Single outward sweep
    # ------------------------------------------------------------------
    for patch, _dist in patches_with_dist:
        (x0, y0), (x1, y1) = patch

        # materials & RHS
        p_eps = eps[x0:x1, y0:y1]
        p_mu = mu[x0:x1, y0:y1]
        p_source = orig_source[x0:x1, y0:y1]

        # current global field over the patch → Dirichlet ring
        sol_patch = solution[x0:x1, y0:y1]
        bc = _extract_dirichlet_bc(sol_patch, halo)

        # local solve (needs new kw-arg)
        p_sol = _solve_patch(
            p_eps,
            p_mu,
            dx,
            dy,
            omega,
            p_source,
            pml_thickness=halo,
            dirichlet_bc=bc,  # ← NEW
        )

        # overwrite interior (no relaxation)
        solution[x0 + halo : x1 - halo, y0 + halo : y1 - halo] = p_sol[inner, inner]

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

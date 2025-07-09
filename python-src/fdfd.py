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
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax import lax
from jax.scipy.sparse.linalg import cg
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import bcoo_dot_general


def solve_linear(A, b):
    return jax.experimental.sparse.linalg.spsolve(A, b)


def _diags(diagonals, offsets, shape) -> jsparse.BCOO:
    """
    Build a sparse BCOO matrix of given shape whose k-th diagonal
    (offset by offsets[k]) is filled from diagonals[k],
    without ever allocating a full dense matrix.
    """
    n_rows, n_cols = shape
    coord_list = []
    data_list = []

    for diag, k in zip(diagonals, offsets):
        # Compute where the diagonal actually lives in the matrix
        start_row = max(0, -k)
        start_col = max(0, k)
        # Maximum length before hitting any boundary
        length = min(diag.shape[0], n_rows - start_row, n_cols - start_col)
        if length <= 0:
            continue

        idxs = jnp.arange(length)
        rows = start_row + idxs
        cols = start_col + idxs

        coord_list.append(jnp.stack([rows, cols], axis=1))
        data_list.append(diag[:length])

    # If we got no valid diagonals, return an all-zero sparse
    if not coord_list:
        empty_coords = jnp.zeros((0, 2), dtype=jnp.int32)
        empty_data = jnp.zeros((0,), dtype=diagonals[0].dtype)
        return jsparse.BCOO((empty_data, empty_coords), shape=shape)

    # Concatenate all the per-diagonal coords & data
    coords = jnp.concatenate(coord_list, axis=0)
    data = jnp.concatenate(data_list, axis=0)

    return jsparse.BCOO((data, coords), shape=shape)


def _kron(A: jsparse.BCOO, B: jsparse.BCOO) -> jsparse.BCOO:
    # Build coordinates for the Kronecker product (A ⊗ B)
    a_r, a_c = A.indices.T
    b_r, b_c = B.indices.T
    rows = (a_r[:, None] * B.shape[0] + b_r[None, :]).ravel()
    cols = (a_c[:, None] * B.shape[1] + b_c[None, :]).ravel()
    data = (A.data[:, None] * B.data[None, :]).ravel()
    idx = jnp.column_stack([rows, cols])
    shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    return jsparse.BCOO((data, idx), shape=shape)


def _matmul(A: jsparse.BCOO, B: jsparse.BCOO, block_size: int = 1024) -> jsparse.BCOO:
    """
    Memory-efficient sparse matrix multiplication for BCOO matrices by processing A in row blocks.

    Args:
      A: A BCOO sparse matrix of shape (M, K).
      B: A BCOO sparse matrix of shape (K, N).
      block_size: Number of rows of A to process at a time to bound memory usage.

    Returns:
      A new BCOO sparse matrix of shape (M, N) representing A @ B without densifying the full result.
    """
    M, K = A.shape
    _, N = B.shape
    dimension_numbers = (((1,), (0,)), ((), ()))

    idx_list = []
    data_list = []

    # Process A in row-wise blocks to cap intermediate memory usage
    for row_start in range(0, M, block_size):
        row_end = min(M, row_start + block_size)

        # Mask indices/data belonging to this row block
        mask = (A.indices[:, 0] >= row_start) & (A.indices[:, 0] < row_end)
        if not jnp.any(mask):
            continue

        # Extract block and rebase row indices to [0, block_size)
        block_indices = A.indices[mask]
        block_indices = block_indices.at[:, 0].add(-row_start)
        block_data = A.data[mask]
        A_block = jsparse.BCOO(
            (block_indices, block_data), shape=(row_end - row_start, K)
        )

        # Multiply block x B
        C_block = bcoo_dot_general(A_block, B, dimension_numbers=dimension_numbers)

        # Rebase block row indices back to global coordinates
        global_idx = C_block.indices.at[:, 0].add(row_start)
        idx_list.append(global_idx)
        data_list.append(C_block.data)

    if not idx_list:
        # No nonzeros
        return jsparse.BCOO.fromdense(jnp.zeros((M, N)))

    # Concatenate blocks
    all_indices = jnp.concatenate(idx_list, axis=0)
    all_data = jnp.concatenate(data_list, axis=0)

    # Note: Rows are partitioned, so no duplicate (i,j) across blocks. No need to sum.
    return jsparse.BCOO((all_indices, all_data), shape=(M, N))


def make_A(
    eps: jnp.ndarray,
    mu: jnp.ndarray,
    dx: float,
    dy: float,
    Nx: int,
    Ny: int,
    omega: float,
    pml_thickness: int = 40,
    sigma_max: float = 2.0,
    m: int = 3,
):
    """
    One-to-one JAX re-implementation of make_A, with memory-efficient
    sparse row-scaling instead of building full diagonal stretching matrices.
    All inputs have the same meaning as in the original SciPy version.
    Returns a sparse JAX matrix (BCOO) whose dense form equals the original `A`.
    """
    print("Starting make_A...")

    # -------------------------
    # 1) Build 1-D PML profiles
    # -------------------------
    print("Building 1D PML profiles...")
    idx = jnp.arange(pml_thickness)
    power = (idx / pml_thickness) ** m

    sigma_x_1d = jnp.zeros(Nx)
    sigma_x_1d = sigma_x_1d.at[:pml_thickness].set(sigma_max * power[::-1])
    sigma_x_1d = sigma_x_1d.at[Nx - pml_thickness :].set(sigma_max * power)

    sigma_y_1d = jnp.zeros(Ny)
    sigma_y_1d = sigma_y_1d.at[:pml_thickness].set(sigma_max * power[::-1])
    sigma_y_1d = sigma_y_1d.at[Ny - pml_thickness :].set(sigma_max * power)

    # ------------------------
    # 2) Lift to 2-D and build
    #    complex stretching
    # ------------------------
    print("Building 2D complex stretching...")
    sigma_x_2d = jnp.tile(sigma_x_1d[None, :], (Ny, 1))
    sigma_y_2d = jnp.tile(sigma_y_1d[:, None], (1, Nx))

    eps0 = 8.85418e-12
    s_x = 1.0 + 1j * sigma_x_2d / (omega * eps0)
    s_y = 1.0 + 1j * sigma_y_2d / (omega * eps0)

    # ---------------------------------
    # 3) Finite-difference curl blocks
    # ---------------------------------
    print("Building finite-difference curl blocks...")
    Dx = _diags(
        [-jnp.ones(Nx), jnp.ones(Nx)], offsets=jnp.array([-1, 1]), shape=(Nx, Nx)
    ) / (2.0 * dx)
    Dy = _diags(
        [-jnp.ones(Ny), jnp.ones(Ny)], offsets=jnp.array([-1, 1]), shape=(Ny, Ny)
    ) / (2.0 * dy)

    I_Nx = _diags([jnp.ones(Nx)], offsets=jnp.array([0]), shape=(Nx, Nx))
    I_Ny = _diags([jnp.ones(Ny)], offsets=jnp.array([0]), shape=(Ny, Ny))

    print("Computing Kronecker products...")
    C_x = _kron(I_Ny, Dx)
    C_y = _kron(Dy, I_Nx)

    # -----------------------------------
    # 4) Apply complex stretching scales
    # -----------------------------------
    print("Applying complex stretching scales (sparse row-scaling)...")
    N = Nx * Ny

    # Flatten and invert stretch factors
    sx_flat = s_x.ravel()
    sy_flat = s_y.ravel()
    sx_inv = 1.0 / sx_flat
    sy_inv = 1.0 / sy_flat

    # Row-scale C_x
    rows_cx = C_x.indices[:, 0]
    data_cx = C_x.data * sx_inv[rows_cx]
    C_x = BCOO((data_cx, C_x.indices), shape=(N, N))

    # Row-scale C_y
    rows_cy = C_y.indices[:, 0]
    data_cy = C_y.data * sy_inv[rows_cy]
    C_y = BCOO((data_cy, C_y.indices), shape=(N, N))

    # ----------------------
    # 5) Material matrices
    # ----------------------
    print("Building material matrices...")
    M_eps = _diags([eps.ravel()], offsets=jnp.array([0]), shape=(N, N))
    M_mu_inv = _diags([1.0 / mu.ravel()], offsets=jnp.array([0]), shape=(N, N))

    # -------------------------------
    # 6) Assemble the Helmholtz block
    # -------------------------------
    print("Assembling final Helmholtz block...")
    A = (
        _matmul(C_x, _matmul(M_mu_inv, C_x.T))
        + _matmul(C_y, _matmul(M_mu_inv, C_y.T))
        - (omega**2) * M_eps
    )
    print("make_A complete!")
    return A


def apply_A(v_flat):
    v = v_flat.reshape(Ny, Nx)
    dμ = 1.0 / mu
    term = (jnp.diff(v, 1, 1, append=0) / dx - jnp.diff(v, 1, 0, append=0) / dy) * dμ
    out = (
        jnp.diff(term, 1, 1, prepend=0) / dx
        + jnp.diff(term, 1, 0, prepend=0) / dy
        - (omega**2) * eps * v
    )
    return out.ravel()


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
    Nx = 1000
    Ny = 1000
    dx = dy = 1e-3
    omega = 17e9

    source = np.zeros((Nx, Ny))
    source[197:199, 190] = 10
    source = jnp.array(source)

    eps, mu = material_init("assets/example_structure.png", Nx, Ny, 3)

    c_medium = 1 / jnp.sqrt(eps * mu)
    c_min = jnp.min(c_medium)
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

    print("starting make A")

    A = make_A(eps, mu, dx, dy, Nx, Ny, omega)
    b = -1j * omega * source.flatten()

    print("finished making A")

    b = jnp.array(b)
    Ez_new = solve_linear(A, b)
    Ez_new = Ez_new.reshape(Ny, Nx)
    plot_Ez(
        Ez_new,
        eps,
        source,
        "Ez_tiled.png",
        jnp.max(jnp.abs(Ez_new)),
        -jnp.max(jnp.abs(Ez_new)),
    )

import jax
from jax.experimental.sparse import CSR
from scipy.sparse import csr_matrix
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from typing import Union
import numpy as np
from jax.experimental.sparse.linalg import spsolve
from scipy.sparse.linalg import spsolve as spsolve_scipy
from typing import Tuple
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)


def _spsolve(A: CSR, b: jnp.ndarray) -> jnp.ndarray:
    A_scipy = csr_matrix(
        (
            np.array(A.data),
            np.array(A.indices),
            np.array(A.indptr),
        ),
        shape=A.shape,
    )
    b_scipy = np.array(b.flatten())
    ret = spsolve_scipy(A_scipy, b_scipy)
    return jnp.array(ret)


def _sp_add(A: CSR, B: CSR) -> CSR:
    """
    Sparse matrix addition.
    """
    A_scipy = csr_matrix(
        (np.array(A.data), np.array(A.indices), np.array(A.indptr)), shape=A.shape
    )
    B_scipy = csr_matrix(
        (np.array(B.data), np.array(B.indices), np.array(B.indptr)), shape=B.shape
    )
    C = A_scipy + B_scipy
    return CSR((C.data, C.indices, C.indptr), shape=C.shape)


def _sp_matmul(A: CSR, B: CSR) -> CSR:
    """
    Sparse matrix multiplication.
    """
    A_scipy = csr_matrix(
        (np.array(A.data), np.array(A.indices), np.array(A.indptr)), shape=A.shape
    )
    B_scipy = csr_matrix(
        (np.array(B.data), np.array(B.indices), np.array(B.indptr)), shape=B.shape
    )
    C = A_scipy @ B_scipy
    # wrap back into JAX CSR
    return CSR((C.data, C.indices, C.indptr), shape=C.shape)


def _sp_matmul_jvp(
    primals: Tuple[CSR, CSR], tangents: Tuple[CSR, CSR]
) -> Tuple[CSR, CSR]:
    A, B = primals
    dA, dB = tangents

    # primal output
    y = _sp_matmul(A, B)
    # tangent: d(AB) = (dA)B + A(dB)
    dy = _sp_add(_sp_matmul(dA, B), _sp_matmul(A, dB))
    return y, dy


def _sp_matmul_transpose(ct: CSR, A: CSR, B: CSR) -> Tuple[CSR, CSR]:
    # cotangent wrt A: ct @ Bᵀ,  wrt B: Aᵀ @ ct
    dA = _sp_matmul(ct, B.T)
    dB = _sp_matmul(A.T, ct)
    return dA, dB


# === FIXED VJP REGISTRATION ===
# decorate and capture the returned function
_sp_matmul = jax.custom_vjp(_sp_matmul, nondiff_argnums=())


# fwd: return primal plus residuals
def _sp_matmul_fwd(A: CSR, B: CSR):
    y = _sp_matmul(A, B)
    return y, (A, B)


# bwd: given residuals and cotangent, call your transpose rule
def _sp_matmul_bwd(residuals, ct: CSR):
    A, B = residuals
    return _sp_matmul_transpose(ct, A, B)


# tie it all together
_sp_matmul.defvjp(_sp_matmul_fwd, _sp_matmul_bwd)


def _diags(diagonals, offsets, shape=None):
    """
    Build a sparse CSR matrix matching scipy.sparse.diags behavior.

    Parameters
    ----------
    diagonals : scalar, array-like, or list thereof
        Values to fill on diagonals. Scalars are broadcast; 1-D arrays must match diagonal length.
    offsets : int or list of ints
        Diagonal offsets. Positive for super-diagonals, negative for sub-diagonals.
    shape : tuple of int (n_rows, n_cols), optional
        Desired shape of the matrix. If None, shape is inferred to fit all diagonals.
    """
    # Normalize to lists
    diag_list = diagonals if isinstance(diagonals, (list, tuple)) else [diagonals]
    off_list = (
        offsets if isinstance(offsets, (list, tuple)) else [offsets] * len(diag_list)
    )
    if len(off_list) != len(diag_list):
        raise ValueError("`diagonals` and `offsets` must have the same length")

    # Infer shape if not provided
    if shape is None:
        lengths_offsets = []
        for diag, k in zip(diag_list, off_list):
            arr = jnp.asarray(diag)
            if arr.ndim == 0:
                length = 1
            elif arr.ndim == 1:
                length = arr.shape[0]
            else:
                raise ValueError("`diagonals` elements must be scalars or 1-D arrays")
            lengths_offsets.append((length, k))
        n_rows = max(length + max(0, -k) for length, k in lengths_offsets)
        n_cols = max(length + max(0, k) for length, k in lengths_offsets)
        shape = (n_rows, n_cols)

    n_rows, n_cols = shape
    row_list, col_list, data_list = [], [], []

    # Build triplets
    for diag, k in zip(diag_list, off_list):
        arr = jnp.asarray(diag)
        start_row = max(0, -k)
        start_col = max(0, k)
        max_len = min(n_rows - start_row, n_cols - start_col)
        if max_len <= 0:
            continue
        if arr.ndim == 0:
            data = jnp.full((max_len,), arr)
        elif arr.ndim == 1:
            if arr.shape[0] != max_len:
                raise ValueError(
                    f"Diagonal at offset {k} has length {arr.shape[0]}; expected {max_len}"
                )
            data = arr
        else:
            raise ValueError("`diagonals` elements must be scalars or 1-D arrays")
        idx = jnp.arange(max_len)
        row_list.append(start_row + idx)
        col_list.append(start_col + idx)
        data_list.append(data)

    # Handle empty result
    if not data_list:
        first_dtype = jnp.asarray(diag_list[0]).dtype
        empty_data = jnp.zeros((0,), dtype=first_dtype)
        empty_indices = jnp.zeros((0,), dtype=jnp.int32)
        empty_indptr = jnp.zeros((n_rows + 1,), dtype=jnp.int32)
        return jsparse.CSR((empty_data, empty_indices, empty_indptr), shape=shape)

    # Concatenate and sort
    rows = jnp.concatenate(row_list)
    cols = jnp.concatenate(col_list)
    data = jnp.concatenate(data_list)
    order = jnp.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    # Build CSR index pointer
    counts = jnp.bincount(rows, length=n_rows)
    indptr = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)]
    )

    return jsparse.CSR((data, cols, indptr), shape=shape)


def _kron(A: jsparse.CSR, B: jsparse.CSR) -> jsparse.CSR:
    """
    Compute the Kronecker product (A ⊗ B) as a sparse CSR matrix,
    directly in CSR format, without intermediate dense tensors.
    """
    # Extract COO-like rows/cols from CSR A
    a_indptr = A.indptr
    a_indices = A.indices
    a_data = A.data
    # row id for each nonzero in A
    a_row_ids = jnp.repeat(jnp.arange(A.shape[0]), a_indptr[1:] - a_indptr[:-1])

    # Same for B
    b_indptr = B.indptr
    b_indices = B.indices
    b_data = B.data
    b_row_ids = jnp.repeat(jnp.arange(B.shape[0]), b_indptr[1:] - b_indptr[:-1])

    # Outer combination
    # Compute all pairwise row, col, data
    kron_rows = (a_row_ids[:, None] * B.shape[0] + b_row_ids[None, :]).ravel()
    kron_cols = (a_indices[:, None] * B.shape[1] + b_indices[None, :]).ravel()
    kron_data = (a_data[:, None] * b_data[None, :]).ravel()

    # Sort by row then col
    sort_idx = jnp.lexsort((kron_cols, kron_rows))
    kron_rows = kron_rows[sort_idx]
    kron_cols = kron_cols[sort_idx]
    kron_data = kron_data[sort_idx]

    # Build indptr for result
    n_rows = A.shape[0] * B.shape[0]
    counts = jnp.bincount(kron_rows, length=n_rows)
    indptr = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)]
    )

    return jsparse.CSR(
        (kron_data, kron_cols, indptr), shape=(n_rows, A.shape[1] * B.shape[1])
    )


def _make_A(
    eps, mu, dx, dy, Nx, Ny, omega, pml_thickness=40, sigma_max=2, m=3
) -> jsparse.CSR:
    # define the 1d profiles for the pml first
    sigma_x_1d = jnp.zeros(Nx)
    sigma_x_1d = sigma_x_1d.at[0:pml_thickness].set(
        sigma_max * (jnp.flip(jnp.arange(pml_thickness), axis=0) / (pml_thickness)) ** m
    )
    sigma_x_1d = sigma_x_1d.at[Nx - pml_thickness :].set(
        sigma_max * (jnp.arange(pml_thickness) / (pml_thickness)) ** m
    )

    sigma_y_1d = jnp.zeros(Ny)
    sigma_y_1d = sigma_y_1d.at[0:pml_thickness].set(
        sigma_max * (jnp.flip(jnp.arange(pml_thickness), axis=0) / (pml_thickness)) ** m
    )
    sigma_y_1d = sigma_y_1d.at[Ny - pml_thickness :].set(
        sigma_max * (jnp.arange(pml_thickness) / (pml_thickness)) ** m
    )

    # lift to 2d
    sigma_x_2d_extrapolated = jnp.tile(sigma_x_1d[None, :], (Ny, 1))
    sigma_y_2d_extrapolated = jnp.tile(sigma_y_1d[:, None], (1, Nx))

    # form stretching array
    s_x = 1 + 1j * sigma_x_2d_extrapolated / (omega * 8.85418e-12)
    s_y = 1 + 1j * sigma_y_2d_extrapolated / (omega * 8.85418e-12)

    # Construct Dx array
    Dx = _diags([-1, 1], [-1, 1], shape=(Nx, Nx))
    Dx.data = Dx.data * 1 / (2 * dx)
    Dy = _diags([-1, 1], [-1, 1], shape=(Ny, Ny))
    Dy.data = Dy.data * 1 / (2 * dy)

    I_N_x = jsparse.eye(Nx, sparse_format="csr")
    I_N_y = jsparse.eye(Ny, sparse_format="csr")

    C_x = _kron(I_N_y, Dx)
    C_y = _kron(Dy, I_N_x)

    C_x = _sp_matmul(
        _diags(1 / s_x.flatten(), 0, shape=(Nx * Ny, Nx * Ny)),
        C_x,
    )
    C_y = _sp_matmul(
        _diags(1 / s_y.flatten(), 0, shape=(Nx * Ny, Nx * Ny)),
        C_y,
    )

    eps_flat = eps.flatten()
    mu_flat = mu.flatten()

    M_eps = _diags(eps_flat, 0, shape=(Nx * Ny, Nx * Ny))
    M_mu = _diags(1 / mu_flat, 0, shape=(Nx * Ny, Nx * Ny))

    M_eps.data = M_eps.data * (-1 * omega**2)

    A = _sp_add(
        _sp_add(
            _sp_matmul(C_x, _sp_matmul(M_mu, C_x.T)),
            _sp_matmul(C_y, _sp_matmul(M_mu, C_y.T)),
        ),
        M_eps,
    )

    return A


def _make_A_scipy_jax(
    eps, mu, dx, dy, Nx, Ny, omega, pml_thickness=40, sigma_max=2, m=3
):
    """Implements make_A using both scipy and jax, printing intermediate results for comparison"""

    # === Step 1: PML Profile Setup ===
    print("\nStep 1: PML Profile Setup")

    # Scipy version
    sigma_x_1d_scipy = np.zeros(Nx)
    sigma_x_1d_scipy[0:pml_thickness] = (
        sigma_max * (np.flip(np.arange(pml_thickness), axis=0) / pml_thickness) ** m
    )
    sigma_x_1d_scipy[Nx - pml_thickness :] = (
        sigma_max * (np.arange(pml_thickness) / pml_thickness) ** m
    )

    sigma_y_1d_scipy = np.zeros(Ny)
    sigma_y_1d_scipy[0:pml_thickness] = (
        sigma_max * (np.flip(np.arange(pml_thickness), axis=0) / pml_thickness) ** m
    )
    sigma_y_1d_scipy[Ny - pml_thickness :] = (
        sigma_max * (np.arange(pml_thickness) / pml_thickness) ** m
    )

    # Jax version
    sigma_x_1d_jax = jnp.zeros(Nx)
    sigma_x_1d_jax = sigma_x_1d_jax.at[0:pml_thickness].set(
        sigma_max * (jnp.flip(jnp.arange(pml_thickness), axis=0) / pml_thickness) ** m
    )
    sigma_x_1d_jax = sigma_x_1d_jax.at[Nx - pml_thickness :].set(
        sigma_max * (jnp.arange(pml_thickness) / pml_thickness) ** m
    )

    sigma_y_1d_jax = jnp.zeros(Ny)
    sigma_y_1d_jax = sigma_y_1d_jax.at[0:pml_thickness].set(
        sigma_max * (jnp.flip(jnp.arange(pml_thickness), axis=0) / pml_thickness) ** m
    )
    sigma_y_1d_jax = sigma_y_1d_jax.at[Ny - pml_thickness :].set(
        sigma_max * (jnp.arange(pml_thickness) / pml_thickness) ** m
    )

    print(
        "Max difference in sigma_x_1d:",
        np.max(np.abs(sigma_x_1d_scipy - np.array(sigma_x_1d_jax))),
    )
    print(
        "Max difference in sigma_y_1d:",
        np.max(np.abs(sigma_y_1d_scipy - np.array(sigma_y_1d_jax))),
    )

    # === Step 2: 2D Extrapolation ===
    print("\nStep 2: 2D Extrapolation")

    # Scipy version
    sigma_x_2d_scipy = np.tile(sigma_x_1d_scipy[None, :], (Ny, 1))
    sigma_y_2d_scipy = np.tile(sigma_y_1d_scipy[:, None], (1, Nx))

    # Jax version
    sigma_x_2d_jax = jnp.tile(sigma_x_1d_jax[None, :], (Ny, 1))
    sigma_y_2d_jax = jnp.tile(sigma_y_1d_jax[:, None], (1, Nx))

    print(
        "Max difference in sigma_x_2d:",
        np.max(np.abs(sigma_x_2d_scipy - np.array(sigma_x_2d_jax))),
    )
    print(
        "Max difference in sigma_y_2d:",
        np.max(np.abs(sigma_y_2d_scipy - np.array(sigma_y_2d_jax))),
    )

    # === Step 3: Stretching Arrays ===
    print("\nStep 3: Stretching Arrays")

    # Scipy version
    s_x_scipy = 1 + 1j * sigma_x_2d_scipy / (omega * 8.85418e-12)
    s_y_scipy = 1 + 1j * sigma_y_2d_scipy / (omega * 8.85418e-12)

    # Jax version
    s_x_jax = 1 + 1j * sigma_x_2d_jax / (omega * 8.85418e-12)
    s_y_jax = 1 + 1j * sigma_y_2d_jax / (omega * 8.85418e-12)

    print("Max difference in s_x:", np.max(np.abs(s_x_scipy - np.array(s_x_jax))))
    print("Max difference in s_y:", np.max(np.abs(s_y_scipy - np.array(s_y_jax))))

    # === Step 4: Derivative Operators ===
    print("\nStep 4: Derivative Operators")

    # Scipy version
    Dx_scipy = sp.diags([-1, 1], [-1, 1], shape=(Nx, Nx), format="csr") / (2 * dx)
    Dy_scipy = sp.diags([-1, 1], [-1, 1], shape=(Ny, Ny), format="csr") / (2 * dy)

    # Jax version
    Dx_jax = _diags([-1, 1], [-1, 1], shape=(Nx, Nx))
    Dx_jax.data = Dx_jax.data * 1 / (2 * dx)
    Dy_jax = _diags([-1, 1], [-1, 1], shape=(Ny, Ny))
    Dy_jax.data = Dy_jax.data * 1 / (2 * dy)

    print("Max difference in Dx:", np.max(np.abs(Dx_jax.data - Dx_scipy.data)))
    print("Max difference in Dy:", np.max(np.abs(Dy_jax.data - Dy_scipy.data)))

    # === Step 5: Final Matrix Assembly ===
    print("\nStep 5: Final Matrix Assembly")

    # Scipy version
    # === Step 5a: Identity Matrices ===
    I_N_x_scipy = sp.eye(Nx)
    I_N_y_scipy = sp.eye(Ny)
    I_N_x_jax = _diags([1], [0], shape=(Nx, Nx))
    I_N_y_jax = _diags([1], [0], shape=(Ny, Ny))

    # === Step 5b: Kronecker Products ===
    C_x_scipy = sp.kron(I_N_y_scipy, Dx_scipy)
    C_y_scipy = sp.kron(Dy_scipy, I_N_x_scipy)
    C_x_jax = _kron(I_N_y_jax, Dx_jax)
    C_y_jax = _kron(Dy_jax, I_N_x_jax)
    print("Max difference in C_x:", np.max(np.abs(C_x_scipy.data - C_x_jax.data)))
    print("Max difference in C_y:", np.max(np.abs(C_y_scipy.data - C_y_jax.data)))

    # === Step 5c: Stretching Matrices ===
    C_x_scipy = (
        sp.diags(1 / s_x_scipy.flatten(), 0, shape=(Nx * Ny, Nx * Ny)) @ C_x_scipy
    )
    C_y_scipy = (
        sp.diags(1 / s_y_scipy.flatten(), 0, shape=(Nx * Ny, Nx * Ny)) @ C_y_scipy
    )
    C_x_jax = _sp_matmul(
        _diags(1 / s_x_jax.flatten(), 0, shape=(Nx * Ny, Nx * Ny)), C_x_jax
    )
    C_y_jax = _sp_matmul(
        _diags(1 / s_y_jax.flatten(), 0, shape=(Nx * Ny, Nx * Ny)), C_y_jax
    )
    print(
        "Max difference in stretched C_x:",
        np.max(np.abs(C_x_scipy.data - C_x_jax.data)),
    )
    print(
        "Max difference in stretched C_y:",
        np.max(np.abs(C_y_scipy.data - C_y_jax.data)),
    )

    # === Step 5d: Material Matrices ===
    eps_flat_scipy = eps.flatten()
    mu_flat_scipy = mu.flatten()
    eps_flat_jax = eps.flatten()
    mu_flat_jax = mu.flatten()
    M_eps_scipy = sp.diags(eps_flat_scipy, 0, shape=(Nx * Ny, Nx * Ny), format="csr")
    M_mu_scipy = sp.diags(1 / mu_flat_scipy, 0, shape=(Nx * Ny, Nx * Ny), format="csr")
    M_eps_jax = _diags(eps_flat_jax, 0, shape=(Nx * Ny, Nx * Ny))
    M_mu_jax = _diags(1 / mu_flat_jax, 0, shape=(Nx * Ny, Nx * Ny))
    print("Max difference in M_eps:", np.max(np.abs(M_eps_scipy.data - M_eps_jax.data)))
    print("Max difference in M_mu:", np.max(np.abs(M_mu_scipy.data - M_mu_jax.data)))

    # Check indptr alignment for matrices involved in operations
    print("\nChecking indptr alignment:")
    print("C_x indptr length:", len(C_x_jax.indptr), "shape:", C_x_jax.shape)
    print("M_mu indptr length:", len(M_mu_jax.indptr), "shape:", M_mu_jax.shape)
    print("C_y indptr length:", len(C_y_jax.indptr), "shape:", C_y_jax.shape)

    # Verify JAX and SciPy indptrs match
    assert (C_x_jax.indptr == C_x_scipy.indptr).all(), (
        "C_x indptr mismatch between JAX and SciPy"
    )
    assert (M_mu_jax.indptr == M_mu_scipy.indptr).all(), (
        "M_mu indptr mismatch between JAX and SciPy"
    )
    assert (C_y_jax.indptr == C_y_scipy.indptr).all(), (
        "C_y indptr mismatch between JAX and SciPy"
    )

    # Check transpose operations
    print("\nChecking transpose operations:")
    print(
        "Max difference in C_x transpose:",
        np.max(np.abs(C_x_scipy.T.data - C_x_jax.T.data)),
    )
    print(
        "Max difference in C_y transpose:",
        np.max(np.abs(C_y_scipy.T.data - C_y_jax.T.data)),
    )

    # Verify transpose indptrs match
    assert (C_x_jax.T.indptr == C_x_scipy.T.indptr).all(), (
        "C_x transpose indptr mismatch between JAX and SciPy"
    )
    assert (C_y_jax.T.indptr == C_y_scipy.T.indptr).all(), (
        "C_y transpose indptr mismatch between JAX and SciPy"
    )

    # Convert to dense and check full matrix equality
    print("\nChecking dense matrix equality:")
    C_x_dense_scipy = C_x_scipy.toarray()
    C_x_dense_jax = C_x_jax.todense()
    print(
        "Max difference in C_x dense:", np.max(np.abs(C_x_dense_scipy - C_x_dense_jax))
    )
    assert np.allclose(C_x_dense_scipy, C_x_dense_jax, rtol=1e-10), (
        "C_x matrices differ"
    )

    M_mu_dense_scipy = M_mu_scipy.toarray()
    M_mu_dense_jax = M_mu_jax.todense()
    print(
        "Max difference in M_mu dense:",
        np.max(np.abs(M_mu_dense_scipy - M_mu_dense_jax)),
    )
    assert np.allclose(M_mu_dense_scipy, M_mu_dense_jax, rtol=1e-10), (
        "M_mu matrices differ"
    )

    C_y_dense_scipy = C_y_scipy.toarray()
    C_y_dense_jax = C_y_jax.todense()
    print(
        "Max difference in C_y dense:", np.max(np.abs(C_y_dense_scipy - C_y_dense_jax))
    )
    assert np.allclose(C_y_dense_scipy, C_y_dense_jax, rtol=1e-10), (
        "C_y matrices differ"
    )

    # === Step 5e: Final Assembly ===
    # Scipy version
    term0_5_scipy = C_x_scipy @ M_mu_scipy
    term1_scipy = term0_5_scipy @ C_x_scipy.T
    term1_5_scipy = C_y_scipy @ M_mu_scipy
    term2_scipy = term1_5_scipy @ C_y_scipy.T
    term3_scipy = -(omega**2) * M_eps_scipy
    A_scipy = term1_scipy + term2_scipy + term3_scipy

    # Jax version
    term0_5_jax = _sp_matmul(C_x_jax, M_mu_jax)
    term1_jax = _sp_matmul(term0_5_jax, C_x_jax.T)
    term1_5_jax = _sp_matmul(C_y_jax, M_mu_jax)
    term2_jax = _sp_matmul(term1_5_jax, C_y_jax.T)
    term3_jax = _sp_matmul(
        _diags([-(omega**2)], [0], shape=(Nx * Ny, Nx * Ny)), M_eps_jax
    )
    A_jax = _sp_add(_sp_add(term1_jax, term2_jax), term3_jax)

    print("Max difference in term1:", np.max(np.abs(term1_scipy.data - term1_jax.data)))
    print("Max difference in term2:", np.max(np.abs(term2_scipy.data - term2_jax.data)))
    print("Max difference in term3:", np.max(np.abs(term3_scipy.data - term3_jax.data)))
    print(
        "Max difference in term0_5:",
        np.max(np.abs(term0_5_scipy.data - term0_5_jax.data)),
    )
    print(
        "Max difference in term1_5:",
        np.max(np.abs(term1_5_scipy.data - term1_5_jax.data)),
    )

    # Convert JAX CSR to scipy CSR for comparison
    A_jax_scipy = csr_matrix(
        (A_jax.data, A_jax.indices, A_jax.indptr), shape=A_jax.shape
    )

    diff = (A_scipy - A_jax_scipy).data
    print(
        "Max difference in final matrix A:",
        np.max(np.abs(diff)) if len(diff) > 0 else 0,
    )

    return A_scipy, A_jax


def run_FDFD_jax(
    eps: Union[np.ndarray, jnp.ndarray],
    mu: Union[np.ndarray, jnp.ndarray],
    source: Union[np.ndarray, jnp.ndarray],
    dx: float,
    dy: float,
    Nx: int,
    Ny: int,
    omega: float,
    pml_thickness: int = 40,
    sigma_max: float = 2,
    m: int = 3,
) -> jnp.ndarray:
    if isinstance(eps, np.ndarray):
        print("Converting eps array to jax array. This may impact performance.")
        eps = jnp.array(eps)
    if isinstance(mu, np.ndarray):
        print("Converting mu array to jax array. This may impact performance.")
        mu = jnp.array(mu)
    if isinstance(source, np.ndarray):
        print("Converting source array to jax array. This may impact performance.")
        source = jnp.array(source)

    A = _make_A(eps, mu, dx, dy, Nx, Ny, omega, pml_thickness, sigma_max, m)
    b = -1j * omega * source.flatten()

    Ez_new = _spsolve(A, b)
    Ez = Ez_new.reshape(Ny, Nx)

    return Ez

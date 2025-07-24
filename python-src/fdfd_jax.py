import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union
from jax import tree_util

# Alias for readability
CSR = sp.csr_matrix

# -----------------------------------------------------------------------------
# Register SciPy CSR as a JAX pytree (static leaves)
# -----------------------------------------------------------------------------


def _csr_flatten(mat):
    """Flatten function for csr_matrix -> treats data as static arrays."""
    children = (
        jnp.asarray(mat.data),
        jnp.asarray(mat.indices),
        jnp.asarray(mat.indptr),
    )
    aux = mat.shape
    return children, aux


def _csr_unflatten(aux, children):
    data, indices, indptr = children
    shape = aux
    return sp.csr_matrix(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=shape
    )


# Register once at import time
try:
    tree_util.register_pytree_node(sp.csr_matrix, _csr_flatten, _csr_unflatten)
except ValueError:
    # Already registered in interactive sessions
    pass

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _pml_profile(N: int, thickness: int, sigma_max: float, m: int) -> jnp.ndarray:
    """Return a 1-D PML conductivity profile of length *N*."""
    if thickness == 0:
        return jnp.zeros((N,))
    ramp = jnp.arange(thickness)
    left = sigma_max * (jnp.flip(ramp) / thickness) ** m
    right = sigma_max * (ramp / thickness) ** m
    sigma = jnp.zeros((N,))
    sigma = sigma.at[:thickness].set(left)
    sigma = sigma.at[N - thickness :].set(right)
    return sigma


# -----------------------------------------------------------------------------
# Custom sparse helpers with analytic VJPs
# -----------------------------------------------------------------------------


@jax.custom_vjp
def _sp_diags(diag_vec: jnp.ndarray) -> CSR:  # pragma: no cover
    """Sparse diag with backward pass = extract(diag)."""
    return sp.diags(np.asarray(diag_vec), offsets=0, format="csr")


def _sp_diags_fwd(diag_vec):  # pragma: no cover
    csr = sp.diags(np.asarray(diag_vec), offsets=0, format="csr")
    # No need to stash anything except maybe size for the backward
    return csr, diag_vec.shape


def _sp_diags_bwd(res, g):  # pragma: no cover
    (n,) = res
    if g is None:
        return (jnp.zeros((n,), dtype=jnp.result_type(float)),)
    # g is expected to be a CSR matrix coming from further sparse algebra
    if isinstance(g, sp.spmatrix):
        grad = jnp.asarray(g.diagonal())
    else:
        grad = jnp.zeros((n,), dtype=jnp.result_type(float))
    return (grad,)


_sp_diags.defvjp(_sp_diags_fwd, _sp_diags_bwd)

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def make_A_jax(
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
) -> CSR:
    """Assemble the 2-D TM operator *A* with PML in **CSR** form.

    The routine stays mostly on the host (SciPy) side but allows differentiation
    w.r.t. the diagonal material parameters through custom VJPs.
    """

    # 1. PML stretching profiles
    sig_x = _pml_profile(Nx, pml_thickness, sigma_max, m)
    sig_y = _pml_profile(Ny, pml_thickness, sigma_max, m)

    sigma_x_2d = jnp.tile(sig_x[None, :], (Ny, 1))
    sigma_y_2d = jnp.tile(sig_y[:, None], (1, Nx))

    eps0 = 8.85418e-12  # vacuum permittivity (F/m)
    s_x = 1.0 + 1j * sigma_x_2d / (omega * eps0)
    s_y = 1.0 + 1j * sigma_y_2d / (omega * eps0)

    # 2. Derivative operators (host-side sparse)
    Dx = sp.diags([-1.0, 1.0], [-1, 1], shape=(Nx, Nx), format="csr") / (2 * dx)
    Dy = sp.diags([-1.0, 1.0], [-1, 1], shape=(Ny, Ny), format="csr") / (2 * dy)

    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")

    Cx = sp.kron(Iy, Dx, format="csr")
    Cy = sp.kron(Dy, Ix, format="csr")

    # 3. Stretching and material matrices
    Sx_inv = _sp_diags(1.0 / s_x.flatten())
    Sy_inv = _sp_diags(1.0 / s_y.flatten())

    Cx = Sx_inv @ Cx
    Cy = Sy_inv @ Cy

    M_eps = _sp_diags(eps.flatten())
    M_mu_inv = _sp_diags(1.0 / mu.flatten())

    # 4. Assemble operator
    A = Cx @ M_mu_inv @ Cx.T + Cy @ M_mu_inv @ Cy.T - (omega**2) * M_eps
    return A


# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import tree_util


def make_A_jax(
    eps: jnp.Array, mu: jnp.Array, dx: float, dy: float, Nx: int, Ny: int, omega: float
) -> jax.experimental.sparse.CSR: ...

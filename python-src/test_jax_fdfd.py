import numpy as np
from main import material_init
from fdfd_jax import make_A_jax as make_A_jax
from fdfd import make_A
import jax.numpy as jnp


if __name__ == "__main__":
    Nx = 1000
    Ny = 1000
    dx = dy = 1e-3
    omega = 17e9

    source = np.zeros((Nx, Ny))
    source[200, 200] = 10

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

    eps_jax = jnp.array(eps)
    mu_jax = jnp.array(mu)

    A = make_A(eps, mu, dx, dy, Nx, Ny, omega)
    A_jax = make_A_jax(eps_jax, mu_jax, dx, dy, Nx, Ny, omega)

    A_data = A.data
    A_jax_data = A_jax.data

    # Check if they are close within reasonable tolerance
    tolerance = 1e-6
    print(jnp.sum(jnp.abs(A_jax_data)) / len(A_jax_data))
    is_close = np.allclose(A_data, A_jax_data, rtol=tolerance, atol=tolerance)
    print(f"Matrices are close within tolerance {tolerance}: {is_close}")

from fdfd_jax import _kron, _diags, _sp_matmul, _spsolve
from scipy.sparse import kron, diags
from scipy.sparse.linalg import spsolve
import jax.numpy as jnp


def test_diags():
    # Create test arrays
    diagonals = [[1, 2, 3], [4, 5]]
    offsets = [0, 1]
    shape = (3, 3)

    # Solve with scipy
    scipy_result = diags(diagonals, offsets, shape=shape)

    # Solve with jax
    jax_result = _diags(diagonals, offsets, shape)

    # Compare results
    assert (scipy_result.toarray() == jax_result.todense()).all(), (
        "Scipy and JAX results do not match for diags"
    )


def test_kron():
    # Create test arrays
    A = diags([[1, 2], [3]], [0, 1], shape=(2, 2))
    B = diags([[4, 5], [6]], [0, 1], shape=(2, 2))

    # Solve with scipy
    scipy_result = kron(A, B)

    # Convert to JAX CSR format
    A_jax = _diags([[1, 2], [3]], [0, 1], shape=(2, 2))
    B_jax = _diags([[4, 5], [6]], [0, 1], shape=(2, 2))

    # Solve with jax
    jax_result = _kron(A_jax, B_jax)

    # Compare results
    assert (scipy_result.toarray() == jax_result.todense()).all(), (
        "Scipy and JAX results do not match for kron"
    )


def test_matmul():
    # Create test arrays
    A = diags([[1, 2], [3]], [0, 1], shape=(2, 2))
    B = diags([[4, 5], [6]], [0, 1], shape=(2, 2))

    # Solve with scipy
    scipy_result = A @ B

    # Convert to JAX CSR format
    A_jax = _diags([[1, 2], [3]], [0, 1], shape=(2, 2))
    B_jax = _diags([[4, 5], [6]], [0, 1], shape=(2, 2))

    # Solve with jax
    jax_result = _sp_matmul(A_jax, B_jax)

    # Compare results
    assert (scipy_result.toarray() == jax_result.todense()).all(), (
        "Scipy and JAX results do not match for matmul"
    )


def test_spsolve():
    # Create test arrays
    A = diags([[4, 5], [1]], [0, 1], shape=(2, 2))
    b = [1, 2]

    # Solve with scipy
    scipy_result = spsolve(A, b)

    # Convert to JAX format
    A_jax = _diags([[4, 5], [1]], [0, 1], shape=(2, 2))
    b_jax = jnp.array(b)

    # Solve with jax
    jax_result = _spsolve(A_jax, b_jax)

    # Compare results
    assert jnp.allclose(scipy_result, jax_result), (
        "Scipy and JAX results do not match for spsolve"
    )


if __name__ == "__main__":
    print("Running tests...")
    print("Testing diags...")
    test_diags()
    print("Testing kron...")
    test_kron()
    print("Testing matmul...")
    test_matmul()
    print("Testing spsolve...")
    test_spsolve()

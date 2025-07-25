from fdfd_jax import make_A_jax as make_A, solve_linear
import jax
import jax.numpy as jnp
import tqdm
import matplotlib.pyplot as plt


# Function to simulate responses for a given design_region using explicit loop
def compute_responses(design_region, eps_base, source, mu, dx, dy, Nx, Ny, omegas):
    responses = []
    for omega in omegas:
        # Embed design into permittivity map
        eps = eps_base.at[75:175, 75:175].set(design_region)
        # Build system matrix and source vector
        A = make_A(eps, mu, dx, dy, Nx, Ny, omega.item())
        b = source.flatten() * 1j * omega
        # Solve and reshape
        sol = jnp.abs(solve_linear(A, b)).reshape((Nx, Ny))
        # Compute mean field in observation region
        response_val = jnp.mean(sol[110:140, 210])
        responses.append(response_val)
    # Stack into array
    return jnp.stack(responses)


# Loss function comparing normalized response to ideal
def loss_fn(
    design_region, eps_base, source, mu, dx, dy, Nx, Ny, omegas, ideal_response
):
    responses = compute_responses(
        design_region, eps_base, source, mu, dx, dy, Nx, Ny, omegas
    )
    responses_norm = responses / jnp.max(responses)
    return jnp.mean((responses_norm - ideal_response) ** 2)


# Main training routine
if __name__ == "__main__":
    # Simulation grid params
    Nx, Ny = 250, 250
    dx = dy = 1.0

    # Base permittivity: background = 1, obstacles = 3
    eps_base = jnp.ones((Nx, Ny))
    eps_base = eps_base.at[100:150, 0:75].set(3)
    eps_base = eps_base.at[100:150, 175:250].set(3)

    # Source distribution
    source = jnp.zeros((Nx, Ny))
    source = source.at[110:140, 40].set(3)

    # Initial design region (to be optimized)
    design_region = jnp.ones((100, 100))

    # Magnetic permeability (uniform)
    mu = jnp.ones((Nx, Ny))

    # Frequencies to probe
    omegas = jnp.linspace(10e9, 100e9, 10)
    # Desired binary frequency response
    ideal_frequency_response = jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # Plot initial permittivity and source
    plt.figure(figsize=(8, 8))
    plt.imshow(eps_base, cmap="gray")
    plt.imshow(source, cmap="Reds", alpha=(source > 0).astype(float) * 0.7)
    plt.colorbar(label="Permittivity")
    plt.title("Base Permittivity with Source")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Training parameters
    num_steps = 100
    lr = 0.1  # learning rate

    # Compute gradient w.r.t. design_region without JIT
    grad_fn = jax.grad(
        lambda d: loss_fn(
            d, eps_base, source, mu, dx, dy, Nx, Ny, omegas, ideal_frequency_response
        )
    )

    # Optimization loop
    for step in tqdm.trange(num_steps):
        # Compute current loss
        loss = loss_fn(
            design_region,
            eps_base,
            source,
            mu,
            dx,
            dy,
            Nx,
            Ny,
            omegas,
            ideal_frequency_response,
        )
        # Compute gradient
        grad_design = grad_fn(design_region)
        # Gradient descent update
        design_region = design_region - lr * grad_design
        # Enforce physical bounds (permittivity between 1 and 3)
        design_region = jnp.clip(design_region, 1.0, 3.0)

        # Logging
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step}, Loss: {loss:.6f}")

    # Plot optimized design region
    plt.figure(figsize=(6, 6))
    plt.imshow(design_region, cmap="viridis")
    plt.colorbar(label="Design Permittivity")
    plt.title("Optimized Design Region")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Compute and plot final frequency response
    final_responses = compute_responses(
        design_region, eps_base, source, mu, dx, dy, Nx, Ny, omegas
    )
    final_responses_norm = final_responses / jnp.max(final_responses)

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, final_responses_norm, "o-", label="Measured")
    plt.plot(omegas, ideal_frequency_response, "x--", label="Ideal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Response")
    plt.legend()
    plt.title("Frequency Response")
    plt.show()

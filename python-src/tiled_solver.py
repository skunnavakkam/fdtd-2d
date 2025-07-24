import numpy as np
from fdfd import make_A
import scipy
from collections import namedtuple


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


def run_fdfd_tiled(
    eps,
    mu,
    dx,
    dy,
    omega,
    source,
    *,
    patch_size: int = 100,
    padding: int = 30,
    pml_thickness: int = 10,
    n_passes: int = 3,  # <-- “a few passes”
    relax: float = 0.5,  # 0 < relax ≤ 1
    tol: float = 1e-2,
):
    """
    Multi-pass tiled FDFD solve (additive Schwarz with Dirichlet halo exchange).

    One outward-sorted sweep propagates the field by one patch; `n_passes`
    repeats that to let the solution settle.  Use `relax<1` for stability.
    """
    Nx, Ny = eps.shape
    assert eps.shape == mu.shape == source.shape, "shape mismatch"

    halo = pml_thickness
    inner = slice(halo, -halo or None)  # interior slice in a patch
    orig_source = source.copy()  # fixed RHS
    solution = np.zeros_like(source)  # start from zero

    # ---------------------------------------------------------------
    # 1.  Generate patches (same as before)
    # ---------------------------------------------------------------
    x_centers = range(patch_size // 2, Nx, patch_size)
    y_centers = range(patch_size // 2, Ny, patch_size)

    patches = []
    for cx in x_centers:
        for cy in y_centers:
            x0 = max(0, cx - patch_size // 2 - padding)
            x1 = min(Nx, cx + patch_size // 2 + padding)
            y0 = max(0, cy - patch_size // 2 - padding)
            y1 = min(Ny, cy + patch_size // 2 + padding)
            if (x1 - x0) > 2 * halo and (y1 - y0) > 2 * halo:
                patches.append(((x0, y0), (x1, y1)))

    # ---------------------------------------------------------------
    # 2.  Sort patches by distance from any with non-zero interior source
    # ---------------------------------------------------------------
    src_bool = orig_source != 0
    patches_with_dist, frontier, visited = [], set(), set()

    for idx, p in enumerate(patches):
        (x0, y0), (x1, y1) = p
        if np.any(src_bool[x0 + halo : x1 - halo, y0 + halo : y1 - halo]):
            patches_with_dist.append((p, 0))
            frontier.add(idx)
            visited.add(idx)

    d = 0
    while frontier and len(visited) < len(patches):
        d += 1
        nxt = set()
        for i in frontier:
            (ax0, ay0), (ax1, ay1) = patches[i]
            for j, p2 in enumerate(patches):
                if j in visited:
                    continue
                (bx0, by0), (bx1, by1) = p2
                if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                    visited.add(j)
                    nxt.add(j)
                    patches_with_dist.append((p2, d))
        frontier = nxt

    patches_with_dist.sort(key=lambda t: t[1])  # outward order

    # ---------------------------------------------------------------
    # 3.  Schwarz passes
    # ---------------------------------------------------------------
    for sweep in range(1, n_passes + 1):
        max_delta = 0.0

        for patch, _ in patches_with_dist:
            (x0, y0), (x1, y1) = patch

            p_eps = eps[x0:x1, y0:y1]
            p_mu = mu[x0:x1, y0:y1]
            p_source = orig_source[x0:x1, y0:y1]

            sol_patch = solution[x0:x1, y0:y1]
            bc = _extract_dirichlet_bc(sol_patch, halo)

            p_sol = _solve_patch(
                p_eps,
                p_mu,
                dx,
                dy,
                omega,
                p_source,
                pml_thickness=halo,
                dirichlet_bc=bc,
            )

            tgt = solution[x0 + halo : x1 - halo, y0 + halo : y1 - halo]
            new = p_sol[inner, inner]
            max_delta = max(max_delta, np.max(np.abs(new - tgt)))

            tgt[:] = (1 - relax) * tgt + relax * new

        print(max_delta)
        if max_delta < tol:
            break  # early convergence

    return solution

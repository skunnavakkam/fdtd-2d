from main import (
    grid_init,
    material_init,
    update_Hx_Hy,
    update_Ez,
    capture_snapshot,
    make_video_from_frames,
    sinusoidal,
    ricker,
)
import tqdm
import numpy as np

if __name__ == "__main__":
    rows = 200
    cols = 200
    dt = 1e-12
    dx = 1e-3
    nsteps = 1000
    nframes = 200

    Ez, Hx, Hy = grid_init(rows, cols)
    eps, mu = material_init(None, rows, cols)

    # Check Courant stability condition
    c = 1 / np.sqrt(eps.min() * mu.min())  # Speed of light in material
    courant = (c * dt) / dx
    print(courant)
    assert courant <= 1.0, f"Courant stability condition not met: {courant} > 1.0"

    for i in tqdm.trange(nsteps):
        Hx, Hy = update_Hx_Hy(Ez, Hx, Hy, mu, eps, dt, dx)
        Ez = update_Ez(Ez, Hx, Hy, mu, eps, dt, dx)

        Ez += ricker(rows, cols, rows // 2, cols // 2, i * dt, 3e9)

        if i % (nsteps // nframes) == 0:
            frame_num = i // (nsteps // nframes)
            capture_snapshot(Ez, eps, f"frames/frame_{frame_num:04d}.png", 1e-3, -1e-3)

    make_video_from_frames()

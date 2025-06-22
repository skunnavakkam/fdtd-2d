import numpy as np
import PIL 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil   
import ffmpeg
import tqdm

if os.path.exists("frames"):
    shutil.rmtree("frames")
os.makedirs("frames")


def update_Ez(Ez, Hx, Hy, mu, eps, dt, dx):
    # Ez.shape = (rows, cols)
    # Hx.shape = (rows, cols-1)
    # Hy.shape = (rows-1, cols)

    # Store previous Ez values for Mur ABC
    Ez_prev = Ez.copy()

    # ∂Hy/∂x: rows 1..rows-2, cols 1..cols-2
    dHy_dx = Hy[1:,   1:-1]  - Hy[1:,   :-2]

    # ∂Hx/∂y: rows 1..rows-2, cols 1..cols-2
    dHx_dy = Hx[1:-1, 1:]    - Hx[:-2,  1:]

    # now both dHy_dx and dHx_dy are (rows-2, cols-2), matching Ez[1:-1,1:-1]
    Ez[1:-1, 1:-1] += (dHy_dx - dHx_dy) * (dt / (eps[1:-1, 1:-1] * dx))

    # Apply Mur ABC on all boundaries
    c = 1/np.sqrt(mu[0,0] * eps[0,0])  # Speed of light in medium
    coef = (c*dt - dx)/(c*dt + dx)

    # Left boundary (5 px)
    for i in range(5):
        Ez[1:-1,i] = Ez_prev[1:-1,i+1] + coef*(Ez[1:-1,i+1] - Ez_prev[1:-1,i])
    
    # Right boundary (5 px) 
    for i in range(5):
        Ez[1:-1,-(i+1)] = Ez_prev[1:-1,-(i+2)] + coef*(Ez[1:-1,-(i+2)] - Ez_prev[1:-1,-(i+1)])
    
    # Top boundary (5 px)
    for i in range(5):
        Ez[i,1:-1] = Ez_prev[i+1,1:-1] + coef*(Ez[i+1,1:-1] - Ez_prev[i,1:-1])
    
    # Bottom boundary (5 px)
    for i in range(5):
        Ez[-(i+1),1:-1] = Ez_prev[-(i+2),1:-1] + coef*(Ez[-(i+2),1:-1] - Ez_prev[-(i+1),1:-1])
    
    # Corner points (simple average of adjacent points)
    for i in range(5):
        for j in range(5):
            Ez[i,j] = (Ez[i,j+1] + Ez[i+1,j])/2  # Top left
            Ez[i,-j-1] = (Ez[i,-j-2] + Ez[i+1,-j-1])/2  # Top right 
            Ez[-i-1,j] = (Ez[-i-2,j] + Ez[-i-1,j+1])/2  # Bottom left
            Ez[-i-1,-j-1] = (Ez[-i-2,-j-1] + Ez[-i-1,-j-2])/2  # Bottom right

    return Ez

def update_Hx_Hy(Ez, Hx, Hy, mu, eps, dt, dx) -> tuple[np.ndarray, np.ndarray]:
    # Update Hx: Hx -= (dt/mu) * dEz/dy
    # Hx has shape (rows, cols-1), Ez has shape (rows, cols)
    dEz_dy = Ez[1:, :-1] - Ez[:-1, :-1]    # → (N-1, M-1), matches Hx[:-1,:]
    Hx[:-1, :] -= (dt / (mu[:-1, :-1] * dx)) * dEz_dy

    # ∂Ez/∂x for Hy update: Hy_ij between Ez[i,j] and Ez[i,j+1]
    dEz_dx = Ez[:-1, 1:] - Ez[:-1, :-1]    # → (N-1, M-1), matches Hy[:,:-1]
    Hy[:, :-1] += (dt / (mu[:-1, :-1] * dx)) * dEz_dx
    
    return Hx, Hy

def grid_init(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns Ez, Hx, Hy
    return np.zeros((rows, cols)), np.zeros((rows, cols -1 )), np.zeros((rows - 1, cols))

def material_init(path: str | None, rows: int, cols: int, black_point: float = 1.3) -> tuple[np.ndarray, np.ndarray]:
    # Load and resize the material map
    if path is None:
        return np.ones((rows, cols)) * 8.85418e-12, np.ones((rows, cols)) * 4 * np.pi * 1e-7
    
    img = Image.open(path)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((cols, rows), Image.LANCZOS)
    
    # Convert to numpy arrays and normalize
    eps = 1 + (black_point - 1) * np.array(img) / 255.0  # Scale from 1 to black_point
    mu = np.ones((rows, cols))   # Permeability defaults to 1
    
    eps *= 8.85418e-12 # value of vacuum permittivity
    mu *= 4 * np.pi * 1e-7 # value of vacuum permeability

    return eps, mu

def make_video_from_frames():
    # frames contains png files
    # make them into a video of 15 fps using ffmpeg
    import subprocess
    
    try:
        # Construct ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '15',
            '-i', 'frames/frame_%04d.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            'animation.mp4'
        ]
        
        # Run ffmpeg command
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")

def capture_snapshot(Ez, eps, path, vmax=20, vmin=-20):
    # Normalize to [-1, 1]
    # print(np.max(Ez), np.min(Ez))

    normed = np.clip(Ez, vmin, vmax)

    # Create background grayscale image from eps
    eps_min = 8.85418e-12  # Vacuum permittivity
    eps_max = np.max(eps)
    if eps_max == eps_min:
        eps_gray = np.full_like(eps, 255, dtype=np.uint8)  # White if uniform
    else:
        eps_normed = (eps - eps_min) / (eps_max - eps_min)
        eps_gray = (eps_normed * 128 + 127).astype(np.uint8) # Scale to 127-255 (light gray to white)
    
    # Create RGB array with eps as background
    background = np.stack([eps_gray]*3, axis=-1)
    
    # Map Ez through colormap
    cmap = cm.get_cmap('seismic')   # blue-white-red
    rgba = cmap((normed - vmin) / (vmax - vmin))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Blend Ez values with background where significant
    mask = np.abs(normed) > (vmax - vmin) * 0.05  # 5% threshold
    mask = np.stack([mask]*3, axis=-1)
    final = np.where(mask, rgb, background)

    Image.fromarray(final).save(path)

def ricker(rows, cols, x_pos, y_pos, t, fc):
    tau = np.pi * fc * (t - 1/fc)
    amp = (1 - 2*tau**2) * np.exp(-tau**2)
    src = np.zeros((rows, cols), dtype=float)
    src[x_pos, y_pos] = amp
    return src

def sinusoidal(rows, cols, x_pos, y_pos, t, fc):
    src = np.zeros((rows, cols), dtype=float)
    # Gaussian envelope with width of 5 periods
    envelope = 1 - np.exp(-(t - 3000/fc)**2 / (2*(2/fc)**2))
    src[x_pos, y_pos] = envelope * np.sin(2 * np.pi * fc * t)
    return src

if __name__ == "__main__":
    rows = 1000
    cols = 1000
    dt = 5e-13
    dx = 5e-4
    nsteps = 10000
    nframes = 200

    Ez, Hx, Hy = grid_init(rows, cols)
    eps, mu = material_init("example_structure.png", rows, cols)

    # Check Courant stability condition
    c = 1 / np.sqrt(eps.min() * mu.min())  # Speed of light in material
    courant = (c * dt) / dx
    print(courant)
    assert courant <= 1.0, f"Courant stability condition not met: {courant} > 1.0"


    
    for i in tqdm.trange(nsteps):
        Hx, Hy = update_Hx_Hy(Ez, Hx, Hy, mu, eps, dt, dx)
        Ez = update_Ez(Ez, Hx, Hy, mu, eps, dt, dx)

        Ez += sinusoidal(rows, 
                        cols, 
                        200, 
                        50, 
                        i * dt, 
                        318e9)


        if i % (nsteps // nframes) == 0:
            frame_num = i // (nsteps // nframes)
            capture_snapshot(Ez, eps, f"frames/frame_{frame_num:04d}.png", 1e-3 , -1e-3) 

    make_video_from_frames()
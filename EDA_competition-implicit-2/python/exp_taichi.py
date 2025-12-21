import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
import time
import os

ti.init(arch=ti.cpu)

img = io.imread('data/DRIVE/training/1st_manual/21_manual1.gif')[0,:,:]
# Global variables
phi_n = np.zeros(img.shape, dtype=np.float32)
nx, ny = phi_n.shape

dt = 1e-3
nt = 2001
reinit_freq = 5
reinit_steps = 10
vis_freq = 500

def grad(x):
    return np.array(np.gradient(x))

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

# def stopping_fun(x):
#     return 1. / (1. + norm(grad(x))**2)

def stopping_fun(x):
    return np.exp(-(norm(grad(x))**2)/2*(0.1)**2)

F_vn = stopping_fun(img)

phi = ti.field(ti.f32, shape=(nx, ny))
F_v = ti.field(ti.f32, shape=(nx, ny))
        
phi.from_numpy(phi_n)
F_v.from_numpy(F_vn)


@ti.kernel
def initialize_phi():
    for i, j in phi:
        # phi[i, j] = ti.sqrt((i-nx/2)**2 + (j-ny/2)**2) - (nx/4)
        phi[i, j] = ti.max(nx/8 - i, i - nx/3, ny/8 - j, j - ny/3)

@ti.kernel
def apply_boundary_condition():
    # Left and right boundaries
    for j in range(ny):
        phi[0, j] = phi[1, j]
        phi[nx-1, j] = phi[nx-2, j]
    
    # Top and bottom boundaries
    for i in range(nx):
        phi[i, 0] = phi[i, 1]
        phi[i, ny-1] = phi[i, ny-2]

@ti.kernel
def upwind_scheme(phi_next: ti.template()):
    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    
    for i, j in phi:
        if i > 0 and i < nx-1 and j > 0 and j < ny-1:
            # Compute finite differences
            delt_xn = phi[i, j] - phi[i-1, j]
            delt_yn = phi[i, j] - phi[i, j-1]
            delt_xp = phi[i+1, j] - phi[i, j]
            delt_yp = phi[i, j+1] - phi[i, j]
            
            # Compute deltas for upwind scheme
            deltp = ti.max(delt_xn/dx, 0.0)**2 + ti.max(delt_yn/dy, 0.0)**2 + \
                    ti.min(delt_xp/dx, 0.0)**2 + ti.min(delt_yp/dy, 0.0)**2
            
            deltn = ti.max(delt_xp/dx, 0.0)**2 + ti.max(delt_yp/dy, 0.0)**2 + \
                    ti.min(delt_xn/dx, 0.0)**2 + ti.min(delt_yn/dy, 0.0)**2
            
            # Speed function
            F = F_v[i, j]
            
            # Right-hand side of the level set equation
            rhs = ti.max(F, 0.0) * ti.sqrt(deltp) + ti.min(F, 0.0) * ti.sqrt(deltn)
            
            # Euler forward time integration
            phi_next[i, j] = phi[i, j] - dt * rhs

@ti.kernel
def reinitialize(phi_temp: ti.template()):
    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    
    for i, j in phi:
        if i > 0 and i < nx-1 and j > 0 and j < ny-1:
            # Compute gradient using central differences
            grad_phi_x = (phi[i+1, j] - phi[i-1, j]) / (2.0 * dx)
            grad_phi_y = (phi[i, j+1] - phi[i, j-1]) / (2.0 * dy)
            grad_phi = ti.sqrt(grad_phi_x**2 + grad_phi_y**2)
            
            # Sign function
            s = phi[i, j] / ti.sqrt(phi[i, j]**2 + (grad_phi*dx)**2)
            
            # Update phi
            dphi = s * (grad_phi - 1.0)
            phi_temp[i, j] = phi[i, j] - dt * dphi

@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(src):
        dst[I] = src[I]

def evolve():
    # Create temporary fields for updates
    phi_next = ti.field(dtype=ti.f32, shape=(nx, ny))
    phi_temp = ti.field(dtype=ti.f32, shape=(nx, ny))
    
    # Main evolution loop
    for t in range(nt):
        # Visualization
        if t % vis_freq == 0:
            visualize(t)
        
        # Level set evolution step
        upwind_scheme(phi_next)
        copy_field(phi_next, phi)
        apply_boundary_condition()
        
        # Reinitialization
        if t % reinit_freq == 0:
            copy_field(phi, phi_temp)
            for _ in range(reinit_steps):
                reinitialize(phi_temp)
                apply_boundary_condition()
                # Swap fields
                copy_field(phi_temp, phi)

def visualize(t):
    # Copy data from Taichi fields to NumPy arrays for visualization
    phi_np = phi.to_numpy()

    
    plt.figure(figsize=(10, 8))
    plt.contour(phi_np, levels=[0], colors='r')
    plt.imshow(img, cmap='gray')
    plt.title(f"time={t*dt}")
    plt.colorbar()
    plt.savefig(f"output/levelset_t{t}.png")
    plt.close()

def main():
    
    # os.remove('output')
    # os.mkdir('output')
    # Initialize level set function
    initialize_phi()
    
    start = time.time()
    # Run level set evolution
    evolve()
    end = time.time()
    print(f'Time : {end - start}')

if __name__ == '__main__':
    main()
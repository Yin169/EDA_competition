import numpy as np
import scipy.ndimage
import scipy.signal
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from skimage import color, io, transform
import taichi as ti
import time
import os

ti.init(arch=ti.cpu)

def grad(x):
    return np.array(np.gradient(x))

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

# def stopping_fun(x):
#     return 1. / (1. + norm(grad(x))**2)

def stopping_fun(x):
    return np.exp(-(norm(grad(x))**2)/2*(0.1)**2)

@ti.data_oriented
class DualTimeStepping:
    def __init__(self, phi_np, dt, nt, nx, ny, dx, dy, sudot, gamma, max_pseudo_iter, pseudo_tol, reindt):
        # Convert numpy arrays to Taichi fields
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt  # physical time step
        self.nt = nt  # number of physical time steps
        self.sudot = sudot  # time step for pseudo-time iterations
        self.gamma = gamma
        self.max_pseudo_iter = max_pseudo_iter  # maximum pseudo-time iterations
        self.pseudo_tol = pseudo_tol  # convergence tolerance for pseudo-time iterations
        self.reindt = reindt
        
        # Define Taichi fields
        self.phi = ti.field(ti.f32, shape=(nx, ny))
        self.phi_m = ti.field(ti.f32, shape=(nx, ny))
        self.phi_n = ti.field(ti.f32, shape=(nx, ny))
        self.phi_nm1 = ti.field(ti.f32, shape=(nx, ny))
        self.delta_m = ti.field(ti.f32, shape=(nx, ny))
        self.F_v = ti.field(ti.f32, shape=(nx, ny))
        
        # Copy data to Taichi fields
        self.phi.from_numpy(phi_np)
        self.F_v.from_numpy(F_v)
        
        # Initialize
        self.initialize()

    @ti.func
    def spatialsch(self, u: ti.template(), i, j, ii, jj):
        return u[i, j] - u[ii, jj]

    @ti.kernel
    def initialize(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            # self.phi[i, j] = ti.sqrt((i-self.nx/2)*(i-self.nx/2)+(j-self.ny/2)*(j-self.ny/2))-(self.nx/3)
            # Alternative initialization:
            self.phi[i, j] = ti.max(self.nx/8 - i, i - self.nx/3, self.ny/8 - j, j - self.ny/3)


    @ti.func
    def F(self, i, j):
        return self.F_v[i, j]

    @ti.func
    def rhs(self, phi: ti.template(), i, j):
        delt_xn = self.spatialsch(phi, i, j, i-1, j)
        delt_yn = self.spatialsch(phi, i, j, i, j-1)
        delt_xp = self.spatialsch(phi, i+1, j, i, j)
        delt_yp = self.spatialsch(phi, i, j+1, i, j)
        
        deltp = ti.max(delt_xn/self.dx, 0.0)**2 + ti.max(delt_yn/self.dy, 0.0)**2 + \
                ti.min(delt_xp/self.dx, 0.0)**2 + ti.min(delt_yp/self.dy, 0.0)**2
                
        deltn = ti.max(delt_xp/self.dx, 0.0)**2 + ti.max(delt_yp/self.dy, 0.0)**2 + \
                ti.min(delt_xn/self.dx, 0.0)**2 + ti.min(delt_yn/self.dy, 0.0)**2
                
        F_val = self.F(i, j)
        return ti.max(F_val, 0.0) * ti.sqrt(deltp) + ti.min(F_val, 0.0) * ti.sqrt(deltn)

    @ti.func
    def residual(self, i, j):
        return ((1 + self.gamma) * (self.phi_m[i, j] - self.phi_n[i, j]) - 
                self.gamma * (self.phi_n[i, j] - self.phi_nm1[i, j]))/self.dt + self.rhs(self.phi_m, i, j)
    
    @ti.func
    def dRdphi(self, i, j):
        # Numerical approximation of the derivative of the residual with respect to phi
        epsilon = 1e-6
        
        # Store original value
        original_val = self.phi_m[i, j]
        
        # Calculate first residual
        res1 = self.rhs(self.phi_m, i, j)
        
        # Perturb and calculate second residual
        self.phi_m[i, j] += epsilon
        res2 = self.rhs(self.phi_m, i, j)
        
        # Restore original value
        self.phi_m[i, j] = original_val
        
        return (res2 - res1) / epsilon

    @ti.func
    def Coe(self, i, j):
        return 1/self.sudot + (1+self.gamma) / self.dt + self.dRdphi(i, j)

    @ti.kernel
    def copy_field(self, dst: ti.template(), src: ti.template()):
        for i, j in dst:
            dst[i, j] = src[i, j]

    @ti.kernel
    def apply_boundary_conditions(self, phi: ti.template()):
        # Zero gradient at boundaries
        for j in range(self.ny):
            phi[0, j] = phi[1, j]
            phi[self.nx-1, j] = phi[self.nx-2, j]
        
        for i in range(self.nx):
            phi[i, 0] = phi[i, 1]
            phi[i, self.ny-1] = phi[i, self.ny-2]

    @ti.kernel
    def compute_delta_m(self) -> ti.f32:
        max_residual = 0.0
        for i, j in ti.ndrange((1, self.nx-1), (1, self.ny-1)):
            self.delta_m[i, j] = -self.residual(i, j) / self.Coe(i, j)
            max_residual = ti.max(max_residual, ti.abs(self.delta_m[i, j]))
        return max_residual

    @ti.kernel
    def update_phi_m(self):
        for i, j in self.phi_m:
            self.phi_m[i, j] += self.delta_m[i, j]
		

    @ti.kernel
    def reset_delta_m(self):
        for i, j in self.delta_m:
            self.delta_m[i, j] = 0.0

    @ti.kernel
    def reinitialize(self, step: int):
        # Create a temporary field to store phi values
        for i, j in self.phi:
            self.phi_m[i, j] = self.phi[i, j]  # Use phi_m as temporary storage
        
        for _ in range(step):
            for i, j in ti.ndrange((1, self.nx-1), (1, self.ny-1)):
                grad_phi_x = (self.phi_m[i+1, j] - self.phi_m[i-1, j]) / (2 * self.dx)
                grad_phi_y = (self.phi_m[i, j+1] - self.phi_m[i, j-1]) / (2 * self.dy)
                grad_phi = ti.sqrt(grad_phi_x**2 + grad_phi_y**2)
                s = self.phi_m[i, j] / ti.sqrt(self.phi_m[i, j]**2 + (grad_phi*self.dx)**2)
                dphi = s * (grad_phi - 1)
                self.phi[i, j] = self.phi_m[i, j] -  self.sudot * dphi
            
            # Apply boundary conditions after each iteration
            self.apply_boundary_conditions(self.phi)
            # Copy updated values back to phi_m for next iteration
            self.copy_field(self.phi_m, self.phi)

    def solve(self):
        # Initialize solution arrays
        self.copy_field(self.phi_m, self.phi)
        self.copy_field(self.phi_n, self.phi)
        self.copy_field(self.phi_nm1, self.phi)

        for t in range(self.nt):
            if t % 50 == 0:
                # Visualize current solution
                phi_np = self.phi.to_numpy()
                plt.contour(phi_np, levels=[0], colors = 'r')
                io.imshow(img)
                plt.title(f"time={t*self.dt}")
                plt.savefig(f"out/implicit_{t}.png")
                plt.clf()
            
            # Store previous solutions properly
            self.copy_field(self.phi_nm1, self.phi_n)
            self.copy_field(self.phi_n, self.phi)
            
            self.apply_boundary_conditions(self.phi)
            self.copy_field(self.phi_m, self.phi)
            
            # Pseudo-time iterations
            for pst in range(self.max_pseudo_iter):
                # Reset delta_m

                self.reset_delta_m()
                # Compute update
                residual_norm = self.compute_delta_m()
                
                # Apply update
                self.update_phi_m()
                self.apply_boundary_conditions(self.phi_m)
                
                # Check for convergence
                # if pst % 100 == 0:
                    # print(f"Pseudo-time step {pst+1}: Residual norm = {residual_norm}")
                if residual_norm < self.pseudo_tol:
                    break
            
            self.copy_field(self.phi, self.phi_m)
            
            if t % 5 == 0:
                self.reinitialize(10)

    def get_solution(self):
        return self.phi.to_numpy()

if __name__ == '__main__':
    # os.remove('out')
    # os.mkdir('out')
    img = io.imread('data/DRIVE/training/1st_manual/21_manual1.gif')[0,:,:]

    # idx = np.where(gt == True)
    # x, y = int(np.mean(idx[0])), int(np.mean(idx[1]))
    # print(x, y)

    # # img = img - np.mean(img)
    # img = color.rgb2gray(img)
    # # img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma=1)


    F_v = stopping_fun(img)
    print(img.shape)

    phi = np.zeros(img.shape, dtype=np.float32)
    nx, ny = phi.shape
    dx = np.float32(1.0/(nx-1))
    dy = np.float32(1.0/(ny-1))
    dt = 1e-2
    nt = 201
    sudo_t = 1e-3
    gamma = 0.5
    max_pseudo_iter = 2000
    pseudo_tol = 1e-2
    reindt = 1e-3

    solver = DualTimeStepping(phi, dt, nt, nx, ny, dx, dy, sudo_t, gamma, max_pseudo_iter, pseudo_tol, reindt)
    start = time.time()
    solver.solve()
    end = time.time()
    print(f'Time : {end - start}')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 定义神经网络
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # 最后一层不需要激活函数
                self.layers.append(nn.Tanh())
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

import torch
import numpy as np
import matplotlib.pyplot as plt

def burgers_exact_solution(x, t, nu=0.01/np.pi):
    """
    使用谱方法求解Burgers方程（修复NaN问题）
    
    参数:
    x: 空间坐标 (numpy数组)
    t: 时间坐标 (numpy数组)
    nu: 粘性系数
    
    返回:
    u: 速度场 (2D numpy数组, 形状为(len(x), len(t)))
    """
    # 转换为PyTorch张量
    x_torch = torch.tensor(x, dtype=torch.float32)
    t_torch = torch.tensor(t, dtype=torch.float32)
    
    N_x = len(x)
    N_t = len(t)
    dx = x[1] - x[0]
    
    # 创建初始条件: u(x,0) = -sin(pi*x)
    u = -torch.sin(np.pi * x_torch)
    
    # 时间步进
    u_solution = torch.zeros((N_t, N_x), dtype=torch.float32)
    u_solution[0, :] = u
    
    # 实际时间步长（确保至少有2个时间点）
    dt = t[1] - t[0] if N_t > 1 else 0.0
    
    # 预计算波数
    k = 2 * np.pi * torch.fft.fftfreq(N_x, d=dx)
    
    # 2/3规则：创建去混叠掩码（关键修复！）
    cutoff = 2 * N_x // 3
    dealias = torch.ones(N_x, dtype=torch.float32)
    # 处理偶数和奇数N_x的情况
    if N_x % 2 == 0:
        dealias[cutoff:N_x-cutoff+1] = 0.0
    else:
        dealias[cutoff:N_x-cutoff] = 0.0
    
    # 时间推进（使用更稳定的半隐式方法）
    for i in range(1, N_t):
        # 检查NaN（安全措施）
        if torch.isnan(u).any():
            print(f"Warning: NaN detected at time step {i-1}. Using previous state.")
            u = u_solution[i-1, :]
            u_solution[i:, :] = u
            break
        
        # 计算一阶和二阶导数
        u_x, u_xx = spectral_derivative_1d_torch(u, dx)
        
        # 计算非线性项 (u·u_x)
        nonlinear_term = u * u_x
        
        # 应用2/3规则去混叠（关键修复！）
        nonlinear_hat = torch.fft.fft(nonlinear_term)
        nonlinear_hat = nonlinear_hat * dealias
        nonlinear_term = torch.fft.ifft(nonlinear_hat).real
        
        # 使用半隐式方法（关键修复！）
        # u_t = -nonlinear_term + nu·u_xx
        # 将扩散项处理为隐式，提高稳定性
        
        # 转换到频域
        u_hat = torch.fft.fft(u)
        
        # 计算右侧（频域）
        # 注意：这里我们使用半隐式方法
        rhs_hat = torch.fft.fft(-nonlinear_term) + nu * (-k**2) * u_hat
        
        # 更新频域解（使用隐式欧拉方法处理扩散项）
        # (1 + nu*dt*k^2) * u_hat_new = u_hat + dt * rhs_hat
        u_hat_new = (u_hat + dt * torch.fft.fft(-nonlinear_term)) / (1 + nu * dt * k**2)
        
        # 转换回物理域
        u = torch.fft.ifft(u_hat_new).real
        
        # 再次检查NaN（安全措施）
        if torch.isnan(u).any():
            print(f"Warning: NaN after update at time step {i}. Reducing time step.")
            # 尝试使用更小的时间步长
            dt_small = dt / 2
            # 重新计算（简化版，实际应用中应递归减小步长）
            u_hat = torch.fft.fft(u_solution[i-1, :])
            u_hat_new = (u_hat + dt_small * torch.fft.fft(-nonlinear_term)) / (1 + nu * dt_small * k**2)
            u = torch.fft.ifft(u_hat_new).real
            
            # 如果仍然有NaN，使用上一个状态
            if torch.isnan(u).any():
                u = u_solution[i-1, :]
        
        # 保存解
        u_solution[i, :] = u
    
    # 转换为numpy数组并调整维度
    u_solution = u_solution.numpy().T  # 转置以匹配(x,t)维度
    
    return u_solution

# 使用PyTorch实现谱导数计算
def spectral_derivative_1d_torch(u, dx):
    """
    使用PyTorch实现谱方法计算一阶和二阶导数
    
    参数:
    u: 形状为(N,)的张量，表示函数值
    dx: 空间步长
    
    返回:
    du_dx: 一阶导数
    d2u_dx2: 二阶导数
    """
    N = len(u)
    k = 2 * np.pi * torch.fft.fftfreq(N, d=dx, device=u.device)
    
    # 傅里叶变换
    u_hat = torch.fft.fft(u)
    
    # 计算导数
    du_hat = 1j * k * u_hat
    d2u_hat = -k**2 * u_hat
    
    # 逆傅里叶变换
    du_dx = torch.fft.ifft(du_hat).real
    d2u_dx2 = torch.fft.ifft(d2u_hat).real
    
    return du_dx, d2u_dx2


# 物理信息压缩感知求解器
class PhysicsInformedCompressedSensing:
    def __init__(self, x, t, u_obs, observed_mask, nu, lambda_sparse=0.1):
        """
        初始化物理信息压缩感知求解器
        
        参数:
        x: 空间坐标 (1D numpy数组)
        t: 时间坐标 (1D numpy数组)
        u_obs: 观测到的数据 (2D numpy数组，形状为(N_x, N_t))
        observed_mask: 布尔掩码，表示哪些点被观测到
        nu: 粘性系数
        lambda_sparse: 稀疏性正则化的权重
        """
        self.x = x
        self.t = t
        self.u_obs = u_obs
        self.observed_mask = observed_mask
        self.nu = nu
        self.lambda_sparse = lambda_sparse
        
        self.N_x = len(x)
        self.N_t = len(t)
        
        # 创建网格
        self.X, self.T = np.meshgrid(x, t, indexing='ij')
        
        # 创建神经网络
        layers = [2, 50, 50, 50, 50, 1]
        self.model = PINN(layers)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def compute_data_loss(self):
        """计算数据损失"""
        # 获取观测点的坐标
        x_obs = self.X[self.observed_mask]
        t_obs = self.T[self.observed_mask]
        u_obs = self.u_obs[self.observed_mask]
        
        # 转换为PyTorch张量
        x_obs = torch.tensor(x_obs, dtype=torch.float32).unsqueeze(1)
        t_obs = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
        u_obs = torch.tensor(u_obs, dtype=torch.float32).unsqueeze(1)
        
        # 预测
        u_pred = self.model(x_obs, t_obs)
        
        # 计算损失
        return torch.mean((u_pred - u_obs)**2)
    
    def compute_physics_loss(self):
        """计算物理损失（Burgers方程的残差）"""
        # 采样内部点
        num_samples = 1000
        x = torch.rand(num_samples) * (self.x[-1] - self.x[0]) + self.x[0]
        t = torch.rand(num_samples) * (self.t[-1] - self.t[0]) + self.t[0]
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        
        # 设置梯度跟踪
        x.requires_grad = True
        t.requires_grad = True
        
        # 预测
        u = self.model(x, t)
        
        # 计算时间导数
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # 为每个点计算空间导数（使用自动微分）
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # 计算Burgers方程残差
        residual = u_t + u * u_x - self.nu * u_xx
        
        return torch.mean(residual**2)
    
    def compute_spectral_sparse_loss(self):
        """计算频域稀疏性损失"""
        # 采样一些时间点
        num_time_samples = 10
        time_indices = np.random.choice(self.N_t, num_time_samples, replace=False)
        
        sparse_loss = 0
        
        for t_idx in time_indices:
            # 获取此时间的所有空间点
            x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(1)
            t_val = torch.full((self.N_x, 1), self.t[t_idx], dtype=torch.float32)
            
            # 预测
            with torch.no_grad():
                u = self.model(x, t_val).squeeze()
            
            # 计算频域表示
            u_hat = torch.fft.fft(u)
            
            # L1正则化（鼓励稀疏性）
            sparse_loss += torch.mean(torch.abs(u_hat))
        
        return sparse_loss / num_time_samples
    
    def train(self, n_epochs=2000, lambda_data=1.0, lambda_phys=0.1):
        """训练模型"""
        losses = []
        data_losses = []
        physics_losses = []
        sparse_losses = []
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # 计算各项损失
            data_loss = self.compute_data_loss()
            physics_loss = self.compute_physics_loss()
            sparse_loss = self.compute_spectral_sparse_loss()
            
            # 总损失
            loss = lambda_data * data_loss + lambda_phys * physics_loss + self.lambda_sparse * sparse_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            losses.append(loss.item())
            data_losses.append(data_loss.item())
            physics_losses.append(physics_loss.item())
            sparse_losses.append(sparse_loss.item())
            
            # 打印进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, "
                      f"Data Loss: {data_loss.item():.6f}, "
                      f"Physics Loss: {physics_loss.item():.6f}, "
                      f"Sparse Loss: {sparse_loss.item():.6f}")
        
        return losses, data_losses, physics_losses, sparse_losses
    
    def predict(self):
        """预测完整解"""
        x = torch.tensor(self.X.flatten(), dtype=torch.float32).unsqueeze(1)
        t = torch.tensor(self.T.flatten(), dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            u_pred = self.model(x, t).numpy().reshape(self.N_x, self.N_t)
        
        return u_pred

# 主函数
def main():
    # 参数设置
    x_range = [0, 1]
    t_range = [0, 1]
    N_x = 1000  # 空间离散点数
    N_t = 1000  # 时间离散点数
    nu = 0.01 / np.pi  # 粘性系数
    sparse_ratio = 0.05  # 稀疏测量的比例
    
    # 生成空间和时间网格
    x = np.linspace(x_range[0], x_range[1], N_x)
    t = np.linspace(t_range[0], t_range[1], N_t)
    
    # 生成精确解
    u_exact = burgers_exact_solution(x, t, nu)
    
    # 创建稀疏观测
    observed_mask = np.random.rand(N_x, N_t) < sparse_ratio
    u_obs = np.copy(u_exact)
    u_obs[~observed_mask] = np.nan  # 未观测到的点设为NaN
    
    # 创建并训练求解器
    solver = PhysicsInformedCompressedSensing(x, t, u_exact, observed_mask, nu, lambda_sparse=0.01)
    losses, data_losses, physics_losses, sparse_losses = solver.train(
        n_epochs=2000, lambda_data=1.0, lambda_phys=0.1
    )
    
    # 预测完整解
    u_pred = solver.predict()
    
    # 计算误差
    error = np.mean((u_pred - u_exact)**2)
    print(f"Mean Squared Error: {error:.6f}")
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.semilogy(losses, label='Total')
    plt.semilogy(data_losses, label='Data')
    plt.semilogy(physics_losses, label='Physics')
    plt.semilogy(sparse_losses, label='Sparse')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # 观测数据
    plt.subplot(2, 3, 2)
    observed_data = np.copy(u_exact)
    observed_data[~observed_mask] = np.nan
    plt.pcolormesh(t, x, observed_data, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Observed Data')
    
    # 预测解
    plt.subplot(2, 3, 3)
    plt.pcolormesh(t, x, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'Predicted u(x,t)\nMSE: {error:.6f}')
    
    # 精确解
    plt.subplot(2, 3, 4)
    plt.pcolormesh(t, x, u_exact, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Exact u(x,t)')
    
    # 误差图
    plt.subplot(2, 3, 5)
    plt.pcolormesh(t, x, np.abs(u_pred - u_exact), cmap='hot')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Absolute Error')
    
    # 时间切片比较
    plt.subplot(2, 3, 6)
    time_idx = N_t // 2  # 中间时间点
    plt.plot(x, u_exact[:, time_idx], 'b-', linewidth=2, label='Exact')
    plt.plot(x, u_pred[:, time_idx], 'r--', linewidth=2, label='Prediction')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'u(x,t) at t = {t[time_idx]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('burgers_results.png')
    plt.show()

if __name__ == "__main__":
    main()
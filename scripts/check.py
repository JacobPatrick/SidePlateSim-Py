"""
二维圆环雷诺方程解析解可视化
压力分布: p(ρ,θ) = p0 + p_squeeze(ρ) + p_wedge(ρ,θ)
膜厚假设: h(y) = h0 + k*y (可修改为任意函数)
"""

import numpy as np
import matplotlib.pyplot as plt

# ================== 1. 参数设置 ==================
# 几何参数
r = 0.013  # 内径 [m]
R = 0.033  # 外径 [m]

# 润滑参数
h0 = 1e-5  # 基准膜厚 [m]
k = 1e-5  # 膜厚梯度: h(y)=h0+k*y
mu = 5e-4  # 动力粘度 [Pa·s]
Omega = 523.0  # 旋转角速度 [rad/s], 逆时针为正
Vs = -1e-3  # 挤压速度 [m/s], Vs<0 表示间隙减小
p0 = 1e5  # 边界压力 [Pa]

# 数值参数
N_rho = 200  # 径向网格数
N_theta = 360  # 周向网格数


# ================== 2. 核心函数定义 ==================
def pressure_distribution(rho, theta):
    """计算压力分布 p(ρ,θ) - p0"""
    p_basic = (
        3
        * mu
        * Vs
        * ((rho**2 - r**2) - (R**2 - r**2) * np.log(rho / r) / np.log(R / r))
        / h0**3
    )
    E1 = 9 * Vs * (R**2 + r**2) / (16 * h0) - 3 * Vs * (
        R**2 * np.log(R) - r**2 * np.log(r)
    ) / (8 * h0 * np.log(R / r))
    E2 = -3 * Vs * R**2 * r**2 / (16 * h0)
    F = (
        3
        * mu
        * Omega
        * (rho**3 - (R**2 + r**2) * rho + R**2 * r**2 / rho)
        / (4 * h0**3)
    )
    G = (
        E1 * rho
        + E2 / rho
        - 9 * Vs * rho**3 / (16 * h0)
        + 3 * Vs * (R**2 - r**2) * rho * np.log(rho) / (8 * h0 * np.log(R / r))
    )

    return p_basic + k * (F * np.cos(theta) + G * np.sin(theta))


# ================== 3. 网格生成与计算 ==================
print("正在生成网格并计算压力场...")
rho_vals = np.linspace(r, R, N_rho)
theta_vals = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
RHO, THETA = np.meshgrid(rho_vals, theta_vals, indexing='ij')

# 转换为 Cartesian 坐标 (用于绘图)
X = RHO * np.cos(THETA)
Y = RHO * np.sin(THETA)

# 计算压力场 (向量化加速)
P = np.zeros_like(RHO)
for i in range(N_rho):
    for j in range(N_theta):
        P[i, j] = pressure_distribution(RHO[i, j], THETA[i, j])
P_total = p0 + P  # 加上边界压力

p_min, p_max = P_total.min(), P_total.max()

print(f"✓ 压力范围: [{P_total.min():.2f}, {P_total.max():.2f}] Pa")

# ================== 4. 可视化 ==================
_, ax = plt.subplots(figsize=(7, 6))
mask = (RHO >= r) & (RHO <= R)  # 确保只显示圆环域
cf = ax.pcolormesh(
    X, Y, P_total, shading='auto', cmap='jet', vmin=p_min, vmax=p_max
)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Pressure Distribution')
plt.colorbar(cf, ax=ax, label='Pressure [Pa]', fraction=0.046, pad=0.04)
# 绘制内外边界
circle_in = plt.Circle(
    (0, 0), r, color='white', fill=False, linewidth=1.5, linestyle='--'
)
circle_out = plt.Circle(
    (0, 0), R, color='white', fill=False, linewidth=1.5, linestyle='--'
)
ax.add_artist(circle_in)
ax.add_artist(circle_out)

plt.tight_layout()
plt.savefig(
    'results/figures/annular_pressure.png', dpi=300, bbox_inches='tight'
)
print("✓ 2D 云图已保存为 'annular_pressure.png'")

plt.close()

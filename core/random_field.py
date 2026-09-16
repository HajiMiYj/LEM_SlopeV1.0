# -*- coding: utf-8 -*-
"""
空间随机场生成器

用途: 给定区域, 生成具有空间相关性的参数场 (c、φ、γ 等)
方法: 移动平均法 (高斯核卷积白噪声)
  · 计算量 O(N), 支持大网格
  · 各向异性通过 θx / θy 独立控制
"""
import numpy as np
from scipy.signal import convolve2d


class RandomField:
    """二维平稳高斯随机场

    用法:
      rf = RandomField(nx=80, ny=60, theta_x=5.0, theta_y=2.0)
      values = rf.sample(mean=20.0, std=5.0, seed=42)   # shape (ny, nx)
    """

    def __init__(self, nx: int, ny: int,
                 theta_x: float, theta_y: float):
        self.nx = max(3, int(nx))
        self.ny = max(3, int(ny))
        self.theta_x = max(0.1, float(theta_x))
        self.theta_y = max(0.1, float(theta_y))

        # 预生成高斯核 (按网格索引计算)
        kx = int(max(1, np.ceil(3 * self.theta_x / (self.nx / 20.0))))
        ky = int(max(1, np.ceil(3 * self.theta_y / (self.ny / 20.0))))
        kx = min(kx, self.nx)
        ky = min(ky, self.ny)
        gx = np.arange(-kx, kx + 1)
        gy = np.arange(-ky, ky + 1)
        GX, GY = np.meshgrid(gx, gy)
        # θ 单位是米, 网格间距在这里用"相对格数"近似 (后续按实际 grid 调整)
        self.kernel = np.exp(-(GX ** 2 / (2 * (self.theta_x ** 2))
                               + GY ** 2 / (2 * (self.theta_y ** 2))))
        self.kernel /= self.kernel.sum()

    def sample(self, mean: float, std: float, seed: int = None) -> np.ndarray:
        """生成一次随机场实现, shape = (ny, nx)"""
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((self.ny, self.nx))
        field = convolve2d(noise, self.kernel, mode="same", boundary="symm")
        s = field.std()
        if s > 1e-9:
            field = (field - field.mean()) / s
        else:
            field = np.zeros_like(field)
        return field * std + mean


def generate_field_on_grid(xs, ys, theta_x, theta_y,
                           mean, std, seed=None):
    """在给定 x/y 坐标网格上生成场

    xs: 1D (nx,)   横坐标
    ys: 1D (ny,)   纵坐标
    返回: (ny, nx) 值矩阵
    """
    nx = len(xs)
    ny = len(ys)
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0

    # θ 用"格数"表示
    thx_cells = max(0.5, theta_x / abs(dx))
    thy_cells = max(0.5, theta_y / abs(dy))

    rf = RandomField(nx=nx, ny=ny, theta_x=thx_cells, theta_y=thy_cells)
    return rf.sample(mean, std, seed)
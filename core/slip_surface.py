# -*- coding: utf-8 -*-
"""
边坡滑动面几何抽象体系
支持经典圆弧 (Circular) 与任意多段折线形/顺层弱面 (Polygonal / Non-Circular)
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class BaseSlipSurface(ABC):
    """滑动面抽象基类"""
    def __init__(self, surface_type: str):
        self.surface_type = surface_type  # 'circular' 或 'polygonal'

    @abstractmethod
    def get_x_range(self, gx: np.ndarray, gy: np.ndarray) -> Optional[Tuple[float, float]]:
        """计算滑面与地表线相交形成的进出坡范围 (x_start, x_end)"""
        pass

    @abstractmethod
    def get_y_base(self, x: float) -> float:
        """获取指定水平 X 坐标处的滑面底高程"""
        pass

    @abstractmethod
    def get_alpha(self, x: float) -> float:
        """获取指定水平 X 坐标处滑面底坡倾角 (弧度, 下滑方向倾角为正)"""
        pass

    @abstractmethod
    def get_moment_center(self) -> Tuple[float, float]:
        """获取力矩参考基准中心 (x0, y0)"""
        pass


class CircularSlipSurface(BaseSlipSurface):
    """经典圆弧滑动面"""
    def __init__(self, xc: float, yc: float, R: float):
        super().__init__("circular")
        self.xc = float(xc)
        self.yc = float(yc)
        self.R = float(max(0.1, R))

    def get_x_range(self, gx: np.ndarray, gy: np.ndarray) -> Optional[Tuple[float, float]]:
        x_min = max(gx[0], self.xc - self.R + 1e-4)
        x_max = min(gx[-1], self.xc + self.R - 1e-4)
        if x_min >= x_max:
            return None
        xs = np.linspace(x_min, x_max, 1000)
        yc_circle = self.yc - np.sqrt(np.maximum(0.0, self.R**2 - (xs - self.xc)**2))
        yg = np.interp(xs, gx, gy)
        inside = np.where(yc_circle - yg < 0)[0]
        if len(inside) < 2:
            return None
        x_s, x_e = float(xs[inside[0]]), float(xs[inside[-1]])
        return (x_s, x_e) if (x_e - x_s > 0.1) else None

    def get_y_base(self, x: float) -> float:
        val = max(0.0, self.R**2 - (x - self.xc)**2)
        return float(self.yc - np.sqrt(val))

    def get_alpha(self, x: float) -> float:
        sin_alpha = np.clip((self.xc - x) / self.R, -0.9999, 0.9999)
        return float(np.arcsin(sin_alpha))

    def get_moment_center(self) -> Tuple[float, float]:
        return self.xc, self.yc


class PolygonalSlipSurface(BaseSlipSurface):
    """任意非圆弧折线滑面 (适用于顺层、节理、阶梯状滑动)"""
    def __init__(self, points: List[Tuple[float, float]]):
        super().__init__("polygonal")
        # 按 X 递增排序
        self.points = sorted(points, key=lambda p: p[0])
        self.px = np.array([p[0] for p in self.points], dtype=float)
        self.py = np.array([p[1] for p in self.points], dtype=float)

    def get_x_range(self, gx: np.ndarray, gy: np.ndarray) -> Optional[Tuple[float, float]]:
        x_min = max(gx[0], self.px[0])
        x_max = min(gx[-1], self.px[-1])
        if x_min >= x_max:
            return None
        xs = np.linspace(x_min, x_max, 1000)
        yg = np.interp(xs, gx, gy)
        yp = np.interp(xs, self.px, self.py)
        inside = np.where(yp - yg < -1e-6)[0]   # 加一个容差
        if len(inside) < 2:
            return None
        x_s, x_e = float(xs[inside[0]]), float(xs[inside[-1]])
        return (x_s, x_e) if (x_e - x_s > 0.1) else None

    def get_y_base(self, x: float) -> float:
        x = float(np.clip(x, self.px[0], self.px[-1]))
        return float(np.interp(x, self.px, self.py))

    def get_alpha(self, x: float) -> float:
        idx = max(0, min(np.searchsorted(self.px, x) - 1, len(self.px) - 2))
        dx = self.px[idx + 1] - self.px[idx]
        dy = self.py[idx + 1] - self.py[idx]
        # 向下倾斜 dy < 0 对应正下滑坡角 alpha > 0
        return float(np.arctan2(-dy, max(1e-5, dx)))

    def get_moment_center(self) -> Tuple[float, float]:
        # 严密法对任意参考中心均满足平衡，取几何中心点
        return float(np.mean(self.px)), float(np.mean(self.py))
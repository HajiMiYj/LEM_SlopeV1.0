# -*- coding: utf-8 -*-
"""
极限平衡求解器抽象基类与接口规范
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from core.slip_surface import BaseSlipSurface
from core.slicing import Slice
from core.geometry import SlopeGeometry


class BaseLEMSolver(ABC):
    supports_non_circular: bool = False  # 是否支持非圆弧滑面

    def __init__(
        self,
        slices: List[Slice],
        xc: float = 0.0,
        yc: float = 0.0,
        R: float = 0.0,
        geom: SlopeGeometry = None,
        x_edges: np.ndarray = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        slip_surface: Optional[BaseSlipSurface] = None
    ):
        self.slices = slices or []
        self.xc = xc
        self.yc = yc
        self.R = R
        self.geom = geom
        self.tol = tol
        self.max_iter = max_iter
        self.slip_surface = slip_surface

        # ============================================================
        # 【P1-1 修改】x_edges 兜底
        # ------------------------------------------------------------
        # 若调用方未提供 x_edges，则从 slices 自动派生出：
        #     [第一条左边 x, 最后一条右边 x]
        # 这样 Janbu / Spencer / M-P 里访问 self.x_edges[0] / [-1]
        # 就不会抛 IndexError 或 NoneType 错误。
        # ============================================================
        if x_edges is not None:
            self.x_edges = np.asarray(x_edges, dtype=float)
        elif self.slices:
            xl_first = float(self.slices[0].xm - 0.5 * self.slices[0].b)
            xr_last = float(self.slices[-1].xm + 0.5 * self.slices[-1].b)
            self.x_edges = np.array([xl_first, xr_last], dtype=float)
        else:
            # 极端情况：没有任何切片，给一个安全的占位数组，避免 None[0] 崩
            self.x_edges = np.array([0.0, 0.0], dtype=float)
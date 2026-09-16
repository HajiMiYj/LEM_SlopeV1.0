# -*- coding: utf-8 -*-
"""
简化毕肖普法 (Simplified Bishop Method)
支持非饱和表观黏聚力、外荷载与拟静力地震荷载
"""
import numpy as np
from typing import Tuple, Optional
from core.solvers.base import BaseLEMSolver


class BishopSolver(BaseLEMSolver):
    supports_non_circular = False
    def solve(self):
        if self.slip_surface and self.slip_surface.surface_type != "circular":
            return None, "该算法仅限圆弧滑面"
        driving = sum(
            s.W * np.sin(s.alpha) + s.Fh * np.cos(s.alpha)
            for s in self.slices
        )
        if driving <= 0:
            return None, "下滑滑动力矩 <= 0"

        fs = 1.2
        for it in range(self.max_iter):
            num = 0.0
            for s in self.slices:
                m_alpha = np.cos(s.alpha) * (1.0 + np.tan(s.alpha) * np.tan(s.phi) / fs)
                if abs(m_alpha) < 1e-5:
                    m_alpha = 1e-5
                num += (s.c * s.b + (s.W - s.u * s.b - s.Fh * np.tan(s.alpha)) * np.tan(s.phi)) / m_alpha

            new_fs = num / driving
            if abs(new_fs - fs) < self.tol:
                return new_fs, f"定点迭代收敛于第 {it + 1} 步"
            fs = new_fs

        return fs, f"达到设定最大迭代步数 ({self.max_iter})"
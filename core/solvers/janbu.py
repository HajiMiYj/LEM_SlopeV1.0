# -*- coding: utf-8 -*-
"""
简化让布法 (Simplified Janbu Method)
支持非饱和表观黏聚力、外荷载与拟静力地震荷载
"""
import numpy as np
from typing import Tuple, Optional
from core.solvers.base import BaseLEMSolver


class JanbuSolver(BaseLEMSolver):
    supports_non_circular = True
#    原生支持非圆弧与圆弧
    def solve(self) -> Tuple[Optional[float], str]:
        driving = sum(s.W * np.tan(s.alpha) + s.Fh for s in self.slices)
        if driving <= 0:
            return None, "水平推力基准驱动项 <= 0"

        fs0 = 1.2
        for it in range(self.max_iter):
            num = 0.0
            for s in self.slices:
                n_alpha = (np.cos(s.alpha) ** 2) * (1.0 + np.tan(s.alpha) * np.tan(s.phi) / fs0)
                if abs(n_alpha) < 1e-5:
                    n_alpha = 1e-5
                num += (s.c * s.b + (s.W - s.u * s.b - s.Fh * np.tan(s.alpha)) * np.tan(s.phi)) / n_alpha

            new_fs0 = num / driving
            if abs(new_fs0 - fs0) < self.tol:
                fs0 = new_fs0
                break
            fs0 = new_fs0

        x_start, x_end = self.x_edges[0], self.x_edges[-1]
        y_start = self.geom.get_ground_elevation(x_start)
        y_end = self.geom.get_ground_elevation(x_end)
        chord_m = (y_end - y_start) / (x_end - x_start)
        chord_c = y_start - chord_m * x_start

        depths = [chord_m * s.xm + chord_c - s.y_base for s in self.slices]
        max_d = max(0.0, max(depths))
        chord_length = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
        ratio = max_d / max(1e-4, chord_length)

        b1 = 0.69
        ratio_eff = min(ratio, 0.5)
        f0 = 1.0 + b1 * (ratio_eff - 1.4 * (ratio_eff ** 2))
        f0 = max(0.90, min(1.20, f0))
        fs = f0 * fs0
        return fs, f"收敛 (未修正 F0={fs0:.4f}, 经验修正系数 f0={f0:.3f})"
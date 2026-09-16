# -*- coding: utf-8 -*-
"""
瑞典条分法 (Fellenius / Ordinary / Swedish Circle Method)
支持非饱和表观黏聚力、外荷载与拟静力地震荷载
"""
import numpy as np
from typing import Tuple, Optional
from core.solvers.base import BaseLEMSolver


class FelleniusSolver(BaseLEMSolver):
    supports_non_circular = False
    def solve(self) -> Tuple[Optional[float], str]:
        if self.slip_surface and self.slip_surface.surface_type != "circular":
            return None, "该算法仅限圆弧滑面"
        driving = sum(
            s.W * np.sin(s.alpha) + s.Fh * np.cos(s.alpha)
            for s in self.slices
        )
        if driving <= 0:
            return None, "下滑滑动力矩 <= 0（边坡自然稳定）"

        resisting = 0.0
        for s in self.slices:
            N_eff = s.W * np.cos(s.alpha) - s.Fh * np.sin(s.alpha) - s.u * s.l
            resisting += s.c * s.l + max(0.0, N_eff) * np.tan(s.phi)

        fs = resisting / driving
        return fs, "单步显式解析解（含非饱和吸力与抗震）"
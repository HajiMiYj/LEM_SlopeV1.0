# -*- coding: utf-8 -*-
"""
蒙特卡洛模拟法 (MCS) 边坡失效概率与可靠度指标求解器

支持:
  · 任意滑面 (圆弧 CircularSlipSurface / 非圆弧 PolygonalSlipSurface)
  · 任意 LEM 求解器 (Fellenius / Bishop / Janbu / Spencer / M-P)
"""
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from scipy.stats import norm

from core.geometry import SlopeGeometry
from core.materials import SoilMaterial
from core.slicing import create_slices
from core.slip_surface import BaseSlipSurface, CircularSlipSurface


class MonteCarloSimulator:
    def __init__(
        self,
        geom: SlopeGeometry,
        materials: List[SoilMaterial],
        slip_surface: BaseSlipSurface,         # ★ 改为滑面对象
        n_samples: int = 5000,
        solver_class=None,                     # ★ 改为求解器类
        rainfall_depth: float = 0.0,
        kh: float = 0.0,
        n_slices: int = 25,
        cov_c: float = 0.25,
        dist_c: str = "对数正态分布",
        cov_phi: float = 0.15,
        dist_phi: str = "对数正态分布",
        rho_c_phi: float = -0.50,
        cov_gamma: float = 0.08,
    ):
        self.geom = geom
        self.materials = materials
        self.slip_surface = slip_surface
        self.n_samples = max(100, int(n_samples))
        self.rainfall_depth = rainfall_depth
        self.kh = kh
        self.n_slices = n_slices
        self.cov_c = cov_c
        self.dist_c = dist_c
        self.cov_phi = cov_phi
        self.dist_phi = dist_phi
        self.rho_c_phi = rho_c_phi
        self.cov_gamma = cov_gamma

        # 默认求解器: Bishop
        if solver_class is None:
            from core.solvers.bishop import BishopSolver
            solver_class = BishopSolver
        self.solver_class = solver_class

        # 从滑面对象派生 xc / yc / R (供需要力矩中心的求解器使用)
        self.xc, self.yc, self.R = self._derive_center_and_R()

    def _derive_center_and_R(self):
        """从滑面对象派生力矩中心与等效半径"""
        ss = self.slip_surface
        if getattr(ss, "surface_type", "") == "circular":
            return float(ss.xc), float(ss.yc), float(ss.R)
        # 非圆弧: 用几何中心 + 占位 R (run 时按切片重算)
        try:
            cx, cy = ss.get_moment_center()
        except Exception:
            cx, cy = float(np.mean(ss.px)), float(np.mean(ss.py))
        return float(cx), float(cy), 1.0

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        is_interrupted_fn: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        # 1. 基准切片
        base_slices, info, err_msg = create_slices(
            geom=self.geom,
            materials=self.materials,
            slip_surface=self.slip_surface,
            n_slices=self.n_slices,
            rainfall_depth=self.rainfall_depth,
            kh=self.kh,
        )
        if not base_slices or not info:
            return {"success": False, "error": err_msg or "滑面未切出有效滑体"}

        x_edges = info[2]

        # 2. 非圆弧滑面: 按实际切片重算等效半径 R
        if getattr(self.slip_surface, "surface_type", "") != "circular":
            self.R = max(
                float(np.hypot(s.xm - self.xc, s.y_base - self.yc))
                for s in base_slices
            )
            self.R = max(1.0, self.R)

        # 3. 各土层独立抽样
        from core.reliability.sampler import sample_soil_parameters
        layer_samples = []
        for mat in self.materials:
            s_dict = sample_soil_parameters(
                n_samples=self.n_samples,
                mean_c=mat.c_prime,
                cov_c=self.cov_c,
                dist_c=self.dist_c,
                mean_phi=mat.phi_deg,
                cov_phi=self.cov_phi,
                dist_phi=self.dist_phi,
                rho_c_phi=self.rho_c_phi,
                mean_gamma=mat.gamma_dry,
                cov_gamma=self.cov_gamma,
            )
            layer_samples.append(s_dict)

        # 4. 循环抽样
        fs_records = []
        failure_count = 0
        batch_report = max(50, self.n_samples // 50)

        for i in range(self.n_samples):
            if is_interrupted_fn and is_interrupted_fn():
                return {"success": False, "error": "计算被用户中断"}

            # 为每个土条赋随机参数
            for s in base_slices:
                layer_idx = 0
                for m_idx, m in enumerate(self.materials):
                    if m.name == s.layer_name:
                        layer_idx = m_idx
                        break
                samples = layer_samples[layer_idx]
                s.c = float(samples["c"][i])
                s.phi = float(np.radians(samples["phi_deg"][i]))
                weight_ratio = float(
                    samples["gamma"][i] / max(1.0, self.materials[layer_idx].gamma_dry)
                )
                s.W = s.W_soil * weight_ratio + s.q_load
                s.Fh = s.kh * (s.W_soil * weight_ratio)

            # 用用户选择的求解器
            try:
                solver = self.solver_class(
                    base_slices,
                    self.xc, self.yc, self.R,
                    self.geom,
                    x_edges,
                    slip_surface=self.slip_surface,
                )
                fs, _ = solver.solve()
            except Exception:
                fs = None

            if fs is not None and fs > 0.01:
                fs_records.append(fs)
                if fs < 1.0:
                    failure_count += 1

            if progress_callback and (i + 1) % batch_report == 0:
                cur_pf = (failure_count / max(1, len(fs_records))) * 100.0
                progress_callback(i + 1, self.n_samples, cur_pf)

        if len(fs_records) < 10:
            return {"success": False, "error": "有效收敛样本数过低"}

        # 5. 统计
        fs_arr = np.array(fs_records, dtype=float)
        n_valid = len(fs_arr)
        pf = failure_count / n_valid
        pf_pct = pf * 100.0

        mean_fs = float(np.mean(fs_arr))
        std_fs = float(np.std(fs_arr))
        cov_fs = float(std_fs / max(1e-4, mean_fs))

        if pf <= 0.0:
            beta = float((mean_fs - 1.0) / max(1e-4, std_fs))
        elif pf >= 1.0:
            beta = -3.0
        else:
            beta = float(-norm.ppf(pf))

        counts, bin_edges = np.histogram(fs_arr, bins=25)

        return {
            "success": True,
            "n_samples": self.n_samples,
            "n_valid": n_valid,
            "failure_count": failure_count,
            "pf": float(pf),
            "pf_percent": float(pf_pct),
            "beta": float(beta),
            "fs_mean": mean_fs,
            "fs_std": std_fs,
            "fs_cov": cov_fs,
            "fs_min": float(np.min(fs_arr)),
            "fs_max": float(np.max(fs_arr)),
            "hist_counts": counts.tolist(),
            "hist_bins": bin_edges.tolist(),
        }
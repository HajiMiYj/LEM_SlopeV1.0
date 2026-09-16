# -*- coding: utf-8 -*-
"""
Rosenblueth 点估计法 (Point Estimate Method, PEM) 边坡可靠度快速核算

支持任意滑面 + 任意求解器
"""
from typing import List, Dict, Any
import numpy as np
from scipy.stats import norm

from core.geometry import SlopeGeometry
from core.materials import SoilMaterial
from core.slicing import create_slices
from core.slip_surface import BaseSlipSurface


def run_point_estimate_analysis(
    geom: SlopeGeometry,
    materials: List[SoilMaterial],
    slip_surface: BaseSlipSurface,           # ★ 改为滑面对象
    solver_class=None,                       # ★ 新增求解器类
    cov_c: float = 0.25,
    cov_phi: float = 0.15,
    rho_c_phi: float = -0.50,
    rainfall_depth: float = 0.0,
    kh: float = 0.0,
) -> Dict[str, Any]:
    """通过 2^2=4 个特征点评估 FS 均值与方差"""
    if solver_class is None:
        from core.solvers.bishop import BishopSolver
        solver_class = BishopSolver

    mat = materials[0]
    mu_c = mat.c_prime
    sigma_c = mu_c * max(0.01, cov_c)
    mu_phi = mat.phi_deg
    sigma_phi = mu_phi * max(0.01, cov_phi)

    pts = [
        (mu_c + sigma_c, mu_phi + sigma_phi, (1.0 + rho_c_phi) / 4.0),
        (mu_c + sigma_c, mu_phi - sigma_phi, (1.0 - rho_c_phi) / 4.0),
        (mu_c - sigma_c, mu_phi + sigma_phi, (1.0 - rho_c_phi) / 4.0),
        (mu_c - sigma_c, mu_phi - sigma_phi, (1.0 + rho_c_phi) / 4.0),
    ]

    base_slices, info, err_msg = create_slices(
        geom=geom,
        materials=materials,
        slip_surface=slip_surface,
        n_slices=25,
        rainfall_depth=rainfall_depth,
        kh=kh,
    )
    if not base_slices or not info:
        return {"success": False, "error": err_msg or "几何剖分失败"}

    x_edges = info[2]

    # 派生力矩中心与 R
    if getattr(slip_surface, "surface_type", "") == "circular":
        xc_, yc_, R_ = slip_surface.xc, slip_surface.yc, slip_surface.R
    else:
        try:
            xc_, yc_ = slip_surface.get_moment_center()
        except Exception:
            xc_ = float(np.mean(slip_surface.px))
            yc_ = float(np.mean(slip_surface.py))
        R_ = max(
            float(np.hypot(s.xm - xc_, s.y_base - yc_))
            for s in base_slices
        )
        R_ = max(1.0, R_)

    e_fs = 0.0
    e_fs2 = 0.0

    for c_val, phi_val, weight in pts:
        c_val = max(0.1, c_val)
        phi_rad = np.radians(np.clip(phi_val, 1.0, 50.0))
        for s in base_slices:
            s.c = c_val
            s.phi = phi_rad

        try:
            solver = solver_class(
                base_slices,
                xc_, yc_, R_,
                geom, x_edges,
                slip_surface=slip_surface,
            )
            fs, _ = solver.solve()
        except Exception:
            fs = None

        fs_val = fs if fs is not None else 1.0
        e_fs += weight * fs_val
        e_fs2 += weight * (fs_val ** 2)

    var_fs = max(1e-5, e_fs2 - (e_fs ** 2))
    std_fs = float(np.sqrt(var_fs))
    beta = float((e_fs - 1.0) / std_fs)
    pf = float(norm.cdf(-beta))

    return {
        "success": True,
        "method": "Rosenblueth 点估计法",
        "fs_mean": float(e_fs),
        "fs_std": std_fs,
        "fs_cov": float(std_fs / max(1e-4, e_fs)),
        "beta": float(beta),
        "pf": pf,
        "pf_percent": float(pf * 100.0),
    }
# -*- coding: utf-8 -*-
"""
土条离散剖分器模块
支持多层地层自重积分、降雨入渗湿润锋、非饱和基质吸力表观黏聚力、坡顶超载与地震拟静力荷载
"""
import numpy as np
from typing import List, Tuple, Optional
from core.geometry import SlopeGeometry
from core.materials import SoilMaterial
from core.slip_surface import BaseSlipSurface, CircularSlipSurface


class Slice:
    """单个竖直切片土条的综合力学与几何微元对象"""
    def __init__(self, index, xm, b, h, y_top, y_base, alpha, l,
                 W, q_load, kh, u, suction, c_total, phi, layer_name):
        self.index = index
        self.xm = xm
        self.b = b
        self.h = h
        self.y_top = y_top
        self.y_base = y_base
        self.alpha = alpha
        self.l = l
        self.W_soil = W
        self.q_load = q_load
        self.W = W + q_load
        self.kh = kh
        self.Fh = kh * W
        self.u = u
        self.suction = suction
        self.c = c_total
        self.phi = phi
        self.layer_name = layer_name


def create_slices(
    geom: SlopeGeometry,
    materials: List[SoilMaterial],
    xc=None,
    yc: Optional[float] = None,
    R: Optional[float] = None,
    n_slices: int = 30,
    rainfall_depth: float = 0.0,
    kh: float = 0.0,
    slip_surface: Optional[BaseSlipSurface] = None,
    tension_crack=None,
    anchors=None,
    **kwargs,
) -> Tuple[Optional[List[Slice]], Optional[Tuple[float, float, np.ndarray]], str]:
    """通用边坡切片剖分器（同时支持圆弧与非圆弧折线滑面）"""
    if n_slices < 1:
        return None, None, "切片数必须 ≥ 1"

    # 1. 统一提取滑面对象
    if isinstance(xc, BaseSlipSurface) and slip_surface is None:
        slip_surface = xc
        xc = None

    if slip_surface is None:
        if xc is None or yc is None or R is None:
            return None, None, "未指定滑面参数（需提供 slip_surface 或 xc/yc/R）"
        slip_surface = CircularSlipSurface(float(xc), float(yc), float(R))

    # 2. 滑面与地表求交
    inter = slip_surface.get_x_range(geom.gx, geom.gy)
    if not inter:
        return None, None, "滑面未在边坡内部切出有效滑动体"

    x_start, x_end = inter
    if x_end - x_start < 1e-3:
        return None, None, f"滑面水平跨度太小: {x_end - x_start:.4f} m"

    x_edges = np.linspace(x_start, x_end, n_slices + 1)
    slices: List[Slice] = []
    MIN_THICKNESS = 1e-3

    for i in range(n_slices):
        xl, xr = float(x_edges[i]), float(x_edges[i + 1])
        xm = 0.5 * (xl + xr)
        b = xr - xl
        if b <= 0:
            return None, None, f"第 {i+1} 条宽度非法: b={b}"

        y_top = geom.get_ground_elevation(xm)
        y_base = slip_surface.get_y_base(xm)

        if y_base >= y_top - 1e-6:
            return None, None, (
                f"滑面穿出地面: 第{i+1}条 xm={xm:.3f}, "
                f"y_base={y_base:.4f} >= y_top={y_top:.4f}"
            )

        h = max(MIN_THICKNESS, y_top - y_base)
        alpha = float(slip_surface.get_alpha(xm))
        cos_a = np.cos(alpha)
        if abs(cos_a) < 0.1:
            cos_a = 0.1 if cos_a >= 0 else -0.1
        l = b / cos_a

        # 3. 水位与湿润锋
        yw = geom.get_water_elevation(xm)
        wetting_front_y = y_top - max(0.0, float(rainfall_depth))

        # 4. 多层地层自重积分 (★ 关键: 用 get_layer_breakpoints 找分层)
        strata_y = geom.get_layer_breakpoints(xm, y_base, y_top)
        div_points = [y_top]
        for sy in strata_y:
            if y_base + 1e-9 < sy < y_top - 1e-9:
                div_points.append(float(sy))
        div_points.append(y_base)
        div_points = sorted(div_points, reverse=True)

        W_soil = 0.0
        for seg_idx in range(len(div_points) - 1):
            y_high, y_low = div_points[seg_idx], div_points[seg_idx + 1]
            seg_h = y_high - y_low
            if seg_h <= 1e-9:
                continue
            seg_mid_y = 0.5 * (y_high + y_low)

            layer_idx = min(geom.get_layer_index_at(xm, seg_mid_y),
                            len(materials) - 1)
            mat = materials[layer_idx]

            is_saturated = (
                (yw is not None and seg_mid_y <= yw)
                or (rainfall_depth > 0.0 and seg_mid_y >= wetting_front_y)
            )
            use_gamma = mat.gamma_sat if is_saturated else mat.gamma_dry
            W_soil += use_gamma * b * seg_h

        # 5. 外载与孔隙水压力
        q_load = geom.get_surcharge_at(xm) * b

        if yw is not None:
            u = 9.81 * (yw - y_base) if y_base <= yw else 0.0
            suction = 9.81 * (y_base - yw) if y_base > yw else 0.0
        else:
            u, suction = 0.0, 0.0

        if rainfall_depth > 0.0 and y_base >= wetting_front_y:
            suction = 0.0

        # 6. 条底力学参数
        base_layer_idx = min(geom.get_layer_index_at(xm, y_base),
                             len(materials) - 1)
        base_mat = materials[base_layer_idx]

        s_obj = Slice(
            index=i + 1,
            xm=xm, b=b, h=h, y_top=y_top, y_base=y_base,
            alpha=alpha, l=l,
            W=W_soil, q_load=q_load, kh=kh,
            u=u, suction=suction,
            c_total=base_mat.get_apparent_cohesion(suction),
            phi=base_mat.phi,
            layer_name=base_mat.name,
        )
        s_obj.y_cg = 0.5 * (y_top + y_base)
        slices.append(s_obj)

    if not slices:
        return None, None, "切片列表为空"

    return slices, (x_start, x_end, x_edges), "切片剖分成功"
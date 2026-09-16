# -*- coding: utf-8 -*-
"""
边坡几何外轮廓、任意起伏多层地层分界面、地下水位线与外荷载拓扑模块
"""
import numpy as np
from typing import List, Tuple, Optional


class SlopeGeometry:
    """边坡几何多段线与多层地层拓扑管理器

    数据模型:
      · gx/gy          —— 地表轮廓线
      · wx/wy          —— 地下水位线 (可选)
      · layer_regions  —— 土层面 (闭合多边形列表, 唯一的"层"数据)
      · surcharge_loads —— 坡顶荷载
    """
    def __init__(
        self,
        ground_coords: List[Tuple[float, float]],
        water_coords: Optional[List[Tuple[float, float]]] = None,
        layer_regions: Optional[List[dict]] = None,
        surcharge_loads: Optional[List[Tuple[float, float, float]]] = None,
    ):
        sorted_ground = sorted(ground_coords, key=lambda p: p[0])
        self.gx = np.array([p[0] for p in sorted_ground], dtype=float)
        self.gy = np.array([p[1] for p in sorted_ground], dtype=float)

        if water_coords and len(water_coords) >= 2:
            sorted_water = sorted(water_coords, key=lambda p: p[0])
            self.wx = np.array([p[0] for p in sorted_water], dtype=float)
            self.wy = np.array([p[1] for p in sorted_water], dtype=float)
        else:
            self.wx, self.wy = None, None

        self.layer_regions = layer_regions if layer_regions else []
        self.surcharge_loads = surcharge_loads if surcharge_loads else []

    # ------------------------------------------------------------------
    # 地表 / 水位
    # ------------------------------------------------------------------
    def get_ground_elevation(self, x: float) -> float:
        return float(np.interp(x, self.gx, self.gy))

    def get_water_elevation(self, x: float) -> Optional[float]:
        if self.wx is not None:
            return float(np.interp(x, self.wx, self.wy))
        return None

    # ------------------------------------------------------------------
    # 土层判定 (基于面, 后画覆盖先画)
    # ------------------------------------------------------------------
    def get_layer_index_at(self, x: float, y: float) -> int:
        """从后往前遍历 regions, 返回第一个"包含 (x,y) 且不在其洞内"的面"""
        for region in reversed(self.layer_regions):
            points = region.get("points", []) if isinstance(region, dict) else []
            holes = region.get("holes", []) if isinstance(region, dict) else []
            if self._point_in_region(x, y, points, holes):
                return max(0, int(region.get("material_index", 0)))
        return 0

    def _point_in_region(self, x, y, outer, holes):
        if not self._point_in_polygon(x, y, outer):
            return False
        for h in holes:
            if len(h) >= 3 and self._point_in_polygon(x, y, h):
                return False
        return True
    @staticmethod
    def _point_in_polygon(x: float, y: float, points: list) -> bool:
        """射线法判定点是否在多边形内"""
        if len(points) < 3:
            return False
        inside = False
        px, py = points[-1]
        for cx, cy in points:
            if (cy > y) != (py > y):
                x_cross = (px - cx) * (y - cy) / (py - cy) + cx
                if x < x_cross:
                    inside = not inside
            px, py = cx, cy
        return inside

    def get_layer_breakpoints(self, x: float, y_base: float, y_top: float,
                              n_samples: int = 200) -> List[float]:
        """在垂直段 [y_base, y_top] 上找材料变化的 y 坐标 (自上而下)。

        slicing.py 用它来把一条土条沿深度分成若干层。
        """
        if y_top <= y_base or n_samples < 2:
            return []

        ys = np.linspace(y_base, y_top, n_samples)
        layers = [self.get_layer_index_at(x, yy) for yy in ys]

        breaks = []
        for i in range(len(layers) - 1):
            if layers[i] != layers[i + 1]:
                breaks.append(0.5 * (ys[i] + ys[i + 1]))
        return breaks

    # ------------------------------------------------------------------
    # 外荷载
    # ------------------------------------------------------------------
    def get_surcharge_at(self, x: float) -> float:
        total_q = 0.0
        for x1, x2, q in self.surcharge_loads:
            if min(x1, x2) <= x <= max(x1, x2):
                total_q += q
        return total_q

    # ------------------------------------------------------------------
    # 兼容旧接口: 圆弧与地表交点
    # ------------------------------------------------------------------
    def intersect_circle(self, xc: float, yc: float, R: float):
        xs = np.linspace(xc - R + 1e-5, xc + R - 1e-5, 2000)
        yc_circle = yc - np.sqrt(np.maximum(0, R**2 - (xs - xc)**2))
        yg = np.interp(xs, self.gx, self.gy)
        diff = yc_circle - yg
        inside = np.where(diff < 0)[0]

        if len(inside) < 2:
            return None
        x_start = float(xs[inside[0]])
        x_end = float(xs[inside[-1]])
        if x_end - x_start < 0.1:
            return None
        return x_start, x_end
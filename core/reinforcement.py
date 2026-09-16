# -*- coding: utf-8 -*-
"""
边坡支护构件数据模型

支持:
  · Anchor    —— 锚索 / 锚杆 / 土钉
  · PileRow   —— 抗滑桩（按排）

设计约定:
  · 所有构件对外暴露 get_resisting_force(slip_surface, xm) → kN/m
  · 返回值单位是"沿滑面切向、折合到单位宽度"的抗滑力
  · 求解器把该力加到抗滑项即可
"""
from typing import Optional
import numpy as np


class ReinforcementBase:
    """支护构件基类"""
    kind = "base"

    def __init__(self, elem_id: int):
        self.id = elem_id
        self.enabled = True

    def get_resisting_force(self, slip_surface, xm: float) -> float:
        raise NotImplementedError

    def affects_slice(self, xm: float) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


# ======================================================================
# 锚索 / 锚杆
# ======================================================================
class Anchor(ReinforcementBase):
    """锚索 / 锚杆 / 土钉"""
    kind = "anchor"

    def __init__(self, elem_id: int, x_head: float, y_head: float,
                 angle_deg: float, length: float,
                 prestress_kN: float, spacing_m: float,
                 grout_bond_kN: Optional[float] = None,
                 safety_factor: float = 0.8):
        """
        x_head, y_head: 锚头坐标
        angle_deg: 与水平夹角（正=向下倾斜）
        length: 锚索总长
        prestress_kN: 预应力
        spacing_m: 沿坡面方向的水平间距
        grout_bond_kN: 锚固段极限抗拔力（可选，None 时忽略）
        safety_factor: 长期松弛/安全折减（默认 0.8）
        """
        super().__init__(elem_id)
        self.x_head = float(x_head)
        self.y_head = float(y_head)
        self.angle_deg = float(angle_deg)
        self.length = float(length)
        self.prestress = float(prestress_kN)
        self.spacing = max(0.1, float(spacing_m))
        self.grout_bond = float(grout_bond_kN) if grout_bond_kN is not None else None
        self.safety_factor = float(safety_factor)

    def get_tip_xy(self):
        rad = np.radians(self.angle_deg)
        x_tip = self.x_head + self.length * np.cos(rad)
        y_tip = self.y_head - self.length * np.sin(rad)
        return x_tip, y_tip

    def affects_slice(self, xm: float) -> bool:
        x_tip, _ = self.get_tip_xy()
        return min(self.x_head, x_tip) <= xm <= max(self.x_head, x_tip)

    def get_resisting_force(self, slip_surface, xm: float) -> float:
        if not self.enabled or not self.affects_slice(xm):
            return 0.0

        # 有效拉力
        T = self.prestress
        if self.grout_bond is not None:
            T = min(T, self.grout_bond)
        T *= self.safety_factor

        # 滑面倾角
        try:
            alpha = slip_surface.get_alpha(xm)
        except Exception:
            return 0.0

        theta = np.radians(self.angle_deg)
        # 锚索力在滑面切向的分量
        F_tangential = T * np.cos(theta - alpha)
        # 按间距折合到单位宽度
        return F_tangential / self.spacing

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": "anchor", "enabled": self.enabled,
            "x_head": self.x_head, "y_head": self.y_head,
            "angle_deg": self.angle_deg, "length": self.length,
            "prestress": self.prestress, "spacing": self.spacing,
            "grout_bond": self.grout_bond,
            "safety_factor": self.safety_factor,
        }

    @classmethod
    def from_dict(cls, data: dict):
        a = cls(
            elem_id=int(data.get("id", 1)),
            x_head=float(data["x_head"]), y_head=float(data["y_head"]),
            angle_deg=float(data["angle_deg"]), length=float(data["length"]),
            prestress_kN=float(data["prestress"]),
            spacing_m=float(data["spacing"]),
            grout_bond_kN=data.get("grout_bond"),
            safety_factor=float(data.get("safety_factor", 0.8)),
        )
        a.enabled = bool(data.get("enabled", True))
        return a


# ======================================================================
# 抗滑桩（按排）
# ======================================================================
class PileRow(ReinforcementBase):
    """抗滑桩（按排，等间距）"""
    kind = "pile"

    def __init__(self, elem_id: int, x: float,
                 top_elev: float, bottom_elev: float,
                 diameter: float, spacing: float,
                 cohesion_kPa: float = 50.0,
                 safety_factor: float = 0.7):
        """
        x: 桩排中心 X
        top_elev / bottom_elev: 桩顶/桩底高程
        diameter: 桩径
        spacing: 桩中心距
        cohesion_kPa: 桩周土体不排水黏聚力（用于 Broms 估算）
        safety_factor: 抗力折减（默认 0.7）
        """
        super().__init__(elem_id)
        self.x = float(x)
        self.top = float(top_elev)
        self.bottom = float(bottom_elev)
        self.diameter = float(diameter)
        self.spacing = max(0.1, float(spacing))
        self.cohesion = float(cohesion_kPa)
        self.safety_factor = float(safety_factor)

    def affects_slice(self, xm: float) -> bool:
        return abs(xm - self.x) <= 0.5 * self.spacing

    def _calc_pile_resistance(self) -> float:
        """Broms 简化法估算单桩极限抗力 (kN)"""
        L = abs(self.top - self.bottom)
        d = self.diameter
        if L <= 1.5 * d:
            return 0.0
        # 黏性土: F = 9 * c_u * d * (L - 1.5d)
        F = 9.0 * self.cohesion * d * (L - 1.5 * d)
        return F * self.safety_factor

    def get_resisting_force(self, slip_surface, xm: float) -> float:
        if not self.enabled or not self.affects_slice(xm):
            return 0.0
        F_pile = self._calc_pile_resistance()
        return F_pile / self.spacing

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": "pile", "enabled": self.enabled,
            "x": self.x, "top": self.top, "bottom": self.bottom,
            "diameter": self.diameter, "spacing": self.spacing,
            "cohesion": self.cohesion,
            "safety_factor": self.safety_factor,
        }

    @classmethod
    def from_dict(cls, data: dict):
        p = cls(
            elem_id=int(data.get("id", 1)),
            x=float(data["x"]),
            top_elev=float(data["top"]), bottom_elev=float(data["bottom"]),
            diameter=float(data["diameter"]),
            spacing=float(data["spacing"]),
            cohesion_kPa=float(data.get("cohesion", 50.0)),
            safety_factor=float(data.get("safety_factor", 0.7)),
        )
        p.enabled = bool(data.get("enabled", True))
        return p


# ======================================================================
# 挡土墙（重力式）
# ======================================================================
class Wall(ReinforcementBase):
    """重力式挡土墙 (坡脚支挡)"""
    kind = "wall"

    def __init__(self, elem_id: int, x: float,
                 top_elev: float, bottom_elev: float,
                 top_width: float = 0.8, bottom_width: float = 1.5,
                 concrete_unit_weight: float = 23.0,
                 base_friction_coef: float = 0.5,
                 passive_phi_deg: float = 30.0,
                 passive_gamma: float = 18.0,
                 safety_factor: float = 0.8):
        """
        x: 墙背(靠坡一侧) X 坐标
        top_elev / bottom_elev: 墙顶/墙底标高
        top_width / bottom_width: 墙顶/墙底宽度
        concrete_unit_weight: 墙身重度 (kN/m³)
        base_friction_coef: 基底摩擦系数
        passive_phi_deg: 墙前填土内摩擦角 (用于被动土压力)
        passive_gamma: 墙前填土重度 (kN/m³)
        safety_factor: 抗力折减 (默认 0.8)
        """
        super().__init__(elem_id)
        self.x = float(x)
        self.top = float(top_elev)
        self.bottom = float(bottom_elev)
        self.top_width = float(top_width)
        self.bottom_width = float(bottom_width)
        self.gamma_concrete = float(concrete_unit_weight)
        self.base_friction = float(base_friction_coef)
        self.passive_phi = float(passive_phi_deg)
        self.passive_gamma = float(passive_gamma)
        self.safety_factor = float(safety_factor)

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)

    def affects_slice(self, xm: float) -> bool:
        # 墙是连续的, 但只在墙位附近(±1.5m)生效, 避免影响太远
        return abs(xm - self.x) <= 1.5

    def _calc_resistance(self) -> float:
        """计算单位墙长可提供的水平抗滑力 (kN/m)"""
        H = self.height
        if H < 0.5:
            return 0.0

        # 1. 墙身自重抗力 (基底摩擦)
        B_avg = 0.5 * (self.top_width + self.bottom_width)
        W_wall = B_avg * H * self.gamma_concrete        # kN/m
        R_base = W_wall * self.base_friction

        # 2. 墙前被动土压力
        Kp = np.tan(np.radians(45.0 + self.passive_phi / 2.0)) ** 2
        Ep = 0.5 * Kp * self.passive_gamma * H * H      # kN/m

        # 取两者较小值作为可用抗力
        R_avail = min(R_base, Ep) * self.safety_factor
        return R_avail

    def get_resisting_force(self, slip_surface, xm: float) -> float:
        if not self.enabled or not self.affects_slice(xm):
            return 0.0

        R_horizontal = self._calc_resistance()          # kN/m (水平方向)

        # 投影到滑面切向
        try:
            alpha = slip_surface.get_alpha(xm)
        except Exception:
            return 0.0

        # 水平力在滑面切向的分量
        return R_horizontal * np.cos(alpha)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": "wall", "enabled": self.enabled,
            "x": self.x, "top": self.top, "bottom": self.bottom,
            "top_width": self.top_width, "bottom_width": self.bottom_width,
            "gamma_concrete": self.gamma_concrete,
            "base_friction": self.base_friction,
            "passive_phi": self.passive_phi,
            "passive_gamma": self.passive_gamma,
            "safety_factor": self.safety_factor,
        }

    @classmethod
    def from_dict(cls, data: dict):
        w = cls(
            elem_id=int(data.get("id", 1)),
            x=float(data["x"]),
            top_elev=float(data["top"]),
            bottom_elev=float(data["bottom"]),
            top_width=float(data.get("top_width", 0.8)),
            bottom_width=float(data.get("bottom_width", 1.5)),
            concrete_unit_weight=float(data.get("gamma_concrete", 23.0)),
            base_friction_coef=float(data.get("base_friction", 0.5)),
            passive_phi_deg=float(data.get("passive_phi", 30.0)),
            passive_gamma=float(data.get("passive_gamma", 18.0)),
            safety_factor=float(data.get("safety_factor", 0.8)),
        )
        w.enabled = bool(data.get("enabled", True))
        return w

# ======================================================================
# 序列化工具
# ======================================================================
def reinforcements_from_dict(data_list) -> list:
    out = []
    for d in data_list or []:
        kind = d.get("kind", "")
        if kind == "anchor":
            out.append(Anchor.from_dict(d))
        elif kind == "pile":
            out.append(PileRow.from_dict(d))
        elif kind == "wall":                    # ★ 新增
            out.append(Wall.from_dict(d))
    return out


def reinforcements_to_dict(reinforcements) -> list:
    return [r.to_dict() for r in (reinforcements or [])]
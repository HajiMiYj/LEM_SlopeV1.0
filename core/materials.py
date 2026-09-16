# -*- coding: utf-8 -*-
"""
土体材料与抗剪强度本构模型模块

支持:
  · 常规有效应力 / Fredlund 双应力非饱和吸力本构
  · 深度效应 (c、φ 随深度线性增长, 可开关)
  · 参数概率分布 (8 个参数, 可开关)
"""
import numpy as np


# ======================================================================
class Distribution:
    """单参数概率分布"""

    DET = "det"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRUNC_NORMAL = "trunc_normal"
    UNIFORM = "uniform"

    KIND_LABELS = {
        DET: "固定值",
        NORMAL: "正态分布",
        LOGNORMAL: "对数正态分布",
        TRUNC_NORMAL: "截断正态分布",
        UNIFORM: "均匀分布",
    }

    def __init__(self, kind: str = DET, mean: float = 1.0,
                 cov: float = 0.0, lower=None, upper=None):
        self.kind = str(kind)
        self.mean = float(mean)
        self.cov = float(cov)
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)

    @property
    def std(self) -> float:
        return abs(self.mean) * self.cov

    def sample(self, n: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        n = int(n)
        if self.kind == self.DET or self.cov <= 0.0:
            s = np.full(n, self.mean)
        elif self.kind == self.NORMAL:
            s = rng.normal(self.mean, self.std, n)
        elif self.kind == self.LOGNORMAL:
            if self.mean <= 0:
                s = np.full(n, self.mean)
            else:
                sig2 = np.log(1.0 + self.cov ** 2)
                mu = np.log(self.mean) - 0.5 * sig2
                s = rng.lognormal(mu, np.sqrt(sig2), n)
        elif self.kind == self.TRUNC_NORMAL:
            s = rng.normal(self.mean, self.std, n)
        elif self.kind == self.UNIFORM:
            lo_u = self.mean - self.std if self.lower is None else self.lower
            hi_u = self.mean + self.std if self.upper is None else self.upper
            s = (np.full(n, self.mean) if hi_u <= lo_u
                 else rng.uniform(lo_u, hi_u, n))
        else:
            s = np.full(n, self.mean)

        # ★ 严格 clip
        if self.lower is not None or self.upper is not None:
            lo = -np.inf if self.lower is None else self.lower
            hi = np.inf if self.upper is None else self.upper
            s = np.clip(s, lo, hi)
        return s

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        zero = np.zeros_like(x)
        if self.kind == self.DET or self.cov <= 0.0:
            sigma = max(1e-6, abs(self.mean) * 1e-4)
            return np.exp(-0.5 * ((x - self.mean) / sigma) ** 2) \
                   / (sigma * np.sqrt(2 * np.pi))
        if self.kind == self.NORMAL:
            s = max(self.std, 1e-9)
            return np.exp(-0.5 * ((x - self.mean) / s) ** 2) \
                   / (s * np.sqrt(2 * np.pi))
        if self.kind == self.LOGNORMAL:
            if self.mean <= 0:
                return zero
            sig2 = np.log(1.0 + self.cov ** 2)
            mu = np.log(self.mean) - 0.5 * sig2
            sigma = np.sqrt(sig2)
            mask = x > 0
            out = zero.copy()
            out[mask] = np.exp(-0.5 * ((np.log(x[mask]) - mu) / sigma) ** 2) \
                        / (x[mask] * sigma * np.sqrt(2 * np.pi))
            return out
        if self.kind == self.TRUNC_NORMAL:
            s = max(self.std, 1e-9)
            lo = -np.inf if self.lower is None else self.lower
            hi = np.inf if self.upper is None else self.upper
            raw = np.exp(-0.5 * ((x - self.mean) / s) ** 2) \
                  / (s * np.sqrt(2 * np.pi))
            return np.where((x >= lo) & (x <= hi), raw, 0.0)
        if self.kind == self.UNIFORM:
            lo = self.mean - self.std if self.lower is None else self.lower
            hi = self.mean + self.std if self.upper is None else self.upper
            width = max(1e-9, hi - lo)
            return np.where((x >= lo) & (x <= hi), 1.0 / width, 0.0)
        return zero

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """数值积分 PDF 得 CDF (通用, 不依赖 scipy)"""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        lo, hi = float(x.min()), float(x.max())
        n = 400
        xs = np.linspace(lo, hi, n)
        ys = self.pdf(xs)
        dx = (hi - lo) / (n - 1) if n > 1 else 1.0
        cdf_xs = np.cumsum(ys) * dx
        if cdf_xs[-1] > 1e-12:
            cdf_xs /= cdf_xs[-1]
        # 插值到目标 x
        return np.interp(x, xs, cdf_xs)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "mean": self.mean, "cov": self.cov,
                "lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, d) -> "Distribution":
        if not isinstance(d, dict):
            return cls()
        return cls(kind=d.get("kind", cls.DET),
                   mean=float(d.get("mean", 1.0)),
                   cov=float(d.get("cov", 0.0)),
                   lower=d.get("lower"), upper=d.get("upper"))


# ======================================================================
class SoilMaterial:
    """土体材料 (含深度效应 + 8 参数概率分布)

    8 个主值全部通过 property 指向对应分布的 mean, 双向同步:
      gamma_dry, gamma_sat, c_prime, phi_deg, phi_b_deg,
      suction_cutoff, c_depth_rate, phi_depth_rate
    """

    def __init__(
        self,
        name: str = "默认土层",
        gamma_dry: float = 19.0,
        gamma_sat: float = 21.0,
        c_prime: float = 15.0,
        phi_deg: float = 20.0,
        is_unsaturated: bool = False,
        phi_b_deg: float = 15.0,
        suction_cutoff: float = 100.0,
        # 深度效应
        use_depth_effect: bool = False,
        c_depth_rate: float = 0.0,
        phi_depth_rate: float = 0.0,
        depth_ref: float = 0.0,
        # 概率分布总开关
        use_random_dist: bool = False,
        # 8 个分布 (None → 用主值构造 DET)
        gamma_dist: Distribution = None,
        gamma_sat_dist: Distribution = None,
        c_dist: Distribution = None,
        phi_dist: Distribution = None,
        phi_b_dist: Distribution = None,
        suction_cutoff_dist: Distribution = None,
        c_depth_rate_dist: Distribution = None,
        phi_depth_rate_dist: Distribution = None,
    ):
        self.name = name
        self.is_unsaturated = bool(is_unsaturated)

        self.use_depth_effect = bool(use_depth_effect)
        self.use_random_dist = bool(use_random_dist)
        self.depth_ref = float(depth_ref)

        def _mk(d, mean):
            return d if isinstance(d, Distribution) else \
                Distribution(Distribution.DET, mean, 0.0)

        self.gamma_dist = _mk(gamma_dist, gamma_dry)
        self.gamma_sat_dist = _mk(gamma_sat_dist, gamma_sat)
        self.c_dist = _mk(c_dist, c_prime)
        self.phi_dist = _mk(phi_dist, phi_deg)
        self.phi_b_dist = _mk(phi_b_dist, phi_b_deg)
        self.suction_cutoff_dist = _mk(suction_cutoff_dist, suction_cutoff)
        self.c_depth_rate_dist = _mk(c_depth_rate_dist, c_depth_rate)
        self.phi_depth_rate_dist = _mk(phi_depth_rate_dist, phi_depth_rate)

    # ------------------------------------------------------------------
    # 8 个主值 property
    # ------------------------------------------------------------------
    @property
    def gamma_dry(self): return self.gamma_dist.mean

    @gamma_dry.setter
    def gamma_dry(self, v): self.gamma_dist.mean = float(v)

    @property
    def gamma_sat(self): return self.gamma_sat_dist.mean

    @gamma_sat.setter
    def gamma_sat(self, v): self.gamma_sat_dist.mean = float(v)

    @property
    def c_prime(self): return self.c_dist.mean

    @c_prime.setter
    def c_prime(self, v): self.c_dist.mean = float(v)

    @property
    def phi_deg(self): return self.phi_dist.mean

    @phi_deg.setter
    def phi_deg(self, v): self.phi_dist.mean = float(v)

    @property
    def phi_b_deg(self): return self.phi_b_dist.mean

    @phi_b_deg.setter
    def phi_b_deg(self, v): self.phi_b_dist.mean = float(v)

    @property
    def suction_cutoff(self): return self.suction_cutoff_dist.mean

    @suction_cutoff.setter
    def suction_cutoff(self, v): self.suction_cutoff_dist.mean = float(v)

    @property
    def c_depth_rate(self): return self.c_depth_rate_dist.mean

    @c_depth_rate.setter
    def c_depth_rate(self, v): self.c_depth_rate_dist.mean = float(v)

    @property
    def phi_depth_rate(self): return self.phi_depth_rate_dist.mean

    @phi_depth_rate.setter
    def phi_depth_rate(self, v): self.phi_depth_rate_dist.mean = float(v)

    # 只读派生
    @property
    def phi(self) -> float:
        return float(np.radians(self.phi_dist.mean))

    @property
    def phi_b(self) -> float:
        return float(np.radians(self.phi_b_dist.mean))

    # ------------------------------------------------------------------
    # 深度效应
    # ------------------------------------------------------------------
    def _depth_excess(self, z: float) -> float:
        return max(0.0, float(z) - self.depth_ref)

    def get_c_at_depth(self, z: float = 0.0) -> float:
        if not self.use_depth_effect:
            return self.c_prime
        return self.c_prime + self.c_depth_rate * self._depth_excess(z)

    def get_phi_deg_at_depth(self, z: float = 0.0) -> float:
        if not self.use_depth_effect:
            return self.phi_deg
        return self.phi_deg + self.phi_depth_rate * self._depth_excess(z)

    def get_phi_rad_at_depth(self, z: float = 0.0) -> float:
        return float(np.radians(self.get_phi_deg_at_depth(z)))

    # ------------------------------------------------------------------
    # 抗剪强度
    # ------------------------------------------------------------------
    def get_apparent_cohesion(self, suction: float, z: float = 0.0) -> float:
        c = self.get_c_at_depth(z)
        if not self.is_unsaturated or suction <= 0.0:
            return c
        eff = min(float(suction), self.suction_cutoff)
        return c + eff * np.tan(self.phi_b)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_unsaturated": self.is_unsaturated,
            "use_depth_effect": self.use_depth_effect,
            "use_random_dist": self.use_random_dist,
            "depth_ref": self.depth_ref,
            "gamma_dist": self.gamma_dist.to_dict(),
            "gamma_sat_dist": self.gamma_sat_dist.to_dict(),
            "c_dist": self.c_dist.to_dict(),
            "phi_dist": self.phi_dist.to_dict(),
            "phi_b_dist": self.phi_b_dist.to_dict(),
            "suction_cutoff_dist": self.suction_cutoff_dist.to_dict(),
            "c_depth_rate_dist": self.c_depth_rate_dist.to_dict(),
            "phi_depth_rate_dist": self.phi_depth_rate_dist.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoilMaterial":
        if not isinstance(d, dict):
            return cls()

        def _get_dist(key, fallback_mean):
            if key in d and isinstance(d[key], dict):
                return Distribution.from_dict(d[key])
            # 兼容旧 key 名 (旧工程文件)
            legacy = {"gamma_dist": "gamma_dry", "c_dist": "c_prime",
                      "phi_dist": "phi_deg", "gamma_sat_dist": "gamma_sat",
                      "phi_b_dist": "phi_b_deg",
                      "suction_cutoff_dist": "suction_cutoff",
                      "c_depth_rate_dist": "c_depth_rate",
                      "phi_depth_rate_dist": "phi_depth_rate"}
            lk = legacy.get(key)
            if lk and lk in d:
                return Distribution(Distribution.DET, float(d[lk]), 0.0)
            return Distribution(Distribution.DET, fallback_mean, 0.0)

        return cls(
            name=str(d.get("name", "材料")),
            is_unsaturated=bool(d.get("is_unsaturated", False)),
            use_depth_effect=bool(d.get("use_depth_effect", False)),
            use_random_dist=bool(d.get("use_random_dist", False)),
            depth_ref=float(d.get("depth_ref", 0.0)),
            gamma_dist=_get_dist("gamma_dist", 19.0),
            gamma_sat_dist=_get_dist("gamma_sat_dist", 21.0),
            c_dist=_get_dist("c_dist", 15.0),
            phi_dist=_get_dist("phi_dist", 20.0),
            phi_b_dist=_get_dist("phi_b_dist", 15.0),
            suction_cutoff_dist=_get_dist("suction_cutoff_dist", 100.0),
            c_depth_rate_dist=_get_dist("c_depth_rate_dist", 0.0),
            phi_depth_rate_dist=_get_dist("phi_depth_rate_dist", 0.0),
        )
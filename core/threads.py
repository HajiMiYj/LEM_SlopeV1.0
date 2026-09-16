# -*- coding: utf-8 -*-
"""
基于 QThread 的后台异步寻优计算工作线程
保证多层土、水力与复杂荷载条件下的全局迭代计算期间 GUI 界面平滑响应

非圆弧折线支持两种模式:
  - 自动搜索模式 (poly_auto_search=True): 表格里的点作为形状参考, DE 搜索最优
  - 手动试算模式 (poly_auto_search=False): 表格里的点就是最终滑面, 只算一次
"""
import logging
from typing import List, Tuple

from PyQt5.QtCore import QThread, pyqtSignal
from core.slicing import create_slices
from core.slip_surface import PolygonalSlipSurface
from core.solvers.bishop import BishopSolver
from core.solvers.janbu import JanbuSolver
from core.solvers.spencer import SpencerSolver
import numpy as np

from core.geometry import SlopeGeometry
from core.materials import SoilMaterial
from core.search import (
    PSOSearcher, SimulatedAnnealingSearcher,
    DifferentialEvolutionSearcher, NelderMeadSearcher, GridSearcher
)

logger = logging.getLogger(__name__)

# ======================================================================
# 非圆弧折线滑面搜索 —— 模块级纯函数工具集
# 从 OptimizationWorker 里抽出，方便单独阅读、测试和复用
# ======================================================================

class PolySearchContext:
    """折线滑面搜索的上下文（把 worker 里的依赖打包）"""

    def __init__(
        self,
        geom,
        materials,
        eval_name: str,
        rainfall_depth: float,
        kh: float,
        n_mid: int,
        H: float,
        is_interrupted,          # 无参 callable，返回 bool
    ):
        self.geom = geom
        self.materials = materials
        self.eval_name = eval_name
        self.rainfall_depth = rainfall_depth
        self.kh = kh
        self.n_mid = n_mid
        self.H = H
        self.is_interrupted = is_interrupted

        # 可变状态（在 eval 过程中被更新）
        self.best_fs_cache = [999.0]
        self.reject_stats = {}


def diagnose_surface(geom, pts, H):
    """
    对候选滑面做理论适用性诊断。
    返回 (ok, diag)：
      ok=False 表示数值不可计算（硬拒绝）
      diag['warnings'] 列出理论适用性下降的原因（软警告）
    """
    diag = {
        "max_depth_ratio": 0.0,
        "max_seg_angle_deg": 0.0,
        "min_seg_length": 1e9,
        "warnings": [],
        "reject_reason": None,
    }

    # 段长、段倾角
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        L = float((dx * dx + dy * dy) ** 0.5)
        diag["min_seg_length"] = min(diag["min_seg_length"], L)
        if L < 1e-3:
            diag["reject_reason"] = "相邻控制点重合"
            return False, diag
        ang = abs(float(np.degrees(np.arctan2(-dy, dx))))
        diag["max_seg_angle_deg"] = max(diag["max_seg_angle_deg"], ang)

    # 深度比
    for x, y in pts:
        depth = geom.get_ground_elevation(x) - y
        ratio = depth / max(1.0, H)
        diag["max_depth_ratio"] = max(diag["max_depth_ratio"], ratio)

    # 硬拒绝
    if diag["max_seg_angle_deg"] > 85.0:
        diag["reject_reason"] = (
            f"段倾角 {diag['max_seg_angle_deg']:.1f}° 接近垂直，数值不可计算"
        )
        return False, diag

    # 软警告
    if diag["max_depth_ratio"] > 1.0:
        diag["warnings"].append(
            f"滑面最大深度 {diag['max_depth_ratio']:.2f}H 超过坡高 H，"
            "刚体极限平衡假设适用性下降"
        )
    if diag["max_seg_angle_deg"] > 70.0:
        diag["warnings"].append(
            f"最大段倾角 {diag['max_seg_angle_deg']:.1f}°，"
            "完全平衡法条间力假设可能失效"
        )
    if diag["min_seg_length"] < 1.0:
        diag["warnings"].append(
            f"存在短线段 {diag['min_seg_length']:.2f}m，数值误差放大"
        )
    return True, diag


def evaluate_one_surface(geom, materials, pts, eval_name,
                         rainfall_depth, kh):
    """
    对单个滑面计算 FS，并返回理论适用性诊断。
    返回: (fs, diag)
      fs 为 None 表示几何不合法 / 数值失败
      只要几何合法，无论 FS 是多少都如实返回
    """
    from core.slip_surface import PolygonalSlipSurface
    from core.slicing import create_slices
    from core.solvers import JanbuSolver, SpencerSolver, BishopSolver

    if len(pts) < 2:
        return None, {"reject_reason": "控制点不足"}

    pts = sorted(pts, key=lambda p: p[0])

    # 端点必须贴地
    for i in (0, -1):
        x, y = pts[i]
        yg = geom.get_ground_elevation(x)
        if abs(y - yg) > 1.0:
            return None, {"reject_reason": f"端点未贴地 Δ={abs(y - yg):.2f}m"}

    # 中间点必须低于地面
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        yg = geom.get_ground_elevation(x)
        if y >= yg - 0.1:
            return None, {"reject_reason": f"中间点 {i} 高于地面"}

    # 折线整体不能穿出地面
    px = np.array([p[0] for p in pts])
    py = np.array([p[1] for p in pts])
    xs = np.linspace(px[0], px[-1], 80)
    ys = np.interp(xs, px, py)
    yg_samp = np.array([geom.get_ground_elevation(x) for x in xs])
    if np.any(ys > yg_samp + 0.05):
        return None, {"reject_reason": "折线穿出地面"}

    # 理论适用性诊断
    H = float(np.max(geom.gy) - np.min(geom.gy))
    ok, diag = diagnose_surface(geom, pts, H)
    if not ok:
        return None, diag

    # 计算
    try:
        slip = PolygonalSlipSurface(pts)
        slices, info, _ = create_slices(
            geom=geom, materials=materials,
            slip_surface=slip, n_slices=25,
            rainfall_depth=rainfall_depth, kh=kh
        )
        if not slices or not info:
            diag["reject_reason"] = "切片失败"
            return None, diag

        try:
            mc_x, mc_y = slip.get_moment_center()
        except Exception:
            mc_x = float(np.mean([p[0] for p in pts]))
            mc_y = float(np.mean([p[1] for p in pts]))

        R_eq = max(
            float(np.hypot(s.xm - mc_x, s.y_base - mc_y))
            for s in slices
        ) if slices else 1.0
        R_eq = max(1.0, R_eq)

        if "Spencer" in eval_name:
            solver = SpencerSolver(slices, mc_x, mc_y, R_eq,
                                   geom, info[2], slip_surface=slip)
        elif "Janbu" in eval_name:
            solver = JanbuSolver(slices, mc_x, mc_y, R_eq,
                                 geom, info[2], slip_surface=slip)
        elif "Bishop" in eval_name:
            solver = BishopSolver(slices, mc_x, mc_y, R_eq,
                                  geom, info[2], slip_surface=slip)
        else:
            solver = SpencerSolver(slices, mc_x, mc_y, R_eq,
                                   geom, info[2], slip_surface=slip)

        fs, _ = solver.solve()
        if fs is None or not np.isfinite(fs):
            diag["reject_reason"] = "求解器未返回有限值"
            return None, diag

        # 不做 FS 值过滤，如实返回
        return float(fs), diag
    except Exception as e:
        diag["reject_reason"] = f"求解异常: {e}"
        return None, diag


def make_eval_poly_fn(ctx: PolySearchContext):
    """
    工厂函数：返回给 differential_evolution 用的目标函数。
    闭包捕获 ctx，逻辑保持与原来一致。
    """
    geom = ctx.geom
    n_mid = ctx.n_mid
    H = ctx.H

    def eval_poly(params):
        if ctx.is_interrupted():
            return 999.0

        x_in = float(params[0])
        x_out = float(params[1])
        if x_out - x_in < 5.0:
            ctx.reject_stats["宽度不足"] = ctx.reject_stats.get("宽度不足", 0) + 1
            return 999.0

        y_in = geom.get_ground_elevation(x_in)
        y_out = geom.get_ground_elevation(x_out)

        mid = []
        for k in range(n_mid):
            xi = float(params[2 + 2 * k])
            yi = float(params[3 + 2 * k])
            if xi <= x_in + 1.0 or xi >= x_out - 1.0:
                ctx.reject_stats["x顺序错乱"] = ctx.reject_stats.get("x顺序错乱", 0) + 1
                return 999.0
            yg_i = geom.get_ground_elevation(xi)
            if yi >= yg_i - 0.1:
                ctx.reject_stats["中间点高于地面"] = ctx.reject_stats.get("中间点高于地面", 0) + 1
                return 999.0
            mid.append((xi, yi))

        pts = [(x_in, y_in)] + mid + [(x_out, y_out)]

        fs, diag = evaluate_one_surface(
            geom, ctx.materials, pts,
            ctx.eval_name, ctx.rainfall_depth, ctx.kh
        )
        if fs is None:
            key = diag.get("reject_reason", "其他")
            ctx.reject_stats[key] = ctx.reject_stats.get(key, 0) + 1
            return 999.0

        # 软惩罚：引导 DE 避开几何极端形状，不改变真实 FS 的物理含义
        penalty = 0.0
        depth_ratio = diag.get("max_depth_ratio", 0.0)
        if depth_ratio > 1.0:
            penalty += (depth_ratio - 1.0) * 5.0
        max_ang = diag.get("max_seg_angle_deg", 0.0)
        if max_ang > 60.0:
            penalty += (max_ang - 60.0) / 10.0

        ctx.best_fs_cache[0] = fs
        return fs + penalty

    return eval_poly


def find_crest_toe(gx, gy, y_tol=0.5):
    """自动识别坡肩 / 坡趾"""
    y_max = float(np.max(gy))
    y_min = float(np.min(gy))
    crest_idx = np.where(gy >= y_max - y_tol)[0]
    toe_idx = np.where(gy <= y_min + y_tol)[0]
    crest_x = float(gx[crest_idx[-1]]) if len(crest_idx) else float(gx[0])
    toe_x = float(gx[toe_idx[0]]) if len(toe_idx) else float(gx[-1])
    return crest_x, toe_x



class OptimizationWorker(QThread):
    """全局滑面寻优异步工作线程 (同时兼容圆弧与非圆弧折线)"""
    progress_updated = pyqtSignal(int, int, float)
    search_finished = pyqtSignal(float, list, str)
    search_failed = pyqtSignal(str)
    search_diagnostics = pyqtSignal(dict)

    def __init__(
        self,
        algo_name: str = "PSO",
        eval_name: str = "Bishop",
        geom: SlopeGeometry = None,
        materials: List[SoilMaterial] = None,
        bounds: List[Tuple[float, float]] = None,
        rainfall_depth: float = 0.0,
        kh: float = 0.0,
        surface_type: str = "circular",
        poly_ref_points: List[Tuple[float, float]] = None,
        poly_auto_search: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.algo_name = algo_name
        self.eval_name = eval_name
        self.geom = geom
        self.materials = materials
        self.bounds = bounds
        self.rainfall_depth = rainfall_depth
        self.kh = kh
        self.surface_type = surface_type
        self.poly_ref_points = poly_ref_points or []
        self.poly_auto_search = bool(poly_auto_search)
        self._is_interrupted = False

    def stop(self):
        self._is_interrupted = True

    # ==================================================================
    # 入口
    # ==================================================================
    def run(self):
        try:
            if self.surface_type == "polygonal":
                if self.poly_auto_search:
                    self._run_polygonal_search()
                else:
                    self._run_polygonal_manual()
            else:
                self._run_circular()
        except Exception as e:
            logger.exception("寻优计算顶层异常")
            self.search_failed.emit(f"寻优计算发生异常: {str(e)}")

    # ==================================================================
    # 通用工具: 构造单个滑面并算 FS
    # ==================================================================
 
    # ==================================================================
    # C 模式: 手动试算 (表格里的点就是最终滑面)
    # ==================================================================
    def _run_polygonal_manual(self):
        pts_raw = [(float(x), float(y)) for x, y in self.poly_ref_points]
        if len(pts_raw) < 2:
            self.search_failed.emit("非圆弧滑面至少需要 2 个控制点。")
            return

        pts_raw.sort(key=lambda p: p[0])
        x_in = pts_raw[0][0]
        x_out = pts_raw[-1][0]
        y_in = self.geom.get_ground_elevation(x_in)
        y_out = self.geom.get_ground_elevation(x_out)

        pts = [(x_in, y_in)]
        for (xi, yi) in pts_raw[1:-1]:
            yg_i = self.geom.get_ground_elevation(xi)
            pts.append((xi, min(yi, yg_i - 0.1)))
        pts.append((x_out, y_out))

        fs, diag = evaluate_one_surface(
            self.geom, self.materials, pts,
            self.eval_name, self.rainfall_depth, self.kh
        )
        if fs is None:
            self.search_failed.emit(
                f"手动滑面不合法: {diag.get('reject_reason', '未知')}"
            )
            return

        self.search_diagnostics.emit({
            "mode": "manual",
            "n_points": len(pts),
            "n_mid": max(0, len(pts) - 2),
            "reject_stats": {},
            "total_rejected": 0,
            "best_fs": float(fs),
        })
        self.search_finished.emit(fs, pts, "手动试算完成")

    # ==================================================================
    # A 模式: 自动搜索 (表格里的点作为形状参考)
    # ==================================================================

    def _run_polygonal_search(self):
        from scipy.optimize import differential_evolution

        n_ref = len(self.poly_ref_points)
        if n_ref < 2:
            self.search_failed.emit("自动搜索至少需要 2 个参考控制点。")
            return

        n_mid = n_ref - 2

        # ---------- 1. 地面高程采样 ----------
        xs_samp = np.linspace(self.geom.gx[0], self.geom.gx[-1], 200)
        ys_samp = np.array([self.geom.get_ground_elevation(x) for x in xs_samp])
        y_gmin = float(np.min(ys_samp))
        y_gmax = float(np.max(ys_samp))
        H = max(1.0, y_gmax - y_gmin)

        # ---------- 2. 处理 bounds ----------
        x_full_lo = float(self.geom.gx[0])
        x_full_hi = float(self.geom.gx[-1])

        x_in_lo, x_in_hi = self.bounds[0]
        x_out_lo, x_out_hi = self.bounds[2]
        x_in_lo = max(float(x_in_lo), x_full_lo)
        x_in_hi = min(float(x_in_hi), x_full_hi)
        x_out_lo = max(float(x_out_lo), x_full_lo)
        x_out_hi = min(float(x_out_hi), x_full_hi)

        # 自动识别坡肩/坡趾
        crest_x, toe_x = find_crest_toe(self.geom.gx, self.geom.gy)
        logger.debug("自动识别: 坡肩 x=%.2f, 坡趾 x=%.2f", crest_x, toe_x)

        x_in_lo = max(x_in_lo, x_full_lo)
        x_in_hi = min(x_in_hi, crest_x - 0.5)
        x_out_lo = max(x_out_lo, toe_x + 0.5)
        x_out_hi = min(x_out_hi, x_full_hi)

        if x_in_hi - x_in_lo < 2.0 or x_out_hi - x_out_lo < 2.0:
            x_mid_geo = 0.5 * (crest_x + toe_x)
            x_in_lo = x_full_lo
            x_in_hi = x_mid_geo - 1.0
            x_out_lo = x_mid_geo + 1.0
            x_out_hi = x_full_hi

        if x_in_lo >= x_in_hi or x_out_lo >= x_out_hi:
            self.search_failed.emit("X 搜索范围自动修正失败。")
            return

        # Y 范围修正
        y_lo_in, y_hi_in = self.bounds[1]
        if y_lo_in >= y_gmin - 0.5:
            y_bot_lo = y_gmin - 1.5 * H
            y_bot_hi = y_gmin - 0.5
        else:
            y_bot_lo = float(y_lo_in)
            y_bot_hi = float(y_hi_in)

        # ---------- 3. 参数 bounds ----------
        param_bounds = [
            (x_in_lo, x_in_hi),
            (x_out_lo, x_out_hi),
        ]
        for _ in range(n_mid):
            param_bounds.append((x_full_lo, x_full_hi))
            param_bounds.append((y_bot_lo, y_bot_hi))

        # ---------- 4. 构造上下文 + 目标函数 ----------
        ctx = PolySearchContext(
            geom=self.geom,
            materials=self.materials,
            eval_name=self.eval_name,
            rainfall_depth=self.rainfall_depth,
            kh=self.kh,
            n_mid=n_mid,
            H=H,
            is_interrupted=lambda: self._is_interrupted,
        )
        eval_fn = make_eval_poly_fn(ctx)

        # ---------- 5. 进度回调 ----------
        dim = len(param_bounds)
        popsize = int(min(25, max(10, 5 * dim)))
        maxiter = int(min(200, max(60, 20 * dim)))

        iter_cnt = [0]

        def cb(xk, convergence):
            iter_cnt[0] += 1
            if not self._is_interrupted:
                self.progress_updated.emit(iter_cnt[0], maxiter,
                                           ctx.best_fs_cache[0])

        # ---------- 6. 执行 DE ----------
        res = differential_evolution(
            eval_fn, param_bounds,
            popsize=popsize, maxiter=maxiter, tol=1e-3,
            callback=cb, seed=42, polish=True
        )

        logger.debug("DE 结果: fun=%.4f success=%s msg=%s",
                     res.fun, res.success, res.message)
        
                # ---------- 诊断信息 emit ----------
        diag_payload = {
            "mode": "auto_search",
            "n_ref": n_ref,
            "n_mid": n_mid,
            "dim": dim,
            "popsize": popsize,
            "maxiter": maxiter,
            "de_success": bool(res.success),
            "de_message": str(res.message),
            "de_fun": float(res.fun),
            "reject_stats": dict(ctx.reject_stats),
            "total_rejected": int(sum(ctx.reject_stats.values())),
            "bounds_final": {
                "x_in":  [float(param_bounds[0][0]), float(param_bounds[0][1])],
                "x_out": [float(param_bounds[1][0]), float(param_bounds[1][1])],
                "y_bot": [float(param_bounds[2][1]), float(param_bounds[3][1])],
            },
            "best_fs": float(res.fun),
        }
        self.search_diagnostics.emit(diag_payload)


        if res.fun >= 900.0:
            self.search_failed.emit(
                "未搜索到任何合法滑面。请检查：\n"
                "1) 滑底 Y 搜索范围是否位于地面以下；\n"
                "2) X 范围是否跨越坡肩与坡趾；\n"
                f"3) 参考控制点数 = {n_ref} 是否过多。"
            )
            return

        if self._is_interrupted:
            self.search_failed.emit("用户终止搜索")
            return

        # ---------- 7. 组装最优解 + 重算真实 FS ----------
        best_x_in = float(res.x[0])
        best_x_out = float(res.x[1])
        best_mid = []
        for k in range(n_mid):
            xi = float(res.x[2 + 2 * k])
            yi = float(res.x[3 + 2 * k])
            best_mid.append((xi, yi))

        best_points = (
            [(best_x_in, self.geom.get_ground_elevation(best_x_in))]
            + best_mid
            + [(best_x_out, self.geom.get_ground_elevation(best_x_out))]
        )
        best_points.sort(key=lambda p: p[0])

        real_fs, real_diag = evaluate_one_surface(
            self.geom, self.materials, best_points,
            self.eval_name, self.rainfall_depth, self.kh
        )

        if real_fs is None:
            self.search_failed.emit(
                f"DE 得到的候选滑面几何不可用: "
                f"{real_diag.get('reject_reason', '未知')}"
            )
            return

        msg = f"非圆弧 {n_ref} 点折线寻优完成"
        if not res.success:
            msg += " (DE 未完全收敛)"

        self.search_finished.emit(real_fs, best_points, msg)
    # ==================================================================
    # 圆弧模式
    # ==================================================================
    def _run_circular(self):
        kwargs = {
            "geom": self.geom,
            "materials": self.materials,
            "bounds": self.bounds,
            "eval_method": self.eval_name,
            "rainfall_depth": self.rainfall_depth,
            "kh": self.kh,
        }

        if "PSO" in self.algo_name:
            searcher = PSOSearcher(**kwargs, n_particles=25, max_iter=30)
        elif "退火" in self.algo_name:
            searcher = SimulatedAnnealingSearcher(**kwargs, cooling_rate=0.90)
        elif "差分进化" in self.algo_name:
            searcher = DifferentialEvolutionSearcher(**kwargs, popsize=10, maxiter=20)
        elif "单纯形" in self.algo_name:
            searcher = NelderMeadSearcher(**kwargs)
        else:
            searcher = GridSearcher(**kwargs, nx=10, ny=10, nr=8)

        def callback_fn(current, total, current_best):
            if not self._is_interrupted:
                self.progress_updated.emit(current, total, current_best)

        best_fs, best_params, info_msg = searcher.search(progress_callback=callback_fn)

        if self._is_interrupted:
            self.search_failed.emit("用户已主动终止搜索计算。")
            return

        try:
            bf = float(best_fs)
        except (TypeError, ValueError):
            bf = 999.0

        if bf >= 900.0:
            self.search_failed.emit(
                f"未搜索到任何合法圆弧滑面 (返回 FS={bf:.2f})。\n"
                f"请检查搜索范围 bounds 是否与坡体内部相交。"
            )
            return

        self.search_diagnostics.emit({
            "mode": "circular",
            "algo": self.algo_name,
            "eval": self.eval_name,
            "reject_stats": {},
            "total_rejected": 0,
            "best_fs": float(bf),
        })
        self.search_finished.emit(bf, [float(p) for p in best_params],
                                  str(info_msg))


class ReliabilityWorker(QThread):
    """边坡可靠度与失效概率评估后台异步工作线程"""
    progress_updated = pyqtSignal(int, int, float)
    reliability_finished = pyqtSignal(dict)
    reliability_failed = pyqtSignal(str)

    def __init__(self, simulator, parent=None):
        super().__init__(parent)
        self.simulator = simulator
        self._is_interrupted = False

    def stop(self):
        self._is_interrupted = True

    def run(self):
        try:
            def callback_fn(current, total, current_pf):
                if not self._is_interrupted:
                    self.progress_updated.emit(current, total, current_pf)

            def is_interrupted():
                return self._is_interrupted

            res = self.simulator.run(
                progress_callback=callback_fn,
                is_interrupted_fn=is_interrupted
            )

            if self._is_interrupted:
                self.reliability_failed.emit("用户已主动终止蒙特卡洛抽样计算。")
            elif not res.get("success", False):
                self.reliability_failed.emit(res.get("error", "抽样计算未收敛。"))
            else:
                self.reliability_finished.emit(res)
        except Exception as e:
            logger.exception("可靠度计算异常")
            self.reliability_failed.emit(f"可靠度计算发生异常: {str(e)}")
            
            

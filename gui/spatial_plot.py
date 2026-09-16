# -*- coding: utf-8 -*-
"""空间参数云图控件 (纯 Qt QPainter, 预计算缓存版)"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QImage,
                         QPainterPath, QPolygonF, QFont, QLinearGradient)
from PyQt5.QtCore import Qt, QPointF, QRectF


# ======================================================================
# 向量化点在多边形内判定 (ray casting, 一次算 6000 点)
# ======================================================================
def points_in_polygon(xx: np.ndarray, yy: np.ndarray, poly) -> np.ndarray:
    """向量化射线法: xx, yy 形状相同, poly 是 [(x,y), ...]
    返回: 同形状 bool 数组
    """
    if len(poly) < 3:
        return np.zeros_like(xx, dtype=bool)
    n = len(poly)
    inside = np.zeros(xx.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > yy) != (yj > yy))
        # 避免除零
        denom = yj - yi
        denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
        x_cross = (xj - xi) * (yy - yi) / denom + xi
        inside ^= (cond & (xx < x_cross))
        j = i
    return inside


# ======================================================================
class ColorMap:
    """简化版 colormap (jet-like, 无需 matplotlib)"""
    STOPS = [
        (0.00, (48, 18, 59)),
        (0.20, (32, 90, 180)),
        (0.40, (52, 180, 200)),
        (0.60, (120, 210, 110)),
        (0.80, (240, 200, 40)),
        (1.00, (200, 40, 40)),
    ]

    @classmethod
    def lut(cls, n=256):
        lut = np.zeros((n, 3), dtype=np.uint8)
        stops = cls.STOPS
        for i in range(n):
            t = i / (n - 1)
            for k in range(len(stops) - 1):
                t0, c0 = stops[k]
                t1, c1 = stops[k + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / max(t1 - t0, 1e-9)
                    lut[i] = [int(c0[j] + (c1[j] - c0[j]) * f)
                              for j in range(3)]
                    break
        return lut


# ======================================================================
class SpatialFieldWidget(QWidget):
    """坡体剖面参数云图控件

    · set_data() 时一次性预计算 RGBA QImage
    · paintEvent() 只做贴图 + 边框 + 色标 (约 1ms)
    """

    _BG = QColor(252, 253, 255)
    _BORDER = QColor(44, 62, 80)
    _TEXT = QColor(60, 70, 80)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._ground_pts = []
        self._regions = []
        self._title = ""
        self._unit = ""
        self._vmin = 0.0
        self._vmax = 1.0

        # 数据范围 (首次 set_data 时确定)
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 1.0

        # ★ 缓存: 每次 set_data 时构造一次
        self._cached_qimage = None
        self._cached_qimage_size = (0, 0)

        self._lut = ColorMap.lut(256)

    # ------------------------------------------------------------------
    def set_data(self, ground_pts, regions, grid_values, xs, ys,
                 title="", unit="", vmin=None, vmax=None):
        self._ground_pts = list(ground_pts or [])
        self._regions = list(regions or [])
        self._title = title
        self._unit = unit

        if grid_values is None or grid_values.size == 0:
            self._cached_qimage = None
            self.update()
            return

        # 数据范围
        if vmin is None:
            vmin = float(np.nanmin(grid_values))
        if vmax is None:
            vmax = float(np.nanmax(grid_values))
        if vmax <= vmin:
            vmax = vmin + 1e-3
        self._vmin = float(vmin)
        self._vmax = float(vmax)

        self._x_min = float(xs.min())
        self._x_max = float(xs.max())
        self._y_min = float(ys.min())
        self._y_max = float(ys.max())
        if self._x_max <= self._x_min: self._x_max = self._x_min + 1.0
        if self._y_max <= self._y_min: self._y_max = self._y_min + 1.0

        # ★ 核心: 一次性构造 RGBA QImage
        self._cached_qimage = self._build_qimage(grid_values)
        self.update()

    def clear(self, msg: str = "无云图数据"):
        self._cached_qimage = None
        self._title = msg
        self.update()

    # ------------------------------------------------------------------
    def _build_qimage(self, vals: np.ndarray) -> QImage:
        """归一化 → LUT → RGBA, 构造 QImage (只做一次)"""
        ny, nx = vals.shape
        span = self._vmax - self._vmin

        # 归一化 (NaN → 0, 后面用 alpha 遮掉)
        nan_mask = ~np.isfinite(vals)
        v_safe = np.where(nan_mask, self._vmin, vals)
        norm = np.clip((v_safe - self._vmin) / span * 255, 0, 255).astype(np.uint8)

        # LUT → RGB
        rgb = self._lut[norm]                # (ny, nx, 3)
        # alpha: NaN 处透明
        alpha = np.where(nan_mask, 0, 255).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])       # (ny, nx, 4)

        # 翻转 Y (数学 Y 向上 → 图像 Y 向下)
        rgba_flipped = rgba[::-1, :, :]
        rgba_flipped = np.ascontiguousarray(rgba_flipped)

        img = QImage(rgba_flipped.data, nx, ny, 4 * nx, QImage.Format_RGBA8888)
        # QImage 不持有 data, 必须 copy
        return img.copy()

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        # ★ 位图放大时关 AA, 关平滑插值 → 快
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.fillRect(self.rect(), self._BG)

        if not self._ground_pts or self._cached_qimage is None:
            p.setPen(QColor(150, 155, 160))
            f = QFont("Microsoft YaHei", 10)
            f.setItalic(True)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter,
                       self._title or "无云图数据")
            return

        W = self.width()
        H = self.height()
        ml, mr, mt, mb = 20, 60, 24, 26
        plot_w = max(50, W - ml - mr)
        plot_h = max(50, H - mt - mb)

        x_min, x_max = self._x_min, self._x_max
        y_min, y_max = self._y_min, self._y_max

        # 保持比例
        data_ar = (x_max - x_min) / max(y_max - y_min, 1e-9)
        plot_ar = plot_w / max(plot_h, 1)
        if data_ar > plot_ar:
            new_h = plot_w / data_ar
            oy = mt + (plot_h - new_h) / 2
            plot_rect = QRectF(ml, oy, plot_w, new_h)
        else:
            new_w = plot_h * data_ar
            ox = ml + (plot_w - new_w) / 2
            plot_rect = QRectF(ox, mt, new_w, plot_h)

        # ★ 一步贴图 (QImage 已经翻好 Y, 直接铺)
        p.drawImage(plot_rect, self._cached_qimage)

        def data_to_px(x, y):
            px = plot_rect.left() + (x - x_min) / (x_max - x_min) * plot_rect.width()
            py = plot_rect.bottom() - (y - y_min) / (y_max - y_min) * plot_rect.height()
            return QPointF(px, py)

        # 坡体轮廓
        if len(self._ground_pts) >= 2:
            path = QPainterPath()
            pts_px = [data_to_px(x, y) for x, y in self._ground_pts]
            path.moveTo(pts_px[0])
            for pt in pts_px[1:]:
                path.lineTo(pt)
            p.setPen(QPen(self._BORDER, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        # 土层边界 (虚线)
        p.setPen(QPen(QColor(80, 90, 100, 160), 1, Qt.DashLine))
        for reg in self._regions:
            pts = reg.get("points", [])
            if len(pts) < 3:
                continue
            poly = QPolygonF([data_to_px(x, y) for x, y in pts])
            p.drawPolygon(poly)

        # 色标
        self._draw_colorbar(p, plot_rect)

        # 标题
        p.setPen(self._TEXT)
        f = QFont("Microsoft YaHei", 9)
        f.setBold(True)
        p.setFont(f)
        title = self._title
        if self._unit:
            title += f"  [{self._unit}]"
        p.drawText(QPointF(plot_rect.left(), plot_rect.top() - 6), title)

    # ------------------------------------------------------------------
    def _draw_colorbar(self, p, plot_rect):
        bar_w = 16
        bar_x = plot_rect.right() + 20
        bar_y = plot_rect.top()
        bar_h = plot_rect.height()

        grad = QLinearGradient(bar_x, bar_y + bar_h, bar_x, bar_y)
        for t, (r, g, b) in ColorMap.STOPS:
            grad.setColorAt(t, QColor(r, g, b))
        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(QBrush(grad))
        p.drawRect(QRectF(bar_x, bar_y, bar_w, bar_h))

        p.setPen(self._TEXT)
        f = QFont("Microsoft YaHei", 8)
        p.setFont(f)
        for i in range(5):
            t = i / 4
            v = self._vmin + (self._vmax - self._vmin) * t
            y = bar_y + bar_h * (1 - t)
            p.drawLine(QPointF(bar_x + bar_w, y), QPointF(bar_x + bar_w + 3, y))
            p.drawText(QPointF(bar_x + bar_w + 5, y + 3), self._fmt(v))

    @staticmethod
    def _fmt(v):
        av = abs(v)
        if av >= 1000: return f"{v:.0f}"
        if av >= 100:  return f"{v:.1f}"
        if av >= 1:    return f"{v:.2f}"
        if av >= 0.01: return f"{v:.3f}"
        return f"{v:.1e}"
# -*- coding: utf-8 -*-
"""分布曲线预览控件 (纯 Qt QPainter, 含 PDF + CDF 双图)"""
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QPainterPath,
                         QFont, QPixmap)
from PyQt5.QtCore import Qt, QPointF, QRectF


class DistributionPlotWidget(QWidget):
    """概率分布预览控件

    上半部: PDF 概率密度曲线 (蓝色填充)
    下半部: CDF 累积分布曲线 (绿色)
    支持: 均值竖线, 截断区间阴影, 空状态提示
    导出: export_png(filepath)
    """

    _BG = QColor(252, 253, 255)
    _GRID = QColor(232, 236, 241)
    _AXIS = QColor(120, 130, 140)
    _PDF = QColor(41, 128, 185)
    _PDF_FILL = QColor(41, 128, 185, 45)
    _CDF = QColor(39, 174, 96)
    _CDF_FILL = QColor(39, 174, 96, 40)
    _MEAN = QColor(192, 57, 43)
    _TRUNC = QColor(241, 196, 15, 55)
    _TEXT = QColor(60, 70, 80)
    _TEXT_DIM = QColor(150, 155, 160)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(360)
        self.setMinimumWidth(320)
        from PyQt5.QtWidgets import QSizePolicy as SP
        self.setSizePolicy(SP.Expanding, SP.Expanding)
        self._dist = None
        self._title = ""
        self._unit = ""

    def set_distribution(self, dist, title: str = "", unit: str = ""):
        self._dist = dist
        self._title = title
        self._unit = unit
        self.update()

    def clear(self, msg: str = "未启用概率分布"):
        self._dist = None
        self._title = msg
        self._unit = ""
        self.update()

    def export_png(self, filepath: str) -> bool:
        try:
            pix = self.grab()
            return bool(pix.save(filepath, "PNG"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self._BG)

        if self._dist is None:
            p.setPen(self._TEXT_DIM)
            f = QFont("Microsoft YaHei", 10)
            f.setItalic(True)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter,
                       self._title or "未启用概率分布")
            return

        W = self.width()
        H = self.height()
        # 上下两图, 上 55% 下 45%
        gap = 22
        h_pdf = int((H - gap) * 0.55)
        h_cdf = H - gap - h_pdf

        # 采样范围
        mean = self._dist.mean
        std = max(self._dist.std, 1e-9)
        if self._dist.kind == self._dist.UNIFORM:
            lo = self._dist.lower if self._dist.lower is not None else mean - std
            hi = self._dist.upper if self._dist.upper is not None else mean + std
        else:
            span = max(std * 4.0, abs(mean) * 0.15, 1e-3)
            lo = mean - span
            hi = mean + span
        pad = (hi - lo) * 0.04
        lo -= pad
        hi += pad
        if hi <= lo:
            hi = lo + 1.0

        n = 200
        xs = np.linspace(lo, hi, n)
        ys_pdf = self._dist.pdf(xs)
        ys_cdf = self._dist.cdf(xs)

        # 上半: PDF
        r_pdf = QRectF(0, 0, W, h_pdf)
        self._draw_curve(p, r_pdf, xs, ys_pdf, lo, hi,
                         y_label="PDF", curve_color=self._PDF,
                         fill_color=self._PDF_FILL,
                         title=(self._title + (f"  [{self._unit}]" if self._unit else "")),
                         show_stats=True, mean=mean)

        # 下半: CDF
        r_cdf = QRectF(0, h_pdf + gap, W, h_cdf)
        self._draw_curve(p, r_cdf, xs, ys_cdf, lo, hi,
                         y_label="CDF", curve_color=self._CDF,
                         fill_color=self._CDF_FILL,
                         title="累积分布 CDF",
                         show_stats=False, mean=mean)

    # ------------------------------------------------------------------
    def _draw_curve(self, p, rect, xs, ys, lo, hi,
                    y_label, curve_color, fill_color,
                    title, show_stats, mean):
        ml, mr, mt, mb = 50, 20, 24, 26
        W = max(1, int(rect.width()) - ml - mr)
        H = max(1, int(rect.height()) - mt - mb)
        ox = rect.left() + ml
        oy = rect.top() + mt
        plot = QRectF(ox, oy, W, H)

        y_min = 0.0
        y_max = float(ys.max()) if ys.size and ys.max() > 1e-12 else 1.0

        def to_px(xv, yv):
            px = plot.left() + (xv - lo) / (hi - lo) * W
            py = plot.bottom() - (yv - y_min) / (y_max - y_min) * H
            return QPointF(px, py)

        # 网格
        p.setPen(QPen(self._GRID, 1, Qt.DotLine))
        for i in range(1, 4):
            yy = plot.top() + H * i / 4
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
        for i in range(1, 4):
            xx = plot.left() + W * i / 4
            p.drawLine(QPointF(xx, plot.top()), QPointF(xx, plot.bottom()))

        # 截断阴影
        if self._dist.kind in (self._dist.TRUNC_NORMAL, self._dist.UNIFORM):
            t_lo = self._dist.lower if self._dist.lower is not None else lo
            t_hi = self._dist.upper if self._dist.upper is not None else hi
            t_lo = max(t_lo, lo)
            t_hi = min(t_hi, hi)
            if t_hi > t_lo:
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(self._TRUNC))
                x1 = plot.left() + (t_lo - lo) / (hi - lo) * W
                x2 = plot.left() + (t_hi - lo) / (hi - lo) * W
                p.drawRect(QRectF(x1, plot.top(), x2 - x1, H))

        # 曲线 + 填充
        curve = QPainterPath()
        fill = QPainterPath()
        fill.moveTo(plot.left(), plot.bottom())
        for i in range(len(xs)):
            pt = to_px(xs[i], ys[i])
            if i == 0:
                curve.moveTo(pt)
            else:
                curve.lineTo(pt)
            fill.lineTo(pt)
        fill.lineTo(plot.right(), plot.bottom())
        fill.closeSubpath()

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(fill_color))
        p.drawPath(fill)
        p.setPen(QPen(curve_color, 2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(curve)

        # 均值竖线
        if lo <= mean <= hi:
            mx = plot.left() + (mean - lo) / (hi - lo) * W
            p.setPen(QPen(self._MEAN, 1.4, Qt.DashLine))
            p.drawLine(QPointF(mx, plot.top()), QPointF(mx, plot.bottom()))

        # 坐标轴
        p.setPen(QPen(self._AXIS, 1.5))
        p.drawLine(plot.bottomLeft(), plot.bottomRight())
        p.drawLine(plot.bottomLeft(), plot.topLeft())

        # X 刻度
        f = QFont("Microsoft YaHei", 8)
        p.setFont(f)
        p.setPen(self._TEXT)
        for i in range(5):
            xv = lo + (hi - lo) * i / 4
            xx = plot.left() + W * i / 4
            p.drawLine(QPointF(xx, plot.bottom()),
                       QPointF(xx, plot.bottom() + 3))
            txt = self._fmt(xv)
            tw = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(QPointF(xx - tw / 2, plot.bottom() + 15), txt)

        # Y 轴标签
        p.setPen(self._TEXT_DIM)
        p.drawText(QPointF(6, plot.top() + 10), y_label)

        # 标题 + 统计
        p.setPen(self._TEXT)
        f2 = QFont("Microsoft YaHei", 9)
        f2.setBold(True)
        p.setFont(f2)
        p.drawText(QPointF(plot.left(), rect.top() + 14), title)

        if show_stats and self._dist is not None:
            p.setFont(QFont("Microsoft YaHei", 8))
            p.setPen(self._TEXT_DIM)
            stat = (f"μ = {self._fmt(self._dist.mean)}   "
                    f"σ = {self._fmt(self._dist.std)}   "
                    f"COV = {self._dist.cov:.3f}")
            tw = p.fontMetrics().horizontalAdvance(stat)
            p.drawText(QPointF(plot.right() - tw, rect.top() + 14), stat)

    @staticmethod
    def _fmt(v: float) -> str:
        av = abs(v)
        if av >= 1000: return f"{v:.0f}"
        if av >= 100:  return f"{v:.1f}"
        if av >= 1:    return f"{v:.2f}"
        if av >= 0.01: return f"{v:.3f}"
        return f"{v:.2e}"
# -*- coding: utf-8 -*-
"""
基于 PyQt5 QGraphicsView 的专业 CAD 级边坡交互视口
"""
import numpy as np
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPolygonItem,
    QGraphicsPathItem, QGraphicsLineItem, QGraphicsRectItem,
    QGraphicsEllipseItem
)
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF, QPainterPath, QPixmap, QTransform
from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF, pyqtSignal


# ======================================================================
# 土条图元: 悬停高亮 + 力学物理量 Tooltip
# ======================================================================
class SliceGraphicsItem(QGraphicsPolygonItem):
    """具有悬停高亮与力学物理量 Tooltip 的土条交互图元"""

    def __init__(self, slice_data, polygon: QPolygonF):
        super().__init__(polygon)
        self.slice_data = slice_data

        self.default_brush = QBrush(QColor(243, 156, 18, 55))
        self.hover_brush = QBrush(QColor(231, 76, 60, 180))
        self.setBrush(self.default_brush)

        pen = QPen(QColor(127, 140, 141), 1)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.setAcceptHoverEvents(True)
        self._update_tooltip()

    def _update_tooltip(self):
        s = self.slice_data
        tip = (
            f"<div style='font-family: Microsoft YaHei, SimHei; font-size: 12px; color: #2c3e50;'>"
            f"<b>【土条 #{s.index} 受力与几何参数】</b><hr style='margin: 4px 0;'>"
            f"<b>所属土层:</b> {s.layer_name}<br>"
            f"<b>水平中点 X:</b> {s.xm:.2f} m<br>"
            f"<b>条块宽度 b:</b> {s.b:.2f} m | <b>平均高度 h:</b> {s.h:.2f} m<br>"
            f"<b>土体净自重:</b> {s.W_soil:.2f} kN<br>"
            f"<b>坡顶附加荷载:</b> {s.q_load:.2f} kN<br>"
            f"<b>总竖向力 W:</b> {s.W:.2f} kN<br>"
            f"<b>水平地震惯性力 Fh:</b> {s.Fh:.2f} kN (kh={s.kh:.2f})<br>"
            f"<b>孔隙水压力 u:</b> {s.u:.2f} kPa<br>"
            f"<b>非饱和基质吸力:</b> {s.suction:.2f} kPa<br>"
            f"<b>综合黏聚力 c_total:</b> {s.c:.2f} kPa<br>"
            f"<b>有效摩擦角 φ':</b> {np.degrees(s.phi):.2f}°<br>"
            f"<b>底坡倾角 α:</b> {np.degrees(s.alpha):.2f}° | <b>底弧长 l:</b> {s.l:.2f} m"
            f"</div>"
        )
        self.setToolTip(tip)

    def hoverEnterEvent(self, event):
        self.setBrush(self.hover_brush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(self.default_brush)
        super().hoverLeaveEvent(event)


# ======================================================================
# 填充图案工厂 (保持原样)
# ======================================================================
class HatchPatternFactory:
    """按岩土工程制图规范生成填充画刷

    ★ 关键设计: pixmap 尺寸由 view 缩放在 paint() 时动态决定,
      保证图案在屏幕上 1:1 像素映射 (不糊)。
    """
    _cache = {}

    @classmethod
    def make_brush(cls, pattern_id: str, base_color: QColor,
                   pix_size: int) -> QBrush:
        if pattern_id == "solid":
            return QBrush(base_color, Qt.SolidPattern)

        pix_size = max(8, min(128, int(pix_size)))
        key = (pattern_id, base_color.name(), pix_size)
        if key in cls._cache:
            return QBrush(cls._cache[key])

        pix = QPixmap(pix_size, pix_size)
        pix.fill(base_color)

        painter = QPainter()
        try:
            if not painter.begin(pix):
                return QBrush(base_color, Qt.SolidPattern)
            painter.setRenderHint(QPainter.Antialiasing)

            fg = base_color.darker(170)
            n = pix_size
            pen_w = max(1.0, n / 40.0)
            painter.setPen(QPen(fg, pen_w))

            if pattern_id == "clay":
                step = max(8, n // 2)
                for i in range(-n, 2 * n, step):
                    painter.drawLine(QPointF(i, 0), QPointF(i + n, n))

            elif pattern_id == "silt":
                step = max(6, n // 3)
                for x in range(step, n, step):
                    painter.drawLine(QPointF(x, n * 0.15), QPointF(x, n * 0.75))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(fg))
                r = max(0.8, n * 0.04)
                for x in range(step + step // 2, n, step):
                    painter.drawEllipse(QPointF(x, n * 0.88), r, r)

            elif pattern_id == "sand":
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(fg))
                r = max(1.0, n * 0.05)
                for xi in (0.25, 0.75):
                    for yi in (0.25, 0.75):
                        painter.drawEllipse(QPointF(n * xi, n * yi), r, r)

            elif pattern_id == "fill":
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(fg))
                r = max(1.0, n * 0.04)
                for xi in (0.3, 0.8):
                    for yi in (0.3, 0.8):
                        painter.drawEllipse(QPointF(n * xi, n * yi), r, r)

            elif pattern_id == "gravel":
                step = max(8, n // 2)
                for i in range(-n, 2 * n, step):
                    painter.drawLine(QPointF(i, 0), QPointF(i + n, n))
                    painter.drawLine(QPointF(i, n), QPointF(i + n, 0))

            elif pattern_id == "rock_strong":
                step = max(6, n // 3)
                for i in range(-n, 2 * n, step):
                    painter.drawLine(QPointF(i, 0), QPointF(i + n, n))
                    painter.drawLine(QPointF(i, n), QPointF(i + n, 0))

            elif pattern_id == "rock_medium":
                step = max(8, n // 2)
                for i in range(-n, 2 * n, step):
                    painter.drawLine(QPointF(i, 0), QPointF(i + n, n))

            elif pattern_id == "rock_fresh":
                painter.setBrush(QBrush(Qt.NoBrush))
                tri = n * 0.35
                for xi, yi in ((0.1, 0.1), (0.55, 0.1), (0.1, 0.55), (0.55, 0.55)):
                    x0, y0 = n * xi, n * yi
                    poly = QPolygonF([
                        QPointF(x0, y0),
                        QPointF(x0 + tri, y0),
                        QPointF(x0 + tri / 2, y0 + tri),
                    ])
                    painter.drawPolygon(poly)

            else:
                return QBrush(base_color, Qt.SolidPattern)

        except Exception:
            return QBrush(base_color, Qt.SolidPattern)
        finally:
            try:
                if painter.isActive():
                    painter.end()
            except Exception:
                pass

        if len(cls._cache) > 64:
            cls._cache.clear()
        cls._cache[key] = pix
        return QBrush(pix)


# ======================================================================
# 土层面图元: 填充 + hover / 选中高亮 + Tooltip + 点击通知
# ======================================================================
class LayerRegionGraphicsItem(QGraphicsPathItem):
    """土层面图元 (带洞 + 规范填充 + hover / 选中高亮 + Tooltip)

    · 三种视觉状态: default / hovered / selected
    · 左键点击 → 通知所属 View 处理选中 (支持 Ctrl 多选)
    """

    def __init__(self, region_data, path, material_info=None,
                 region_index=-1, tile_m=4.0):
        super().__init__(path)
        self.region_data = region_data
        self.material_info = material_info or {}
        self.region_index = region_index
        self._tile_m = max(0.5, float(tile_m))

        self._base_color = QColor(region_data.get("color", "#f3dfaa"))
        self._pattern = region_data.get("pattern", "solid")

        self.default_pen = QPen(self._base_color.darker(160), 1.5)
        self.default_pen.setCosmetic(True)
        self.hover_pen = QPen(QColor(41, 128, 185), 2.5)
        self.hover_pen.setCosmetic(True)
        self.selected_pen = QPen(QColor(192, 57, 43), 3.0)
        self.selected_pen.setCosmetic(True)

        self._hovered = False
        self._selected = False

        self.setPen(self.default_pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.setAcceptHoverEvents(True)
        self._build_tooltip()

    # ------------------------------------------------------------------
    # 重写 paint
    # ------------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 底色 + 图案 (在 path 裁剪区内)
        painter.save()
        painter.setClipPath(self.path())
        painter.fillPath(self.path(), QBrush(self._base_color))
        if self._pattern != "solid":
            self._paint_hatch(painter)
        painter.restore()

        # 2. 边框
        if self._selected:
            pen = self.selected_pen
        elif self._hovered:
            pen = self.hover_pen
        else:
            pen = self.default_pen
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def _paint_hatch(self, painter):
        # view 缩放 (屏幕像素 / 场景米)
        scale = 1.0
        try:
            views = self.scene().views() if self.scene() else []
            if views:
                scale = abs(views[0].transform().m11())
        except Exception:
            scale = 1.0
        scale = max(scale, 0.05)

        # pixmap 尺寸 = tile 在屏幕上的像素数 (1:1 映射 → 不糊)
        T = self._tile_m
        P = int(T * scale)
        P = max(8, min(128, P))

        brush = HatchPatternFactory.make_brush(
            self._pattern, self._base_color, P)

        # brush 缩放: pixmap P 像素 → 场景 T 米
        t = QTransform()
        t.scale(T / P, T / P)
        brush.setTransform(t)

        painter.fillPath(self.path(), brush)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene = self.scene()
            if scene is not None:
                for v in scene.views():
                    if hasattr(v, "_handle_region_click"):
                        v._handle_region_click(self.region_index,
                                               event.modifiers())
                        break
        super().mousePressEvent(event)

    def set_selected(self, flag):
        self._selected = bool(flag)
        self.update()

    # ------------------------------------------------------------------
    # Tooltip
    # ------------------------------------------------------------------
    def _build_tooltip(self):
        name = self.region_data.get("name", "土层面")
        mat = self.material_info
        mat_name = mat.get("name", "—")

        pattern_cn = {
            "solid": "纯色",
            "clay": "45°斜线 (黏土)",
            "silt": "竖短线+点 (粉质黏土)",
            "sand": "点状 (砂土)",
            "gravel": "交叉网格 (砾石)",
            "rock_strong": "交叉斜线 (强风化岩)",
            "rock_medium": "单向斜线 (中风化岩)",
            "rock_fresh": "三角符号 (新鲜岩石)",
            "fill": "稀疏点 (素填土)",
        }.get(self.region_data.get("pattern", "solid"), "—")

        lines = [
            f"<b style='font-size:13px;'>{name}</b>",
            "<hr style='margin:4px 0;'>",
            f"<b>材料:</b> {mat_name}",
        ]
        if mat:
            lines += [
                f"<b>γ:</b> {mat.get('gamma_dry', '—')} kN/m³",
                f"<b>γsat:</b> {mat.get('gamma_sat', '—')} kN/m³",
                f"<b>c':</b> {mat.get('c_prime', '—')} kPa",
                f"<b>φ':</b> {mat.get('phi_deg', '—')}°",
            ]

        pts = self.region_data.get("points", [])
        holes = self.region_data.get("holes", [])
        lines.append(f"<b>填充:</b> {pattern_cn}")
        lines.append(f"<b>外环顶点:</b> {len(pts)}")
        if holes:
            lines.append(f"<b>洞数:</b> {len(holes)}")

        self.setToolTip(
            "<div style='font-family: Microsoft YaHei; font-size: 12px; "
            "color: #2c3e50;'>" + "<br>".join(lines) + "</div>"
        )

    # ------------------------------------------------------------------
    # 状态刷新
    # ------------------------------------------------------------------
    def _refresh_style(self):
        if self._selected:
            self.setBrush(self.selected_brush)
            self.setPen(self.selected_pen)
        elif self._hovered:
            self.setBrush(self.hover_brush)
            self.setPen(self.hover_pen)
        else:
            self.setBrush(self.default_brush)
            self.setPen(self.default_pen)


# ======================================================================
# 主视口
# ======================================================================
class SlopeGraphicsView(QGraphicsView):
    """基于 PyQt5 QGraphicsView 的专业 CAD 级交互视口"""

    polygon_completed = pyqtSignal(list)
    polyline_completed = pyqtSignal(list)
    drawing_status = pyqtSignal(str)
    # 画布侧选中集变化 → 通知外部 (list of region_index)
    regions_selection_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._hatch_scale = 1.0
        self.scale(1, -1)  # Y 轴翻转, 符合工程坐标直觉
        self.setBackgroundBrush(QBrush(QColor(250, 252, 255)))

        # ---- 平移 ----
        self._is_panning = False
        self._pan_start_pos = None

        # ---- 绘制模式 ----
        self._drawing_polygon = False
        self._drawing_mode = None  # "polygon" | "polyline"
        self._polygon_points = []
        self._drawing_item = None
        self._preview_cursor = None

        # ---- 吸附 ----
        self._snap_tol = 0.5  # 场景单位 (米), 由外部设置
        self._snap_mode = "both"  # "none" | "point" | "line" | "both"
        self._snap_points = []
        self._snap_segments = []
        self._snap_marker = None

        # ---- 土层面 ----
        self._layer_items = []
        self._selected_regions = set()  # 当前选中的 region_index 集合

    # ==================================================================
    # 基本设置
    # ==================================================================
    def set_hatch_scale(self, scale: float):
        self._hatch_scale = float(scale)

    def set_snap_options(self, tol: float, mode: str):
        """设置吸附容差 (米) 和吸附模式"""
        self._snap_tol = float(tol)
        self._snap_mode = str(mode)

    # ==================================================================
    # 土层面选中管理 (图层树 ↔ 画布 双向同步)
    # ==================================================================
    def set_selected_regions(self, indices):
        """外部 (图层树) 设置选中集合, 不发射信号, 避免回环"""
        try:
            new_set = set(int(i) for i in (indices or []) if int(i) >= 0)
        except (TypeError, ValueError):
            new_set = set()

        if new_set == self._selected_regions:
            return
        self._selected_regions = new_set
        self._apply_region_selection()

    def _apply_region_selection(self):
        """把 _selected_regions 应用到实际图元 (RuntimeError 兜底)"""
        for item in self._layer_items:
            try:
                item.set_selected(item.region_index in self._selected_regions)
            except RuntimeError:
                # scene.clear() 后 C++ 端已销毁, 忽略
                pass
            except Exception:
                pass

    def _handle_region_click(self, idx, modifiers):
        """画布上点击某个土层面:
          · 无修饰键 → 单选
          · Ctrl     → toggle 该面
          · Shift    → 从当前选中范围扩大到 idx
        """
        ctrl = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)

        if ctrl:
            if idx in self._selected_regions:
                self._selected_regions.discard(idx)
            else:
                self._selected_regions.add(idx)
        elif shift and self._selected_regions:
            lo = min(min(self._selected_regions), idx)
            hi = max(max(self._selected_regions), idx)
            self._selected_regions = set(range(lo, hi + 1))
        else:
            self._selected_regions = {idx}

        self._apply_region_selection()
        self.regions_selection_changed.emit(sorted(self._selected_regions))

    def highlight_region(self, region_index):
        """兼容旧接口: 单值高亮 / <0 取消所有"""
        if region_index is None or region_index < 0:
            self.set_selected_regions([])
        else:
            self.set_selected_regions([region_index])

    # ==================================================================
    # 绘制模式控制
    # ==================================================================
    def start_polygon_drawing(self):
        self._drawing_polygon = True
        self._drawing_mode = "polygon"
        self._polygon_points = []
        self._drawing_item = None
        self.setCursor(Qt.CrossCursor)
        self.drawing_status.emit(
            "绘制土层面：左键加点，点击起点附近闭合，右键结束，Backspace 撤销，Esc 取消"
        )

    def start_polyline_drawing(self):
        self._drawing_polygon = True
        self._drawing_mode = "polyline"
        self._polygon_points = []
        self._drawing_item = None
        self.setCursor(Qt.CrossCursor)
        self.drawing_status.emit(
            "绘制切割线：左键加点，右键结束，Backspace 撤销，Esc 取消"
        )

    def cancel_polygon_drawing(self):
        self._drawing_polygon = False
        self._drawing_mode = None
        self._polygon_points = []
        self._clear_drawing_preview()
        self._clear_snap_marker()
        self.setCursor(Qt.ArrowCursor)
        self._preview_cursor = None

    def _clear_drawing_preview(self):
        if self._drawing_item is not None:
            try:
                self.scene.removeItem(self._drawing_item)
            except RuntimeError:
                pass
            except Exception:
                pass
            finally:
                self._drawing_item = None

    def _update_drawing_preview(self, cursor_point=None):
        if not self._polygon_points:
            return
        path = QPainterPath()
        path.moveTo(self._polygon_points[0])
        for point in self._polygon_points[1:]:
            path.lineTo(point)
        if cursor_point is not None:
            path.lineTo(cursor_point)

        if self._drawing_item is None:
            self._drawing_item = QGraphicsPathItem(path)
            pen = QPen(QColor(41, 128, 185), 2.0, Qt.DashLine)
            pen.setCosmetic(True)
            self._drawing_item.setPen(pen)
            if self._drawing_mode == "polygon":
                self._drawing_item.setBrush(QBrush(QColor(52, 152, 219, 45)))
            else:
                self._drawing_item.setBrush(QBrush(Qt.NoBrush))
            self.scene.addItem(self._drawing_item)
        else:
            try:
                self._drawing_item.setPath(path)
            except RuntimeError:
                self._drawing_item = None
                self._update_drawing_preview(cursor_point)

    def _finish_polygon_drawing(self):
        if len(self._polygon_points) < 3:
            self.drawing_status.emit("土层面至少需要 3 个点")
            return
        points = [(p.x(), p.y()) for p in self._polygon_points]
        self.cancel_polygon_drawing()
        self.polygon_completed.emit(points)
        self.drawing_status.emit("土层面已提交")

    def _finish_polyline_drawing(self):
        if len(self._polygon_points) < 2:
            self.drawing_status.emit("切割线至少需要 2 个点")
            return
        points = [(p.x(), p.y()) for p in self._polygon_points]
        self.cancel_polygon_drawing()
        self.polyline_completed.emit(points)
        self.drawing_status.emit("切割线已提交")

    # ==================================================================
    # 视图操作
    # ==================================================================
    def fit_view_to_slope(self, margin_ratio: float = 0.15):
        rect = self.scene.sceneRect()
        if rect.isEmpty():
            return
        dx = rect.width() * margin_ratio
        dy = rect.height() * margin_ratio
        target_rect = rect.adjusted(-dx, -dy, dx, dy)
        self.fitInView(target_rect, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.15
        elif delta < 0:
            factor = 1.0 / 1.15
        else:
            factor = 1.0
        self.scale(factor, factor)

    # ==================================================================
    # 吸附
    # ==================================================================
    def _snap_to_nearest(self, screen_pos, scene_pos):
        """用场景坐标做吸附比较, 容差用 UI 传入的 _snap_tol (米)

        返回 (snapped_QPointF, snap_type)
        """
        if self._snap_mode == "none" or self._snap_tol <= 0.0:
            return scene_pos, None

        tol = self._snap_tol
        best = None
        best_d2 = tol * tol
        best_type = None

        if self._snap_mode in ("point", "both"):
            for (px, py) in self._snap_points:
                d2 = (px - scene_pos.x()) ** 2 + (py - scene_pos.y()) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (px, py)
                    best_type = "point"

        if self._snap_mode in ("line", "both"):
            for ((x1, y1), (x2, y2)) in self._snap_segments:
                dx, dy = x2 - x1, y2 - y1
                L2 = dx * dx + dy * dy
                if L2 < 1e-9:
                    continue
                t = ((scene_pos.x() - x1) * dx + (scene_pos.y() - y1) * dy) / L2
                t = max(0.0, min(1.0, t))
                cx, cy = x1 + t * dx, y1 + t * dy
                d2 = (cx - scene_pos.x()) ** 2 + (cy - scene_pos.y()) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (cx, cy)
                    best_type = "line"

        if best is not None:
            return QPointF(best[0], best[1]), best_type
        return scene_pos, None

    def _update_snap_marker(self, scene_pos, snap_type):
        self._clear_snap_marker()
        if snap_type is None:
            return

        scale_factor = self.transform().m11()
        r = 0.3 if abs(scale_factor) < 1e-6 else 4.0 / abs(scale_factor)

        color = QColor(231, 76, 60)
        marker = QGraphicsEllipseItem(
            scene_pos.x() - r, scene_pos.y() - r, 2 * r, 2 * r
        )
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        marker.setPen(pen)
        marker.setBrush(QBrush(QColor(231, 76, 60, 120)))
        marker.setZValue(1000)
        self.scene.addItem(marker)
        self._snap_marker = marker

    def _clear_snap_marker(self):
        if self._snap_marker is not None:
            try:
                self.scene.removeItem(self._snap_marker)
            except Exception:
                pass
            self._snap_marker = None

    def _collect_snap_targets(self, ground_x, ground_y, layer_regions):
        """收集吸附点 (顶点) 与吸附线段 (边)"""
        points = []
        segments = []

        for i in range(len(ground_x) - 1):
            x1, y1 = float(ground_x[i]), float(ground_y[i])
            x2, y2 = float(ground_x[i + 1]), float(ground_y[i + 1])
            points.append((x1, y1))
            points.append((x2, y2))
            segments.append(((x1, y1), (x2, y2)))

        if layer_regions:
            for region in layer_regions:
                pts = region.get("points", [])
                n = len(pts)
                for i in range(n):
                    x1, y1 = float(pts[i][0]), float(pts[i][1])
                    x2, y2 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
                    points.append((x1, y1))
                    segments.append(((x1, y1), (x2, y2)))

        # 字典去重 (O(n))
        seen = {}
        unique = []
        for (x, y) in points:
            key = (round(x, 3), round(y, 3))
            if key not in seen:
                seen[key] = True
                unique.append((x, y))

        self._snap_points = unique
        self._snap_segments = segments

    # ==================================================================
    # 鼠标 / 键盘事件
    # ==================================================================
    def mousePressEvent(self, event):
        # ---- 绘制模式下: 右键结束 ----
        if self._drawing_polygon and event.button() == Qt.RightButton:
            if self._drawing_mode == "polyline" and len(self._polygon_points) >= 2:
                self._finish_polyline_drawing()
            elif self._drawing_mode == "polygon" and len(self._polygon_points) >= 3:
                self._finish_polygon_drawing()
            else:
                self.cancel_polygon_drawing()
                self.drawing_status.emit("已取消绘制")
            event.accept()
            return

        # ---- 绘制模式下: 左键加点 ----
        if self._drawing_polygon and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            point, _ = self._snap_to_nearest(event.pos(), scene_pos)

            # 与上一点重合 → 跳过 (处理双击时 Qt 连发两次 press)
            if self._polygon_points:
                last = self._polygon_points[-1]
                if (abs(point.x() - last.x()) < 1e-4
                        and abs(point.y() - last.y()) < 1e-4):
                    event.accept()
                    return

            # 多边形模式下点击起点附近 = 闭合
            if self._drawing_mode == "polygon" and self._polygon_points:
                start_screen = self.mapFromScene(self._polygon_points[0])
                cur_screen = event.pos()
                if (cur_screen - start_screen).manhattanLength() < 10:
                    self._finish_polygon_drawing()
                    event.accept()
                    return

            self._polygon_points.append(point)
            self._update_drawing_preview(point)
            event.accept()
            return

        # ---- 非绘制状态: 中键/右键 = 平移 ----
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing_polygon:
            scene_pos = self.mapToScene(event.pos())
            snapped, snap_type = self._snap_to_nearest(event.pos(), scene_pos)
            self._preview_cursor = snapped
            self._update_drawing_preview(snapped)
            self._update_snap_marker(snapped, snap_type)
            event.accept()
            return

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = False
            if not self._drawing_polygon:
                self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # 说明: 不实现 mouseDoubleClickEvent —— 交互约定为"左键加点, 右键结束"

    def keyPressEvent(self, event):
        if self._drawing_polygon:
            if event.key() == Qt.Key_Escape:
                self.cancel_polygon_drawing()
                self.drawing_status.emit("已取消绘制")
                event.accept()
                return
            if event.key() == Qt.Key_Backspace:
                if self._polygon_points:
                    self._polygon_points.pop()
                    self._update_drawing_preview(self._preview_cursor)
                event.accept()
                return
        super().keyPressEvent(event)

    # ==================================================================
    # 背景网格
    # ==================================================================
    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        painter.save()
        grid_pen = QPen(QColor(232, 236, 241), 1, Qt.DotLine)
        grid_pen.setCosmetic(True)
        painter.setPen(grid_pen)

        step = 5.0
        left = int(rect.left() / step) * step
        right = rect.right()
        bottom = int(rect.top() / step) * step
        top = rect.bottom()

        x = left
        while x <= right:
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step

        y = bottom
        while y <= top:
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

        painter.restore()

    # ==================================================================
    # 主渲染
    # ==================================================================
    def render_model(
            self,
            ground_x: list,
            ground_y: list,
            xc: float,
            yc: float,
            R: float,
            slices: list = None,
            water_pts: list = None,
            layer_regions: list = None,
            rainfall_depth: float = 0.0,
            surcharge_loads: list = None,
            slip_surface=None,
            reinforcements=None,
            materials=None,
            bottom_depth: float = 6.0,
    ):
        self.scene.clear()
        self._drawing_item = None
        self._snap_marker = None

        if len(ground_x) < 2:
            return

        min_y = min(ground_y) - float(bottom_depth)
        max_y = max(ground_y) + 8.0
        min_x = ground_x[0] - 5.0
        max_x = ground_x[-1] + 5.0

        # ---------- 1. 土层面 ----------
        self._layer_items = []
        if layer_regions:
            for r_idx, region in enumerate(layer_regions):
                if not isinstance(region, dict):
                    continue

                outer = region.get("points", [])
                holes = region.get("holes", [])
                if len(outer) < 3:
                    continue

                path = QPainterPath()
                path.setFillRule(Qt.OddEvenFill)
                path.addPolygon(QPolygonF(
                    [QPointF(float(x), float(y)) for x, y in outer]
                ))
                path.closeSubpath()
                for hole in holes:
                    if len(hole) >= 3:
                        path.addPolygon(QPolygonF(
                            [QPointF(float(x), float(y)) for x, y in hole]
                        ))
                        path.closeSubpath()

                mat_idx = int(region.get("material_index", 0))
                mat_info = None
                if materials and 0 <= mat_idx < len(materials):
                    m = materials[mat_idx]
                    mat_info = {
                        "name": getattr(m, "name", f"材料 {mat_idx + 1}"),
                        "gamma_dry": getattr(m, "gamma_dry", "—"),
                        "gamma_sat": getattr(m, "gamma_sat", "—"),
                        "c_prime": getattr(m, "c_prime", "—"),
                        "phi_deg": getattr(m, "phi_deg", "—"),
                    }

                # ★ 每个面用自己的 hatch_scale_pct → tile_m
                pct = int(region.get("hatch_scale_pct", 100))
                pct = max(20, min(500, pct))
                tile_m = 4.0 / max(0.05, pct / 100.0)

                item = LayerRegionGraphicsItem(
                    region, path, mat_info,
                    region_index=r_idx,
                    tile_m=tile_m,
                )
                self.scene.addItem(item)
                self._layer_items.append(item)

        # ---------- 2. 坡体基底轮廓 ----------
        poly_pts = [QPointF(ground_x[0], min_y)]
        for x, y in zip(ground_x, ground_y):
            poly_pts.append(QPointF(x, y))
        poly_pts.append(QPointF(ground_x[-1], min_y))

        soil_item = QGraphicsPolygonItem(QPolygonF(poly_pts))
        soil_item.setBrush(QBrush(Qt.NoBrush))
        pen_ground = QPen(QColor(44, 62, 80), 2.0)
        pen_ground.setCosmetic(True)
        soil_item.setPen(pen_ground)
        self.scene.addItem(soil_item)

        # ---------- 3. 降雨阴影 ----------
        if rainfall_depth > 0.0:
            rain_pts = []
            for x, y in zip(ground_x, ground_y):
                rain_pts.append(QPointF(x, y))
            for x, y in reversed(list(zip(ground_x, ground_y))):
                rain_pts.append(QPointF(x, y - rainfall_depth))
            rain_item = QGraphicsPolygonItem(QPolygonF(rain_pts))
            rain_item.setBrush(QBrush(QColor(52, 152, 219, 45)))
            pen_rain = QPen(QColor(52, 152, 219), 1, Qt.DotLine)
            pen_rain.setCosmetic(True)
            rain_item.setPen(pen_rain)
            self.scene.addItem(rain_item)

        # ---------- 4. 地下水位线 ----------
        if water_pts and len(water_pts) >= 2:
            water_path = QPainterPath()
            water_path.moveTo(water_pts[0][0], water_pts[0][1])
            for pt in water_pts[1:]:
                water_path.lineTo(pt[0], pt[1])
            water_item = QGraphicsPathItem(water_path)
            water_item.setBrush(QBrush(Qt.NoBrush))
            water_pen = QPen(QColor(41, 128, 185), 1.8, Qt.DashDotLine)
            water_pen.setCosmetic(True)
            water_item.setPen(water_pen)
            self.scene.addItem(water_item)

        # ---------- 5. 坡顶荷载 ----------
        if surcharge_loads:
            for x1, x2, q in surcharge_loads:
                if q > 0:
                    y1 = float(np.interp(x1, ground_x, ground_y)) + 0.8
                    y2 = float(np.interp(x2, ground_x, ground_y)) + 0.8
                    load_line = QGraphicsLineItem(x1, y1, x2, y2)
                    pen_load = QPen(QColor(192, 57, 43), 3.0)
                    pen_load.setCosmetic(True)
                    load_line.setPen(pen_load)
                    self.scene.addItem(load_line)

        # ---------- 6. 滑面 ----------
        SLIP_COLOR = QColor(192, 57, 43)

        if slip_surface is not None and getattr(slip_surface, "surface_type", "") == "polygonal":
            poly_path = QPainterPath()
            poly_path.moveTo(slip_surface.px[0], slip_surface.py[0])
            for px, py in zip(slip_surface.px[1:], slip_surface.py[1:]):
                poly_path.lineTo(px, py)
            poly_item = QGraphicsPathItem(poly_path)
            poly_item.setBrush(QBrush(Qt.NoBrush))
            poly_pen = QPen(SLIP_COLOR, 2.2, Qt.DashLine)
            poly_pen.setCosmetic(True)
            poly_item.setPen(poly_pen)
            self.scene.addItem(poly_item)
        else:
            if R is not None and R > 0:
                gx_arr = np.array(ground_x, dtype=float)
                gy_arr = np.array(ground_y, dtype=float)

                xs_samp = np.linspace(xc - R + 1e-4, xc + R - 1e-4, 1500)
                ys_circ = yc - np.sqrt(np.maximum(0.0, R ** 2 - (xs_samp - xc) ** 2))
                ys_grnd = np.interp(xs_samp, gx_arr, gy_arr)
                inside = np.where(ys_circ - ys_grnd < 0)[0]

                if len(inside) >= 2:
                    x_s = float(xs_samp[inside[0]])
                    x_e = float(xs_samp[inside[-1]])
                else:
                    x_s = float(xc - R)
                    x_e = float(xc + R)

                n_arc = 200
                arc_xs = np.linspace(x_s, x_e, n_arc)
                arc_ys = yc - np.sqrt(np.maximum(0.0, R ** 2 - (arc_xs - xc) ** 2))

                arc_path = QPainterPath()
                arc_path.moveTo(arc_xs[0], arc_ys[0])
                for ax, ay in zip(arc_xs[1:], arc_ys[1:]):
                    arc_path.lineTo(ax, ay)
                arc_item = QGraphicsPathItem(arc_path)
                arc_item.setBrush(QBrush(Qt.NoBrush))
                arc_pen = QPen(SLIP_COLOR, 2.2, Qt.DashLine)
                arc_pen.setCosmetic(True)
                arc_item.setPen(arc_pen)
                self.scene.addItem(arc_item)

                r_pen = QPen(SLIP_COLOR, 1.6, Qt.DashLine)
                r_pen.setCosmetic(True)
                line_l = QGraphicsLineItem(
                    QLineF(xc, yc, float(arc_xs[0]), float(arc_ys[0]))
                )
                line_l.setPen(r_pen)
                self.scene.addItem(line_l)
                line_r = QGraphicsLineItem(
                    QLineF(xc, yc, float(arc_xs[-1]), float(arc_ys[-1]))
                )
                line_r.setPen(r_pen)
                self.scene.addItem(line_r)

            cs = max(1.0, (R * 0.03) if R else 1.5)
            c_h = QGraphicsLineItem(xc - cs, yc, xc + cs, yc)
            c_v = QGraphicsLineItem(xc, yc - cs, xc, yc + cs)
            c_pen = QPen(SLIP_COLOR, 2.0)
            c_pen.setCosmetic(True)
            c_h.setPen(c_pen)
            c_v.setPen(c_pen)
            self.scene.addItem(c_h)
            self.scene.addItem(c_v)

        # ---------- 7. 支护构件 ----------
        if reinforcements:
            for r in reinforcements:
                kind = getattr(r, "kind", "")
                if not getattr(r, "enabled", True):
                    continue

                if kind == "anchor":
                    x_head = r.x_head
                    y_head = r.y_head
                    try:
                        x_tip, y_tip = r.get_tip_xy()
                    except Exception:
                        continue
                    line = QGraphicsLineItem(QLineF(x_head, y_head, x_tip, y_tip))
                    pen = QPen(QColor(142, 68, 173), 2.5)
                    pen.setCosmetic(True)
                    line.setPen(pen)
                    self.scene.addItem(line)

                    hs = 0.4
                    head = QGraphicsRectItem(x_head - hs, y_head - hs, 2 * hs, 2 * hs)
                    head.setBrush(QBrush(QColor(142, 68, 173)))
                    head.setPen(QPen(Qt.NoPen))
                    self.scene.addItem(head)

                elif kind == "pile":
                    x = r.x
                    line = QGraphicsLineItem(QLineF(x, r.top, x, r.bottom))
                    pen = QPen(QColor(44, 62, 80), 6.0)
                    pen.setCosmetic(True)
                    line.setPen(pen)
                    self.scene.addItem(line)

                    w = max(0.4, r.diameter * 0.5)
                    for y in (r.top, r.bottom):
                        cap = QGraphicsLineItem(QLineF(x - w, y, x + w, y))
                        cap_pen = QPen(QColor(44, 62, 80), 2.0)
                        cap_pen.setCosmetic(True)
                        cap.setPen(cap_pen)
                        self.scene.addItem(cap)

                elif kind == "wall":
                    x_back = r.x
                    x_ft = x_back + r.top_width
                    x_fb = x_back + r.bottom_width
                    wall_poly = QPolygonF([
                        QPointF(x_back, r.top),
                        QPointF(x_ft, r.top),
                        QPointF(x_fb, r.bottom),
                        QPointF(x_back, r.bottom),
                    ])
                    wall_item = QGraphicsPolygonItem(wall_poly)
                    wall_item.setBrush(QBrush(QColor(127, 140, 141, 180)))
                    wall_pen = QPen(QColor(44, 62, 80), 1.6)
                    wall_pen.setCosmetic(True)
                    wall_item.setPen(wall_pen)
                    self.scene.addItem(wall_item)

        # ---------- 8. 土条 (底面用真实滑面底高程) ----------
        if slices:
            gx_arr = np.asarray(ground_x, dtype=float)
            gy_arr = np.asarray(ground_y, dtype=float)
            for s in slices:
                xl = s.xm - 0.5 * s.b
                xr = s.xm + 0.5 * s.b
                yt_l = float(np.interp(xl, gx_arr, gy_arr))
                yt_r = float(np.interp(xr, gx_arr, gy_arr))

                if slip_surface is not None:
                    try:
                        yb_l = float(slip_surface.get_y_base(xl))
                        yb_r = float(slip_surface.get_y_base(xr))
                    except Exception:
                        yb_l = yb_r = s.y_base
                else:
                    yb_l = yb_r = s.y_base

                slice_poly = QPolygonF([
                    QPointF(xl, yb_l),
                    QPointF(xr, yb_r),
                    QPointF(xr, yt_r),
                    QPointF(xl, yt_l),
                ])
                slice_item = SliceGraphicsItem(s, slice_poly)
                self.scene.addItem(slice_item)

        # ---------- 9. 若仍在绘制, 重建预览 ----------
        if self._drawing_polygon and self._polygon_points:
            self._update_drawing_preview(self._preview_cursor)

        # ---------- 10. 收集吸附目标 ----------
        self._collect_snap_targets(ground_x, ground_y, layer_regions)

        # ---------- 11. 场景范围 ----------
        scene_min_x = min_x
        scene_max_x = max_x
        scene_min_y = min_y
        scene_max_y = max_y

        if slip_surface is None or getattr(slip_surface, "surface_type", "") != "polygonal":
            if R is not None and R > 0:
                scene_min_x = min(scene_min_x, xc - R - 2.0)
                scene_max_x = max(scene_max_x, xc + R + 2.0)
                scene_min_y = min(scene_min_y, yc - R - 2.0)
                scene_max_y = max(scene_max_y, yc + R + 2.0)

        if reinforcements:
            for r in reinforcements:
                if not getattr(r, "enabled", True):
                    continue
                kind = getattr(r, "kind", "")
                if kind == "anchor":
                    try:
                        x_tip, y_tip = r.get_tip_xy()
                        scene_min_x = min(scene_min_x, r.x_head, x_tip)
                        scene_max_x = max(scene_max_x, r.x_head, x_tip)
                        scene_min_y = min(scene_min_y, r.y_head, y_tip)
                        scene_max_y = max(scene_max_y, r.y_head, y_tip)
                    except Exception:
                        pass
                elif kind == "pile":
                    scene_min_x = min(scene_min_x, r.x)
                    scene_max_x = max(scene_max_x, r.x)
                    scene_min_y = min(scene_min_y, r.bottom)
                    scene_max_y = max(scene_max_y, r.top)
                elif kind == "wall":
                    scene_min_x = min(scene_min_x, r.x)
                    scene_max_x = max(scene_max_x, r.x + r.bottom_width)
                    scene_min_y = min(scene_min_y, r.bottom)
                    scene_max_y = max(scene_max_y, r.top)

        self.scene.setSceneRect(
            scene_min_x, scene_min_y,
            (scene_max_x - scene_min_x) * 1.05,
            (scene_max_y - scene_min_y) * 1.05,
        )

        # ---------- 12. 场景重建后重新应用选中状态 ----------
        self._apply_region_selection()

    # ==================================================================
    # 静态工具
    # ==================================================================
    @staticmethod
    def _pattern_for_region(pattern):
        patterns = {
            "solid": Qt.SolidPattern,
            "sand": Qt.Dense4Pattern,
            "clay": Qt.BDiagPattern,
            "gravel": Qt.CrossPattern,
            "rock": Qt.DiagCrossPattern,
        }
        return patterns.get(pattern, Qt.SolidPattern)

    def contextMenuEvent(self, event):
        """右键 → 绘制模式下结束绘制; 非绘制模式不弹菜单

        说明: Windows 平台上 QGraphicsView 的右键优先走 contextMenuEvent,
        覆写的 mousePressEvent 里的右键分支可能根本收不到。
        这里兜底处理, 保证右键一定能结束绘制。
        """
        if self._drawing_polygon:
            if self._drawing_mode == "polyline" and len(self._polygon_points) >= 2:
                self._finish_polyline_drawing()
            elif self._drawing_mode == "polygon" and len(self._polygon_points) >= 3:
                self._finish_polygon_drawing()
            else:
                self.cancel_polygon_drawing()
                self.drawing_status.emit("已取消绘制")
        event.accept()

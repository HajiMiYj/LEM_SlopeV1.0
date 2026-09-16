# -*- coding: utf-8 -*-
"""
专业 CAD/CAE 风格的可拆卸与浮动停靠窗口模块 (PyQt5 QDockWidget)
支持图标与文本一体化工程控件
"""
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QBrush, QColor, QDoubleValidator
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QProgressBar, QTreeWidget, QTreeWidgetItem, QSlider,
    QScrollArea, QFrame, QCheckBox, QMessageBox, QStyledItemDelegate, QLineEdit, QAbstractItemView,
    QInputDialog, QFileDialog, QListWidget
)

from core.materials import SoilMaterial, Distribution
from core.rainfall import RainfallTimeSeries
from core.slicing import Slice
from gui.dist_plot import DistributionPlotWidget
from gui.icons import get_icon
from gui.spatial_plot import SpatialFieldWidget, points_in_polygon


class NumericDelegate(QStyledItemDelegate):
    """只允许输入浮点数的单元格编辑器"""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator(-1e9, 1e9, 6, editor)
        validator.setNotation(QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor

    def setModelData(self, editor, model, index):
        text = editor.text().strip()
        try:
            val = float(text)
        except ValueError:
            return  # 非法输入不写入
        model.setData(index, f"{val:.3f}")


class GeometryDockWidget(QDockWidget):
    """几何与地层拓扑管理停靠窗

    数据模型:
      · data_ground  —— 地表轮廓线
      · data_water   —— 地下水位线
      · data_regions —— 土层面列表 (一组互不重叠的平面图形)
    """
    geometry_changed = pyqtSignal()
    polygon_draw_requested = pyqtSignal()
    strata_draw_requested = pyqtSignal()
    strata_validation_failed = pyqtSignal(str)
    regions_highlight_requested = pyqtSignal(list)
    view_reset_requested = pyqtSignal()

    TAG_GROUND = -100
    TAG_WATER = -101

    _PATTERN_COLORS = {
        "fill": "#e0d0a0", "clay": "#e8dcc8", "silt": "#d9d0b8",
        "sand": "#f0e2a8", "gravel": "#d0c8b0",
        "rock_strong": "#c8c8c8", "rock_medium": "#b8b8b8",
        "rock_fresh": "#a8a8a8", "solid": "#f3dfaa",
    }

    _ICON_SIZE = 20
    _BTN_SIZE = 32
    _REFRESH_DEBOUNCE_MS = 150
    _ADJACENCY_TOL = 0.05
    _ADJACENCY_DIST_TOL = 1e-3
    _MERGE_CLOSE_DELTA = 0.005

    _VERTEX_SYNC_TOL = 1e-6
    _REPLACE_OVERLAP_RATIO = 0.98
    _SLIVER_MIN_AREA = 0.05
    _SLIVER_MIN_EQ_WIDTH = 0.10
    _CUT_EXTEND = 5.0
    _SIMPLIFY_TOL = 0.01

    def __init__(self, parent=None):
        super().__init__("模型几何与地层分界", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._suspend_table_signal = False
        self._ground_dirty = False
        self._syncing_selection = False
        self._bottom_depth = 6.0  # ★ 新增
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(self._REFRESH_DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self._do_refresh)

        self._init_ui()

    # ==================================================================
    # 防抖
    # ==================================================================
    def _schedule_refresh(self):
        self._refresh_timer.start()

    def _emit_now(self):
        self._refresh_timer.stop()
        self.geometry_changed.emit()

    def _do_refresh(self):
        if self._ground_dirty:
            self._clip_regions_to_boundary()
            self._clip_water_to_ground()
            self._ground_dirty = False
        self.geometry_changed.emit()

    # ==================================================================
    # 建模深度
    # ==================================================================
    def get_bottom_depth(self) -> float:
        return float(self._bottom_depth)

    def _on_bottom_depth_changed(self, value: float):
        """深度变化 → 坡体边界伸缩 → 土层块跟随(仅裁剪, 不扩展)

        · 变浅: 土层块超出新边界 → clip 裁掉
        · 变深: 土层块在边界内 → clip 内部自动跳过, 土层块不动
        顺序: 先改边界(由 _bottom_depth 隐式生效), 再裁土层块。
        """
        self._bottom_depth = float(value)
        self._clip_regions_to_boundary()
        self._clip_water_to_ground()
        self._ground_dirty = False
        self.geometry_changed.emit()
        self.view_reset_requested.emit()

    # ==================================================================
    # 细长碎片判定
    # ==================================================================
    @classmethod
    def _is_sliver(cls, poly) -> bool:
        if poly is None or poly.is_empty:
            return True
        try:
            area = float(poly.area)
        except Exception:
            return True
        if area < cls._SLIVER_MIN_AREA:
            return True
        try:
            perim = float(poly.length)
        except Exception:
            return True
        if perim < 1e-9:
            return True
        return 4.0 * area / perim < cls._SLIVER_MIN_EQ_WIDTH

    # ==================================================================
    # 视图重置
    # ==================================================================
    def _reset_to_ground_view(self):
        self.tree.blockSignals(True)
        try:
            self.current_editing_key = "ground"
            self.tree.setCurrentItem(self.item_ground)
        finally:
            self.tree.blockSignals(False)

        self.lbl_table_title.setText("<b>地表轮廓线控制点 (X, Y 单位: 米):</b>")
        self._load_table_data(self.data_ground)
        self.combo_region_pattern.setEnabled(False)
        self.combo_region_material.setEnabled(False)
        self.slider_hatch_scale.setEnabled(False)
        self.lbl_hatch_scale_value.setEnabled(False)
        self.regions_highlight_requested.emit([])

    # ==================================================================
    # 地表线改动 → 裁剪土层
    # ==================================================================
    def _clip_regions_to_boundary(self):
        boundary = self._build_slope_boundary_polygon()
        if boundary is None or boundary.is_empty:
            return

        new_regions = []
        changed = False

        for r in self.data_regions:
            p = self._to_shapely_polygon(r.get("points", []), r.get("holes", []))
            if p is None or p.is_empty:
                new_regions.append(r)
                continue

            try:
                outside = p.difference(boundary)
                if outside.is_empty or outside.area < 1e-6:
                    new_regions.append(r)
                    continue
            except Exception:
                new_regions.append(r)
                continue

            try:
                clipped = p.intersection(boundary)
            except Exception:
                new_regions.append(r)
                continue

            if clipped.is_empty or clipped.area < 0.01:
                changed = True
                continue

            pieces = self._extract_polygons(clipped)
            pieces = [pc for pc in pieces if not self._is_sliver(pc)]
            if not pieces:
                changed = True
                continue

            if len(pieces) == 1:
                new_regions.append(self._polygon_to_region_dict(pieces[0], r))
            else:
                base_name = r.get("name", "土层面")
                for k, piece in enumerate(pieces):
                    new_region = self._polygon_to_region_dict(piece, r)
                    new_region["name"] = f"{base_name}-{k + 1}"
                    new_regions.append(new_region)

            changed = True

        if changed:
            for i, r in enumerate(new_regions):
                r["id"] = i + 1
            self.data_regions = new_regions
            self._refresh_region_tree(keep_selection=False)
            if self.current_editing_key.startswith("region_"):
                self._reset_to_ground_view()

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        lbl_tree = QLabel("几何实体与地层图层树")
        lbl_tree.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(lbl_tree)

        h_toolbar = QHBoxLayout()
        h_toolbar.setSpacing(4)
        h_toolbar.setContentsMargins(0, 0, 0, 0)

        btn_draw_region = self._make_icon_button(
            "draw_polygon",
            "绘制土层面\n\n画闭合多边形; 与已有面重叠的部分\n会从已有面中减去 (旧面消失)。",
            self.polygon_draw_requested.emit,
        )
        btn_draw_cut = self._make_icon_button(
            "draw_cut",
            "绘制切割线\n\n画折线; 所有被线完全贯穿的土层面\n会被切成两块, 未完全贯穿的保持原样。",
            self.strata_draw_requested.emit,
        )
        btn_fill_holes = self._make_icon_button(
            "fill_hole",
            "填充土层空洞\n\n找出坡体内未被任何土层面覆盖的区域,\n用相邻层的材料填上并合并。",
            self.fill_holes,
        )
        btn_merge = self._make_icon_button(
            "merge_layers",
            "合并土层\n\nCtrl 多选 ≥2 个边挨边的土层面,\n合并为一个, 继承指定主面的属性。",
            self.merge_selected_regions,
        )
        btn_del_entity = self._make_icon_button(
            "del_item", "删除选中的土层面", self._del_current_entity,
        )

        for b in (btn_draw_region, btn_draw_cut, btn_fill_holes, btn_merge, btn_del_entity):
            h_toolbar.addWidget(b)
        h_toolbar.addStretch()
        layout.addLayout(h_toolbar)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["图层实体名称", "控制点数", "土层类型"])
        self.tree.setColumnWidth(0, 150)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 80)
        self.tree.setSelectionBehavior(QTreeWidget.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.item_ground = QTreeWidgetItem(["地表轮廓线", "4", ""])
        self.item_water = QTreeWidgetItem(["地下水浸润线", "4", ""])
        self.item_regions_root = QTreeWidgetItem(["土层面", "0", ""])

        for it in (self.item_ground, self.item_water, self.item_regions_root):
            it.setTextAlignment(1, Qt.AlignCenter)
            it.setTextAlignment(2, Qt.AlignCenter)

        self.tree.addTopLevelItem(self.item_ground)
        self.tree.addTopLevelItem(self.item_water)
        self.tree.addTopLevelItem(self.item_regions_root)
        self.item_regions_root.setExpanded(True)

        self.tree.currentItemChanged.connect(self._on_tree_selection_changed)
        self.tree.itemSelectionChanged.connect(self._on_tree_multi_selection_changed)
        layout.addWidget(self.tree, stretch=3)

        # ---------------- ★ 建模范围 ----------------
        grp_depth = QGroupBox("建模范围")
        f_depth = QFormLayout(grp_depth)
        f_depth.setContentsMargins(6, 6, 6, 6)
        f_depth.setSpacing(4)

        self.spin_bottom_depth = QDoubleSpinBox()
        self.spin_bottom_depth.setRange(0.5, 300.0)
        self.spin_bottom_depth.setValue(6.0)
        self.spin_bottom_depth.setSingleStep(1.0)
        self.spin_bottom_depth.setDecimals(2)
        self.spin_bottom_depth.setSuffix(" m")
        self.spin_bottom_depth.setToolTip(
            "坡体底边 = 地表最低点 − 该值。\n"
            "影响: 坡体边界、默认土层大面、画布 Y 轴下界。\n"
            "调整后可能需要点击「填充」按钮补全底部区域。"
        )
        self.spin_bottom_depth.valueChanged.connect(self._on_bottom_depth_changed)
        f_depth.addRow("地表以下深度:", self.spin_bottom_depth)

        lbl_hint = QLabel("<i>坡体底边 = 地表最低点 − 该值</i>")
        lbl_hint.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        f_depth.addRow(lbl_hint)

        layout.addWidget(grp_depth)

        # ---------------- 吸附与容差 ----------------
        grp_snap = QGroupBox("吸附与容差")
        f_snap = QFormLayout(grp_snap)
        f_snap.setContentsMargins(6, 6, 6, 6)
        f_snap.setSpacing(4)

        self.spin_snap_tol = QDoubleSpinBox()
        self.spin_snap_tol.setRange(0.0, 5.0)
        self.spin_snap_tol.setValue(0.5)
        self.spin_snap_tol.setSingleStep(0.1)
        self.spin_snap_tol.setSuffix(" m")
        self.spin_snap_tol.valueChanged.connect(self._on_snap_settings_changed)
        f_snap.addRow("容差:", self.spin_snap_tol)

        self.combo_snap_mode = QComboBox()
        self.combo_snap_mode.addItem("不吸附", "none")
        self.combo_snap_mode.addItem("吸附点", "point")
        self.combo_snap_mode.addItem("吸附线", "line")
        self.combo_snap_mode.addItem("点 + 线", "both")
        self.combo_snap_mode.setCurrentIndex(3)
        self.combo_snap_mode.currentIndexChanged.connect(self._on_snap_settings_changed)
        f_snap.addRow("吸附模式:", self.combo_snap_mode)

        layout.addWidget(grp_snap)

        # ---------------- 面属性 ----------------
        grp_face = QGroupBox("土层面样式与材料")
        v_face = QVBoxLayout(grp_face)
        v_face.setContentsMargins(6, 6, 6, 6)
        v_face.setSpacing(4)

        h_pickers = QHBoxLayout()
        h_pickers.setSpacing(4)

        self.combo_region_pattern = QComboBox()
        self.combo_region_pattern.addItem("素填土 — 稀疏点", "fill")
        self.combo_region_pattern.addItem("黏土 (CL) — 45°斜线", "clay")
        self.combo_region_pattern.addItem("粉质黏土 (ML) — 点划线", "silt")
        self.combo_region_pattern.addItem("砂土 (SM/SP) — 点状", "sand")
        self.combo_region_pattern.addItem("砾石/卵石 — 交叉网格", "gravel")
        self.combo_region_pattern.addItem("强风化岩 — 交叉斜线", "rock_strong")
        self.combo_region_pattern.addItem("中风化岩 — 斜线", "rock_medium")
        self.combo_region_pattern.addItem("新鲜岩石 — 三角符号", "rock_fresh")
        self.combo_region_pattern.addItem("纯色填充", "solid")

        self.combo_region_material = QComboBox()
        self.combo_region_material.addItems(["材料 1"])

        h_pickers.addWidget(self.combo_region_pattern, 1)
        h_pickers.addWidget(self.combo_region_material, 1)
        v_face.addLayout(h_pickers)

        h_scale = QHBoxLayout()
        h_scale.setSpacing(6)
        h_scale.addWidget(QLabel("填充比例:"))
        self.slider_hatch_scale = QSlider(Qt.Horizontal)
        self.slider_hatch_scale.setRange(30, 400)
        self.slider_hatch_scale.setValue(100)
        self.slider_hatch_scale.valueChanged.connect(self._on_hatch_scale_changed)
        h_scale.addWidget(self.slider_hatch_scale, 1)
        self.lbl_hatch_scale_value = QLabel("100%")
        self.lbl_hatch_scale_value.setFixedWidth(48)
        h_scale.addWidget(self.lbl_hatch_scale_value)
        v_face.addLayout(h_scale)

        self.combo_region_pattern.setEnabled(False)
        self.combo_region_material.setEnabled(False)
        self.combo_region_pattern.currentIndexChanged.connect(self._on_style_changed)
        self.combo_region_material.currentIndexChanged.connect(self._on_material_changed)

        layout.addWidget(grp_face)

        # ---------------- 坐标表 ----------------
        self.lbl_table_title = QLabel("<b>地表轮廓线控制点 (X, Y 单位: 米):</b>")
        layout.addWidget(self.lbl_table_title)

        h_table_tools = QHBoxLayout()
        h_table_tools.setSpacing(4)
        btn_add_pt = self._make_icon_button("add_item", "添加坐标点", self._add_point)
        btn_del_pt = self._make_icon_button("del_item", "删除选中点", self._del_point)
        h_table_tools.addWidget(btn_add_pt)
        h_table_tools.addWidget(btn_del_pt)
        h_table_tools.addStretch()
        layout.addLayout(h_table_tools)

        self.tbl_coords = QTableWidget(4, 2)
        self.tbl_coords.setHorizontalHeaderLabels(["X 坐标 (m)", "Y 高程 (m)"])
        self.tbl_coords.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_coords.itemChanged.connect(self._on_table_cell_changed)
        self._numeric_delegate = NumericDelegate(self.tbl_coords)
        self.tbl_coords.setItemDelegate(self._numeric_delegate)
        layout.addWidget(self.tbl_coords, stretch=3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        self.setWidget(scroll)

        # ---------------- 默认数据 ----------------
        self.data_ground = [(0.0, 15.0), (20.0, 15.0), (35.0, 0.0), (60.0, 0.0)]
        self.data_water = []
        self.data_regions = [self._build_default_region()]
        self.current_editing_key = "ground"

        self._refresh_region_tree()
        self._clip_water_to_ground()
        self._load_table_data(self.data_ground)

    def _make_icon_button(self, icon_name, tooltip, slot):
        btn = QPushButton()
        btn.setIcon(get_icon(icon_name))
        btn.setIconSize(QSize(self._ICON_SIZE, self._ICON_SIZE))
        btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(slot)
        btn.setStyleSheet(
            "QPushButton{border:1px solid #bdc3c7;border-radius:4px;background:#fdfdfd;}"
            "QPushButton:hover{background:#ecf0f1;border-color:#3498db;}"
            "QPushButton:pressed{background:#d5dbdb;}"
        )
        return btn

    # ==================================================================
    # 坡体边界 / 默认面
    # ==================================================================
    def _build_slope_boundary_polygon(self):
        try:
            from shapely.geometry import Polygon
        except ImportError:
            return None
        if len(self.data_ground) < 2:
            return None

        gx = [p[0] for p in self.data_ground]
        gy = [p[1] for p in self.data_ground]
        bottom = min(gy) - self._bottom_depth  # ★ 用实例属性
        x_left, x_right = gx[0], gx[-1]

        points = [(x_left, bottom)]
        points.extend(zip(gx, gy))
        points.append((x_right, bottom))

        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    def _build_default_region(self):
        gx = [p[0] for p in self.data_ground]
        gy = [p[1] for p in self.data_ground]
        bottom = min(gy) - self._bottom_depth  # ★ 用实例属性
        x_left, x_right = gx[0], gx[-1]

        points = [(x_left, bottom)]
        for x, y in zip(gx, gy):
            points.append((x, y))
        points.append((x_right, bottom))

        return {
            "id": 1, "name": "土层面 1", "points": points, "holes": [],
            "material_index": 0,
            "color": self._PATTERN_COLORS["solid"],
            "pattern": "solid",
            "hatch_scale_pct": 100,
        }

    # ==================================================================
    # 图层树选择
    # ==================================================================
    def _on_tree_multi_selection_changed(self):
        if self._syncing_selection:
            return

        selected = [it for it in self.tree.selectedItems()
                    if it.parent() is self.item_regions_root]
        if not selected:
            self.regions_highlight_requested.emit([])
            return
        indices = sorted(self.item_regions_root.indexOfChild(it)
                         for it in selected)
        self.regions_highlight_requested.emit(indices)

    def clamp_material_indices(self, n_materials: int):
        """把越界的 material_index 重定向到 0"""
        if n_materials <= 0:
            return
        changed = False
        for r in self.data_regions:
            mi = int(r.get("material_index", 0))
            if mi >= n_materials or mi < 0:
                r["material_index"] = 0
                changed = True
        if changed:
            self._refresh_region_tree(keep_selection=True)
            self._emit_now()

    def set_selected_regions(self, indices):
        """由画布调用: 同步选中状态到图层树"""
        self._syncing_selection = True
        try:
            idx_set = set(int(i) for i in (indices or []) if int(i) >= 0)
            self.item_ground.setSelected(False)
            self.item_water.setSelected(False)
            for i in range(self.item_regions_root.childCount()):
                child = self.item_regions_root.child(i)
                child.setSelected(i in idx_set)
        finally:
            self._syncing_selection = False

    def _on_tree_selection_changed(self, current, previous):
        if not current:
            return

        if self._is_editing_key_consistent(previous):
            self._save_current_table_data()

        if current is self.item_ground:
            self.current_editing_key = "ground"
            self.lbl_table_title.setText("<b>地表轮廓线控制点 (X, Y 单位: 米):</b>")
            self._load_table_data(self.data_ground)
            self.combo_region_pattern.setEnabled(False)
            self.combo_region_material.setEnabled(False)
            self.slider_hatch_scale.setEnabled(False)
            self.lbl_hatch_scale_value.setEnabled(False)

        elif current is self.item_water:
            self.current_editing_key = "water"
            self.lbl_table_title.setText("<b>地下水浸润线控制点 (X, Y 单位: 米):</b>")
            self._load_table_data(self.data_water)
            self.combo_region_pattern.setEnabled(False)
            self.combo_region_material.setEnabled(False)
            self.slider_hatch_scale.setEnabled(False)
            self.lbl_hatch_scale_value.setEnabled(False)


        elif current.parent() is self.item_regions_root:
            idx = self.item_regions_root.indexOfChild(current)
            self.current_editing_key = f"region_{idx}"
            if 0 <= idx < len(self.data_regions):
                region = self.data_regions[idx]
                self.lbl_table_title.setText(f"<b>{region['name']} 外环顶点:</b>")
                self._load_table_data(region.get("points", []))
                self.combo_region_pattern.setEnabled(True)
                self.combo_region_material.setEnabled(True)
                self.slider_hatch_scale.setEnabled(True)
                self.lbl_hatch_scale_value.setEnabled(True)
                self.combo_region_pattern.blockSignals(True)
                self.combo_region_material.blockSignals(True)
                self.slider_hatch_scale.blockSignals(True)
                pat = region.get("pattern", "solid")
                for i in range(self.combo_region_pattern.count()):
                    if self.combo_region_pattern.itemData(i) == pat:
                        self.combo_region_pattern.setCurrentIndex(i)
                        break
                mat_idx = int(region.get("material_index", 0))
                if 0 <= mat_idx < self.combo_region_material.count():
                    self.combo_region_material.setCurrentIndex(mat_idx)
                pct = int(region.get("hatch_scale_pct", 100))
                pct = max(self.slider_hatch_scale.minimum(),
                          min(self.slider_hatch_scale.maximum(), pct))
                self.slider_hatch_scale.setValue(pct)
                self.lbl_hatch_scale_value.setText(f"{pct}%")
                self.combo_region_pattern.blockSignals(False)
                self.combo_region_material.blockSignals(False)
                self.slider_hatch_scale.blockSignals(False)

    def _is_editing_key_consistent(self, item) -> bool:
        if item is None:
            return True
        key = self.current_editing_key
        if key == "ground":
            return item is self.item_ground
        if key == "water":
            return item is self.item_water
        if key.startswith("region_"):
            if item.parent() is not self.item_regions_root:
                return False
            idx = self.item_regions_root.indexOfChild(item)
            try:
                key_idx = int(key.split("_")[1])
            except (ValueError, IndexError):
                return False
            return (idx == key_idx) and (0 <= key_idx < len(self.data_regions))
        return False

    # ==================================================================
    # 表格 <-> 数据
    # ==================================================================
    def _save_current_table_data(self):
        if self._suspend_table_signal:
            return

        pts = []
        for r in range(self.tbl_coords.rowCount()):
            ix = self.tbl_coords.item(r, 0)
            iy = self.tbl_coords.item(r, 1)
            if ix and iy:
                try:
                    pts.append((float(ix.text()), float(iy.text())))
                except ValueError:
                    pass

        if self.current_editing_key == "ground":
            if len(pts) >= 2:
                pts = sorted(pts, key=lambda p: p[0])
                self.data_ground = pts
                self.item_ground.setText(1, str(len(pts)))
        elif self.current_editing_key == "water":
            if len(pts) >= 2:
                pts = sorted(pts, key=lambda p: p[0])
                self.data_water = pts
                self.item_water.setText(1, str(len(pts)))
        elif self.current_editing_key.startswith("region_"):
            try:
                idx = int(self.current_editing_key.split("_")[1])
            except (ValueError, IndexError):
                return
            if 0 <= idx < len(self.data_regions):
                if len(pts) >= 3:
                    self.data_regions[idx]["points"] = pts

    def _load_table_data(self, pts):
        if pts is None:
            pts = []
        self._suspend_table_signal = True
        try:
            self.tbl_coords.setRowCount(len(pts))
            for r, (x, y) in enumerate(pts):
                self.tbl_coords.setItem(r, 0, QTableWidgetItem(f"{float(x):.3f}"))
                self.tbl_coords.setItem(r, 1, QTableWidgetItem(f"{float(y):.3f}"))
        finally:
            self._suspend_table_signal = False

    def _add_point(self):
        row = self.tbl_coords.rowCount()
        self.tbl_coords.insertRow(row)
        self.tbl_coords.setItem(row, 0, QTableWidgetItem("0.0"))
        self.tbl_coords.setItem(row, 1, QTableWidgetItem("0.0"))
        self._save_current_table_data()
        if self.current_editing_key == "ground":
            self._ground_dirty = True
        self._schedule_refresh()

    def _del_point(self):
        r = self.tbl_coords.currentRow()
        min_rows = 3 if self.current_editing_key.startswith("region_") else 2
        if r >= 0 and self.tbl_coords.rowCount() > min_rows:
            self.tbl_coords.removeRow(r)
            self._save_current_table_data()
            if self.current_editing_key == "ground":
                self._ground_dirty = True
            self._schedule_refresh()

    def _on_table_cell_changed(self, item):
        if self._suspend_table_signal:
            return

        if self.current_editing_key == "ground":
            self._save_current_table_data()
            self._ground_dirty = True
            self._schedule_refresh()
            return
        if self.current_editing_key == "water":
            self._save_current_table_data()
            self._clip_water_to_ground()  # ★ 加这行
            self._load_table_data(self.data_water)  # 刷新表格显示新值
            self._schedule_refresh()
            return

        if not self.current_editing_key.startswith("region_"):
            return

        try:
            idx = int(self.current_editing_key.split("_")[1])
        except (ValueError, IndexError):
            return
        if not (0 <= idx < len(self.data_regions)):
            return

        row, col = item.row(), item.column()
        try:
            new_val = float(item.text())
        except ValueError:
            return

        region = self.data_regions[idx]
        pts = region.get("points", [])
        if row >= len(pts):
            return

        old_x, old_y = pts[row]
        new_x, new_y = (new_val, old_y) if col == 0 else (old_x, new_val)

        if abs(new_x - old_x) < 1e-9 and abs(new_y - old_y) < 1e-9:
            return

        TOL = self._VERTEX_SYNC_TOL
        for r in self.data_regions:
            r_pts = r.get("points", [])
            r["points"] = [
                (new_x, new_y) if (abs(px - old_x) < TOL and abs(py - old_y) < TOL)
                else (px, py)
                for (px, py) in r_pts
            ]
        self._clip_water_to_ground()

        self._refresh_region_tree(keep_selection=True)
        self._schedule_refresh()

    def _on_hatch_scale_changed(self, value: int):
        self.lbl_hatch_scale_value.setText(f"{value}%")
        idx = self._get_current_region_index()
        if idx < 0:
            return
        self.data_regions[idx]["hatch_scale_pct"] = int(value)
        self._emit_now()

    def _clip_water_to_ground(self):
        """水位线必须在坡体边界内部 (地表以下, 坡底以上)

        规则:
          1. 初始 (空水位) → 沿地表每个折点下移 _bottom_depth (与地表同形)
          2. x 范围不足 → 两端补默认点
          3. 密集采样 + clamp 到 [坡底, 地表] → 保证不穿出边界
          4. 去共线点保持简洁
        """
        if len(self.data_ground) < 2:
            self.data_water = []
            self.item_water.setText(1, "0")
            return

        gx = [p[0] for p in self.data_ground]
        gy = [p[1] for p in self.data_ground]
        gx_min, gx_max = min(gx), max(gx)
        gx_arr = np.array(gx, dtype=float)
        gy_arr = np.array(gy, dtype=float)

        def _ground_y(x):
            return float(np.interp(x, gx_arr, gy_arr))

        def _bottom_y(x):
            return _ground_y(x) - self._bottom_depth

        def _default_water():
            # ★ 与地表同形: 每个地表折点下移 _bottom_depth
            return [(gx[i], gy[i] - self._bottom_depth) for i in range(len(gx))]

        # ---- 空水位 → 默认 ----
        if not self.data_water:
            self.data_water = _default_water()
            self.item_water.setText(1, str(len(self.data_water)))
            return

        # ---- 排序 + 裁 x 超出地表范围的点 ----
        water = sorted(self.data_water, key=lambda p: p[0])
        water = [(x, y) for (x, y) in water
                 if gx_min - 1e-6 <= x <= gx_max + 1e-6]

        if not water:
            self.data_water = _default_water()
            self.item_water.setText(1, str(len(self.data_water)))
            return

        # ---- 左右端补默认点 ----
        if water[0][0] > gx_min + 1e-6:
            water.insert(0, (gx_min, _ground_y(gx_min) - self._bottom_depth))
        if water[-1][0] < gx_max - 1e-6:
            water.append((gx_max, _ground_y(gx_max) - self._bottom_depth))

        # ---- ★ 密集采样 + clamp 到 [坡底, 地表] ----
        wx_arr = np.array([p[0] for p in water], dtype=float)
        wy_arr = np.array([p[1] for p in water], dtype=float)
        n_samples = max(80, int((gx_max - gx_min) * 4))
        xs = np.linspace(gx_min, gx_max, n_samples)

        sampled = []
        for x in xs:
            y_w = float(np.interp(x, wx_arr, wy_arr))
            y_g = _ground_y(x)
            y_b = y_g - self._bottom_depth
            y = min(y_w, y_g)  # 不高于地表 (相切允许)
            y = max(y, y_b)  # 不低于坡底 (相切允许)
            sampled.append((float(x), float(y)))

        # ---- 去共线点 ----
        water_clean = self._remove_collinear(sampled, tol=1e-3)

        if len(water_clean) < 2:
            water_clean = _default_water()

        self.data_water = water_clean
        self.item_water.setText(1, str(len(self.data_water)))

    @staticmethod
    def _remove_collinear(points, tol=1e-3):
        """去除共线冗余点 (只在真正拐弯处保留顶点)"""
        if len(points) < 3:
            return list(points)
        result = [points[0]]
        for i in range(1, len(points) - 1):
            p0 = result[-1]
            p1 = points[i]
            p2 = points[i + 1]
            dx = p2[0] - p0[0]
            dy = p2[1] - p0[1]
            L = (dx * dx + dy * dy) ** 0.5
            if L < 1e-9:
                continue
            dist = abs((p1[0] - p0[0]) * dy - (p1[1] - p0[1]) * dx) / L
            if dist > tol:
                result.append(p1)
        result.append(points[-1])
        return result

    # ==================================================================
    # 样式 / 材料
    # ==================================================================
    def _get_current_region_index(self):
        curr = self.tree.currentItem()
        if curr is None or curr.parent() is not self.item_regions_root:
            return -1
        idx = self.item_regions_root.indexOfChild(curr)
        if 0 <= idx < len(self.data_regions):
            return idx
        return -1

    def _on_style_changed(self, _=None):
        idx = self._get_current_region_index()
        if idx < 0:
            return
        pattern = self.combo_region_pattern.currentData() or "solid"
        color = self._PATTERN_COLORS.get(pattern, "#f3dfaa")
        mat_idx = self.combo_region_material.currentIndex()
        if mat_idx < 0:
            mat_idx = 0
        pct = int(self.slider_hatch_scale.value())
        self.data_regions[idx]["pattern"] = pattern
        self.data_regions[idx]["color"] = color
        self._refresh_region_tree()
        self._emit_now()

    def _on_material_changed(self, _=None):
        idx = self._get_current_region_index()
        if idx < 0:
            return
        self.data_regions[idx]["material_index"] = self.combo_region_material.currentIndex()
        self._refresh_region_tree()
        self._emit_now()

    # ==================================================================
    # 吸附设置
    # ==================================================================
    def get_snap_tolerance(self) -> float:
        return float(self.spin_snap_tol.value())

    def get_snap_mode(self) -> str:
        return self.combo_snap_mode.currentData() or "none"

    # def get_hatch_scale(self) -> float:
    #     return self.slider_hatch_scale.value() / 100.0

    def _on_snap_settings_changed(self, *_):
        self._emit_now()

    # ==================================================================
    # shapely 工具
    # ==================================================================
    @staticmethod
    def _extract_polygons(geom):
        if geom is None or geom.is_empty:
            return []
        gt = geom.geom_type
        if gt == "Polygon":
            return [geom]
        if gt == "MultiPolygon":
            return list(geom.geoms)
        if gt == "GeometryCollection":
            out = []
            for g in geom.geoms:
                out.extend(GeometryDockWidget._extract_polygons(g))
            return out
        return []

    @staticmethod
    def _to_shapely_polygon(points, holes=None):
        try:
            from shapely.geometry import Polygon
            shell = [(float(x), float(y)) for x, y in points]
            rings = None
            if holes:
                rings = [[(float(x), float(y)) for x, y in h]
                         for h in holes if len(h) >= 3] or None
            p = Polygon(shell=shell, holes=rings)
            if not p.is_valid:
                p = p.buffer(0)
            return p
        except Exception:
            return None

    @staticmethod
    def _close_geometry(geom, delta: float):
        if geom is None or geom.is_empty:
            return geom
        try:
            grown = geom.buffer(delta, join_style=1, cap_style=1)
            shrunk = grown.buffer(-delta, join_style=1, cap_style=1)
            if shrunk.is_empty or not shrunk.is_valid:
                return geom
            return shrunk
        except Exception:
            return geom

    def _is_adjacent(self, pa, pb) -> bool:
        TOL = self._ADJACENCY_DIST_TOL
        try:
            if pa.distance(pb) > TOL:
                return False
        except Exception:
            return False
        try:
            a_bound_buf = pa.boundary.buffer(TOL)
            inter = a_bound_buf.intersection(pb.boundary)
        except Exception:
            return False
        if inter.is_empty:
            return False
        if inter.geom_type in ("LineString", "MultiLineString"):
            return inter.length > TOL
        return False

    def _contact_length(self, pa, pb) -> float:
        TOL = self._ADJACENCY_DIST_TOL
        try:
            if pa.distance(pb) > TOL:
                return 0.0
        except Exception:
            return 0.0
        try:
            a_bound_buf = pa.boundary.buffer(TOL)
            inter = a_bound_buf.intersection(pb.boundary)
        except Exception:
            return 0.0
        if inter.is_empty:
            return 0.0
        if inter.geom_type in ("LineString", "MultiLineString"):
            return float(inter.length)
        return 0.0

    @staticmethod
    def _dedup_coords(coords, tol: float = 1e-4):
        if not coords:
            return []
        out = [coords[0]]
        for p in coords[1:]:
            px, py = float(p[0]), float(p[1])
            lx, ly = out[-1]
            if abs(px - lx) > tol or abs(py - ly) > tol:
                out.append((px, py))
        if len(out) >= 2:
            lx, ly = out[-1]
            fx, fy = out[0]
            if abs(lx - fx) < tol and abs(ly - fy) < tol:
                out.pop()
        return out

    def _polygon_to_region_dict(self, poly, template):
        """从 shapely Polygon 生成 region dict, 属性继承 template"""
        new_region = dict(template)

        # 几何简化: 消除 buffer / union 产生的冗余点
        try:
            simplified = poly.simplify(self._SIMPLIFY_TOL, preserve_topology=True)
            if (not simplified.is_empty
                    and simplified.is_valid
                    and simplified.geom_type == "Polygon"):
                poly = simplified
        except Exception:
            pass

        ext_coords = list(poly.exterior.coords)[:-1]
        new_region["points"] = GeometryDockWidget._dedup_coords(ext_coords)

        holes = []
        for inner in poly.interiors:
            h_coords = list(inner.coords)[:-1]
            h_dedup = GeometryDockWidget._dedup_coords(h_coords)
            if len(h_dedup) >= 3:
                holes.append(h_dedup)
        new_region["holes"] = holes

        return new_region

    @staticmethod
    def _extend_polyline_ends(pts, extend: float):
        if len(pts) < 2 or extend <= 0:
            return list(pts)

        import math
        p0, p1 = pts[0], pts[1]
        pn, pnm1 = pts[-1], pts[-2]

        def _ext(p, q, d):
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            L = math.hypot(dx, dy)
            if L < 1e-9:
                return p
            return (p[0] + dx / L * d, p[1] + dy / L * d)

        return [_ext(p0, p1, extend)] + list(pts[1:-1]) + [_ext(pn, pnm1, extend)]

    # ==================================================================
    # 添加新面
    # ==================================================================
    def add_region(self, points):
        if len(points) < 3:
            self.strata_validation_failed.emit("土层面至少需要 3 个点。")
            return

        self._save_current_table_data()

        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except ImportError:
            self.strata_validation_failed.emit("缺少 shapely 库。")
            return

        boundary = self._build_slope_boundary_polygon()
        if boundary is None:
            self.strata_validation_failed.emit("坡体边界无效。")
            return

        try:
            user_poly = Polygon([(float(x), float(y)) for x, y in points])
            if not user_poly.is_valid:
                user_poly = user_poly.buffer(0)

            clipped = user_poly.intersection(boundary)
            if clipped.is_empty or clipped.area < 0.01:
                self.strata_validation_failed.emit("绘制的面与坡体无交集。")
                return

            new_polys = self._extract_polygons(clipped)
            new_polys = [p for p in new_polys if not self._is_sliver(p)]
            if not new_polys:
                self.strata_validation_failed.emit("裁剪后无有效区域。")
                return

            new_mask = unary_union(new_polys)

            updated_existing = []
            replaced_count = 0
            for r in self.data_regions:
                ep = self._to_shapely_polygon(r.get("points", []), r.get("holes", []))
                if ep is None or ep.is_empty:
                    continue
                if not ep.intersects(new_mask):
                    updated_existing.append(r)
                    continue

                try:
                    inter_area = new_mask.intersection(ep).area
                    overlap_ratio = inter_area / max(ep.area, 1e-9)
                except Exception:
                    overlap_ratio = 0.0

                if overlap_ratio > self._REPLACE_OVERLAP_RATIO:
                    replaced_count += 1
                    continue

                diff = ep.difference(new_mask)
                if diff.is_empty or diff.area < 0.01:
                    continue

                for piece in self._extract_polygons(diff):
                    if piece.is_empty or piece.area < 0.01:
                        continue
                    if self._is_sliver(piece):
                        continue
                    updated_existing.append(self._polygon_to_region_dict(piece, r))

            self.data_regions = updated_existing

            pattern = self.combo_region_pattern.currentData() or "solid"
            color = self._PATTERN_COLORS.get(pattern, "#f3dfaa")
            mat_idx = self.combo_region_material.currentIndex()
            if mat_idx < 0:
                mat_idx = 0
            pct = int(self.slider_hatch_scale.value())
            pct = max(self.slider_hatch_scale.minimum(),
                      min(self.slider_hatch_scale.maximum(), pct))

            added = 0
            for poly in new_polys:
                if poly.is_empty or poly.area < 0.01:
                    continue
                new_id = len(self.data_regions) + 1
                region = {
                    "id": new_id,
                    "name": f"土层面 {new_id}",
                    "points": [(float(x), float(y))
                               for x, y in list(poly.exterior.coords)[:-1]],
                    "holes": [[(float(x), float(y)) for x, y in list(inner.coords)[:-1]]
                              for inner in poly.interiors],
                    "material_index": mat_idx,
                    "color": color,
                    "pattern": pattern,
                    "hatch_scale_pct": pct,
                }
                self.data_regions.append(region)
                added += 1

            for i, r in enumerate(self.data_regions):
                r["id"] = i + 1
                r["name"] = f"土层面 {i + 1}"

            self._refresh_region_tree(keep_selection=False)
            self._reset_to_ground_view()
            self._emit_now()

            msg = f"已添加 {added} 个新面"
            if replaced_count > 0:
                msg += f", 替换 {replaced_count} 个重叠旧面"
            msg += "。"
            self.strata_validation_failed.emit(msg)
        except Exception as e:
            self.strata_validation_failed.emit(f"添加失败: {e}")

    # ==================================================================
    # 切割
    # ==================================================================
    @staticmethod
    def _try_cut_polygon(poly, line):
        """用 line 切割 poly, 返回 (ok, pieces)"""
        try:
            from shapely.ops import split
            result = split(poly, line)
        except Exception:
            return False, []

        pieces = []
        for g in getattr(result, "geoms", []):
            try:
                if g.geom_type == "Polygon" and g.area >= 0.01:
                    pieces.append(g)
            except Exception:
                continue

        if len(pieces) < 2:
            return False, []
        return True, pieces

    def apply_cut_line(self, line_points):
        if len(line_points) < 2:
            self.strata_validation_failed.emit("切割线至少需要 2 个点。")
            return

        try:
            from shapely.geometry import LineString
        except ImportError:
            self.strata_validation_failed.emit("缺少 shapely 库。")
            return

        self._save_current_table_data()

        pts = [(float(x), float(y)) for x, y in line_points]
        if len(pts) < 2:
            self.strata_validation_failed.emit("切割线至少需要 2 个点。")
            return

        extended_pts = self._extend_polyline_ends(pts, self._CUT_EXTEND)
        cut_line = LineString(extended_pts)

        new_regions = []
        cut_count = 0
        skipped_count = 0

        for region in self.data_regions:
            poly = self._to_shapely_polygon(region.get("points", []),
                                            region.get("holes", []))
            if poly is None or poly.is_empty:
                new_regions.append(region)
                continue

            ok, valid_pieces = self._try_cut_polygon(poly, cut_line)
            if not ok:
                new_regions.append(region)
                skipped_count += 1
                continue

            valid_pieces = [p for p in valid_pieces if not self._is_sliver(p)]
            if len(valid_pieces) < 2:
                new_regions.append(region)
                skipped_count += 1
                continue

            cut_count += 1
            for piece in valid_pieces:
                new_regions.append(self._polygon_to_region_dict(piece, region))

        for i, r in enumerate(new_regions):
            r["id"] = i + 1
            r["name"] = f"土层面 {i + 1}"

        self.data_regions = new_regions
        self._refresh_region_tree(keep_selection=False)
        self._reset_to_ground_view()
        self._emit_now()

        if cut_count == 0:
            self.strata_validation_failed.emit(
                f"切割线未能贯穿任何土层面 (共 {skipped_count} 个面保持原样)。\n"
                f"提示: 请让线的两端都延伸到坡体外部。"
            )
        else:
            msg = f"切割完成: {cut_count} 个面被切开"
            if skipped_count > 0:
                msg += f", {skipped_count} 个面未被贯穿被跳过"
            msg += f"; 当前共 {len(self.data_regions)} 个土层面"
            self.strata_validation_failed.emit(msg)

    # ==================================================================
    # 合并
    # ==================================================================
    def merge_selected_regions(self):
        try:
            from shapely.ops import unary_union
        except ImportError:
            self.strata_validation_failed.emit("缺少 shapely 库。")
            return

        selected_items = self.tree.selectedItems()
        indices = sorted({self.item_regions_root.indexOfChild(it)
                          for it in selected_items
                          if it.parent() is self.item_regions_root})
        indices = [i for i in indices if 0 <= i < len(self.data_regions)]

        if len(indices) < 2:
            self.strata_validation_failed.emit(
                "请按住 Ctrl 选中至少 2 个土层面, 再点合并。"
            )
            return

        self._save_current_table_data()

        polys = {}
        for i in indices:
            p = self._to_shapely_polygon(
                self.data_regions[i].get("points", []),
                self.data_regions[i].get("holes", []))
            if p is None or p.is_empty:
                self.strata_validation_failed.emit(f"面 #{i + 1} 几何无效。")
                return
            polys[i] = p

        adjacency = {i: set() for i in indices}
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if self._is_adjacent(polys[i], polys[j]):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        visited = set()
        stack = [indices[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node] - visited)

        if visited != set(indices):
            self.strata_validation_failed.emit(
                "选中的面之间不连通 (必须边挨边), 无法合并。"
            )
            return

        names = [self.data_regions[i].get("name", f"土层面 {i + 1}") for i in indices]
        chosen, ok = QInputDialog.getItem(
            self, "选择主土层",
            "新面将继承哪个面的属性 (材料 / 填充样式 / 名称)?",
            names, 0, False)
        if not ok:
            return

        master_idx = indices[names.index(chosen)]
        master_region = self.data_regions[master_idx]

        try:
            merged_geom = unary_union([polys[i] for i in indices])
        except Exception as e:
            self.strata_validation_failed.emit(f"合并失败: {e}")
            return

        merged_geom = self._close_geometry(merged_geom, self._MERGE_CLOSE_DELTA)

        merged_polys = self._extract_polygons(merged_geom)
        final_regions = [self._polygon_to_region_dict(p, master_region)
                         for p in merged_polys
                         if not p.is_empty and p.area >= 0.01
                         and not self._is_sliver(p)]

        if not final_regions:
            self.strata_validation_failed.emit(
                "合并后只剩下碎片, 已取消操作 (原面保留)。"
            )
            return

        new_data = []
        for i, r in enumerate(self.data_regions):
            if i == master_idx:
                new_data.extend(final_regions)
            elif i in indices:
                continue
            else:
                new_data.append(r)

        for k, r in enumerate(new_data):
            r["id"] = k + 1

        self.data_regions = new_data

        self._refresh_region_tree(keep_selection=False)
        self._reset_to_ground_view()
        self._emit_now()
        self.strata_validation_failed.emit(
            f"已合并 {len(indices)} 个土层面, 当前共 {len(self.data_regions)} 个。"
        )

    # ==================================================================
    # 填充空洞
    # ==================================================================
    def fill_holes(self):
        try:
            from shapely.ops import unary_union
        except ImportError:
            self.strata_validation_failed.emit("缺少 shapely 库。")
            return

        try:
            self._ground_dirty = False

            boundary = self._build_slope_boundary_polygon()
            if boundary is None or boundary.is_empty:
                self.strata_validation_failed.emit("坡体边界无效。")
                return

            boundary_polys = self._extract_polygons(boundary)
            if not boundary_polys:
                return
            boundary = max(boundary_polys, key=lambda p: p.area)

            region_polys = []
            for i, r in enumerate(self.data_regions):
                p = self._to_shapely_polygon(r.get("points", []),
                                             r.get("holes", []))
                if p is not None and not p.is_empty and p.area >= 0.001:
                    region_polys.append((i, p))

            if not region_polys:
                self.strata_validation_failed.emit("没有有效的土层面。")
                return

            region_polys_by_idx = {i: p for i, p in region_polys}

            covered = unary_union([p for _, p in region_polys])
            if not covered.is_valid:
                covered = covered.buffer(0)
            covered = self._close_geometry(covered, self._MERGE_CLOSE_DELTA)

            holes = boundary.difference(covered)
            if holes is None or holes.is_empty or holes.area < 0.01:
                self.strata_validation_failed.emit("没有发现土层空洞。")
                return

            hole_polys = self._extract_polygons(holes)

            pattern = self.combo_region_pattern.currentData() or "solid"
            color = self._PATTERN_COLORS.get(pattern, "#f3dfaa")
            mat_idx = self.combo_region_material.currentIndex()
            if mat_idx < 0:
                mat_idx = 0

            extra_shapes = {}
            appended = []
            skipped = 0

            for hp in hole_polys:
                if hp is None or hp.is_empty or not hp.is_valid:
                    skipped += 1
                    continue
                if hp.area < 0.01:
                    skipped += 1
                    continue
                if self._is_sliver(hp):
                    skipped += 1
                    continue

                exterior = list(hp.exterior.coords)[:-1]
                if len(exterior) < 3:
                    skipped += 1
                    continue

                best_idx = -1
                best_len = 0.0
                for i, p in region_polys:
                    L = self._contact_length(hp, p)
                    if L > best_len:
                        best_len = L
                        best_idx = i

                if best_idx >= 0 and best_len > self._ADJACENCY_DIST_TOL:
                    extra_shapes.setdefault(best_idx, []).append(hp)
                    continue

                clean_outer = []
                bad = False
                for x, y in exterior:
                    if not (np.isfinite(x) and np.isfinite(y)):
                        bad = True
                        break
                    clean_outer.append((float(x), float(y)))
                if bad or len(clean_outer) < 3:
                    skipped += 1
                    continue

                clean_holes = []
                for inner in hp.interiors:
                    inner_pts = list(inner.coords)[:-1]
                    if len(inner_pts) < 3:
                        continue
                    ok = True
                    cpts = []
                    for x, y in inner_pts:
                        if not (np.isfinite(x) and np.isfinite(y)):
                            ok = False
                            break
                        cpts.append((float(x), float(y)))
                    if ok and len(cpts) >= 3:
                        clean_holes.append(cpts)

                appended.append({
                    "points": clean_outer,
                    "holes": clean_holes,
                    "material_index": mat_idx,
                    "color": color,
                    "pattern": pattern,
                })

            merged_count = 0
            for idx, extra_list in extra_shapes.items():
                original = region_polys_by_idx.get(idx)
                if original is None:
                    continue
                try:
                    combined = unary_union([original] + extra_list)
                except Exception:
                    continue

                combined = self._close_geometry(combined, self._MERGE_CLOSE_DELTA)
                pieces = [
                    p for p in self._extract_polygons(combined)
                    if not p.is_empty
                       and p.area >= 0.01
                       and not self._is_sliver(p)
                ]
                if not pieces:
                    continue

                biggest = max(pieces, key=lambda p: p.area)
                master = self.data_regions[idx]
                self.data_regions[idx] = self._polygon_to_region_dict(biggest, master)
                merged_count += 1

            appended_count = 0
            for item in appended:
                new_id = len(self.data_regions) + 1
                self.data_regions.append({
                    "id": new_id,
                    "name": f"土层面 {new_id}",
                    "points": item["points"],
                    "holes": item["holes"],
                    "material_index": item["material_index"],
                    "color": item["color"],
                    "pattern": item["pattern"],
                })
                appended_count += 1

            if merged_count == 0 and appended_count == 0:
                self.strata_validation_failed.emit(
                    f"未填充任何区域 (跳过 {skipped} 个退化几何)。"
                )
                return

            for i, r in enumerate(self.data_regions):
                r["id"] = i + 1
                r["name"] = f"土层面 {i + 1}"

            self._refresh_region_tree(keep_selection=False)
            self._reset_to_ground_view()
            self._emit_now()

            msg_parts = []
            if merged_count > 0:
                msg_parts.append(f"已合并 {merged_count} 个空洞至相邻土层")
            if appended_count > 0:
                msg_parts.append(f"独立生成 {appended_count} 个新土层面")
            msg = ", ".join(msg_parts)
            if skipped > 0:
                msg += f" (跳过 {skipped} 个退化几何)"
            msg += f"; 当前共 {len(self.data_regions)} 个土层面。"
            self.strata_validation_failed.emit(msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.strata_validation_failed.emit(f"填充失败: {e}")

    # ==================================================================
    # 图层树刷新
    # ==================================================================
    def _refresh_region_tree(self, keep_selection: bool = True):
        self.tree.blockSignals(True)
        try:
            prev_selected = set()
            if keep_selection:
                for item in self.tree.selectedItems():
                    if item.parent() is self.item_regions_root:
                        prev_selected.add(self.item_regions_root.indexOfChild(item))

            while self.item_regions_root.childCount() > 0:
                self.item_regions_root.removeChild(self.item_regions_root.child(0))

            for region in self.data_regions:
                pts = region.get("points", [])
                holes = region.get("holes", [])
                color = region.get("color", "#f3dfaa")
                mat_idx = int(region.get("material_index", 0))
                mat_name = (self.combo_region_material.itemText(mat_idx)
                            if 0 <= mat_idx < self.combo_region_material.count()
                            else f"材料 {mat_idx + 1}")

                label = region.get("name", "土层面")
                if holes:
                    label += f" ({len(holes)}洞)"

                item = QTreeWidgetItem([label, str(len(pts)), mat_name])
                item.setBackground(0, QBrush(QColor(color)))
                item.setTextAlignment(1, Qt.AlignCenter)
                item.setTextAlignment(2, Qt.AlignCenter)
                self.item_regions_root.addChild(item)

            self.item_regions_root.setText(1, str(len(self.data_regions)))
            self.item_regions_root.setExpanded(True)

            if prev_selected:
                for idx in prev_selected:
                    if 0 <= idx < self.item_regions_root.childCount():
                        self.item_regions_root.child(idx).setSelected(True)
                first = min(prev_selected)
                if 0 <= first < self.item_regions_root.childCount():
                    self.tree.setCurrentItem(self.item_regions_root.child(first))
        finally:
            self.tree.blockSignals(False)

    # ==================================================================
    # 删除
    # ==================================================================
    def _del_current_entity(self):
        selected_items = self.tree.selectedItems()
        region_indices = sorted({self.item_regions_root.indexOfChild(it)
                                 for it in selected_items
                                 if it.parent() is self.item_regions_root},
                                reverse=True)

        if region_indices:
            for idx in region_indices:
                if 0 <= idx < len(self.data_regions):
                    self.data_regions.pop(idx)
            self._refresh_region_tree(keep_selection=False)
            self._reset_to_ground_view()
            self._emit_now()
            self.strata_validation_failed.emit(
                f"已删除 {len(region_indices)} 个土层面。"
            )
            return

        curr = self.tree.currentItem()
        if curr is None:
            return
        if curr is self.item_ground:
            self.strata_validation_failed.emit("地表轮廓线不可删除。")
            return
        if curr is self.item_water:
            self.data_water = []
            self._clip_water_to_ground()
            self.item_water.setText(1, str(len(self.data_water)))
            self._reset_to_ground_view()
            self._emit_now()
            self.strata_validation_failed.emit("已重置为默认水位线。")
            return

    # ==================================================================
    # 材料名同步
    # ==================================================================
    def set_region_material_names(self, names):
        current_index = self.combo_region_material.currentIndex()
        self.combo_region_material.clear()
        self.combo_region_material.addItems(names or ["材料 1"])
        if 0 <= current_index < self.combo_region_material.count():
            self.combo_region_material.setCurrentIndex(current_index)
        if hasattr(self, "item_regions_root"):
            self._refresh_region_tree()

    # ==================================================================
    # 读取
    # ==================================================================
    def get_ground_points(self):
        self._save_current_table_data()
        return self.data_ground

    def get_water_points(self):
        self._save_current_table_data()
        if not self.data_water:
            return None
        return self.data_water

    def get_layer_regions(self):
        self._save_current_table_data()
        return self.data_regions

    def select_region_by_index(self, region_index: int):
        if not (0 <= region_index < self.item_regions_root.childCount()):
            return
        item = self.item_regions_root.child(region_index)
        if item is not None:
            self.tree.setCurrentItem(item)

    def select_special_entity(self, tag: int):
        if tag == self.TAG_GROUND:
            self.tree.setCurrentItem(self.item_ground)
        elif tag == self.TAG_WATER:
            self.tree.setCurrentItem(self.item_water)

    # ==================================================================
    # 持久化
    # ==================================================================
    def to_dict(self):
        self._save_current_table_data()
        return {
            "ground": [list(p) for p in self.data_ground],
            "water": [list(p) for p in self.data_water] if self.data_water else None,
            "regions": [
                {**{k: v for k, v in r.items() if k not in ("points", "holes")},
                 "points": [list(p) for p in r.get("points", [])],
                 "holes": [[list(p) for p in h] for h in r.get("holes", [])]}
                for r in self.data_regions
            ],
            "snap_tolerance": self.spin_snap_tol.value(),
            "snap_mode": self.combo_snap_mode.currentData() or "both",
            "hatch_scale": self.slider_hatch_scale.value(),
            "bottom_depth": self._bottom_depth,  # ★ 新增
        }

    def from_dict(self, data):
        if not isinstance(data, dict):
            return

        if data.get("ground") and len(data["ground"]) >= 2:
            self.data_ground = [(float(x), float(y)) for x, y in data["ground"]]
        if data.get("water"):
            self.data_water = [(float(x), float(y)) for x, y in data["water"]]
        else:
            self.data_water = []

        self.data_regions = []
        for r in data.get("regions", []) or []:
            pts = r.get("points", [])
            if len(pts) < 3:
                continue
            holes = []
            for h in r.get("holes", []) or []:
                if len(h) >= 3:
                    holes.append([(float(x), float(y)) for x, y in h])
            self.data_regions.append({
                "id": int(r.get("id", len(self.data_regions) + 1)),
                "name": str(r.get("name", f"土层面 {len(self.data_regions) + 1}")),
                "points": [(float(x), float(y)) for x, y in pts],
                "holes": holes,
                "material_index": int(r.get("material_index", 0)),
                "color": str(r.get("color", "#f3dfaa")),
                "pattern": str(r.get("pattern", "solid")),
                "hatch_scale_pct": int(r.get("hatch_scale_pct", 100)),
            })

        if not self.data_regions:
            self.data_regions = [self._build_default_region()]

        if "snap_tolerance" in data:
            self.spin_snap_tol.setValue(float(data["snap_tolerance"]))
        if "snap_mode" in data:
            mode = str(data["snap_mode"])
            for i in range(self.combo_snap_mode.count()):
                if self.combo_snap_mode.itemData(i) == mode:
                    self.combo_snap_mode.setCurrentIndex(i)
                    break
        if "hatch_scale" in data:
            self.slider_hatch_scale.setValue(int(data["hatch_scale"]))
        if "bottom_depth" in data:
            self.spin_bottom_depth.blockSignals(True)
            self.spin_bottom_depth.setValue(float(data["bottom_depth"]))
            self._bottom_depth = float(data["bottom_depth"])
            self.spin_bottom_depth.blockSignals(False)

        self._clip_water_to_ground()
        self._refresh_region_tree(keep_selection=False)
        self._reset_to_ground_view()


class MaterialDockWidget(QDockWidget):
    """材料库 + 深度效应 + 概率分布 + 空间云图

    云图 = 深度趋势 + 空间噪声 (二者叠加)
      · 未启用深度效应 → 趋势 = 均值 (常数)
      · COV = 0 或 DET → 噪声 = 0
      · 二者都开 → 显示"梯度 + 斑驳"的复合云图
    """

    materials_changed = pyqtSignal()

    _ICON_SIZE = 20
    _BTN_SIZE = 32

    # (key, 中文名, 单位, 是否非饱和, 建议下限, 建议上限)
    _DIST_ROWS = [
        ("gamma_dist",           "γ 天然重度",    "kN/m³", False,  0, 40),
        ("gamma_sat_dist",       "γsat 饱和重度",  "kN/m³", False,  0, 40),
        ("c_dist",               "c' 黏聚力",      "kPa",   False,  0, 1000),
        ("phi_dist",             "φ' 摩擦角",      "°",     False,  0, 60),
        ("phi_b_dist",           "φb 吸力摩擦角",  "°",     True,   0, 50),
        ("suction_cutoff_dist",  "吸力截断",        "kPa",   True,   0, 2000),
        ("c_depth_rate_dist",    "c 深度增长率",    "kPa/m", False,  0, 100),
        ("phi_depth_rate_dist",  "φ 深度增长率",    "°/m",   False,  0, 10),
    ]

    # ==================================================================
    def __init__(self, parent=None):
        super().__init__("物理力学参数与本构模型", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.current_material_index = 0
        self._preview_mode = "curve"      # "curve" | "field"
        self._ground_pts = []
        self._layer_regions = []
        self._grid_cache_key = None
        self._grid_cache = None
        self._last_field = None
        self._locked_field = None

        self._theta_timer = QTimer(self)
        self._theta_timer.setSingleShot(True)
        self._theta_timer.setInterval(200)
        self._theta_timer.timeout.connect(self._refresh_spatial_view)

        self._init_ui()

        self.materials_db = [
            SoilMaterial("默认土层", 19.0, 21.0, 15.0, 20.0, False, 15.0, 100.0)
        ]
        self._refresh_material_list()
        self._load_material_params(0)

    # ==================================================================
    # 图标按钮
    # ==================================================================
    def _icon_btn(self, icon_name: str, tooltip: str, slot):
        b = QPushButton()
        b.setIcon(get_icon(icon_name))
        b.setIconSize(QSize(self._ICON_SIZE, self._ICON_SIZE))
        b.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        b.setToolTip(tooltip)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(slot)
        b.setStyleSheet(
            "QPushButton{border:1px solid #bdc3c7;border-radius:4px;background:#fdfdfd;}"
            "QPushButton:hover{background:#ecf0f1;border-color:#3498db;}"
            "QPushButton:pressed{background:#d5dbdb;}"
        )
        return b

    # ==================================================================
    # 唯一名称
    # ==================================================================
    def _unique_name(self, base: str, exclude_idx: int = -1) -> str:
        base = (base or "新土层").strip()
        names = {m.name for i, m in enumerate(self.materials_db)
                 if i != exclude_idx}
        if base not in names:
            return base
        k = 2
        while f"{base} ({k})" in names:
            k += 1
        return f"{base} ({k})"

    # ==================================================================
    # UI
    # ==================================================================
    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---------- 力学分析工况 ----------
        grp_regime = QGroupBox("力学分析工况与本构模式")
        f_regime = QFormLayout(grp_regime)
        self.combo_regime = QComboBox()
        self.combo_regime.addItems([
            "常规有效应力模式 (饱和/天然工况)",
            "非饱和吸力强度模式 (Fredlund 双应力准则)",
        ])
        self.combo_regime.currentIndexChanged.connect(self._on_regime_changed)
        f_regime.addRow("分析工况:", self.combo_regime)
        layout.addWidget(grp_regime)

        # ---------- 材料库 (单列列表) ----------
        grp_list = QGroupBox("材料库")
        v_list = QVBoxLayout(grp_list)

        self.list_materials = QListWidget()
        self.list_materials.setSelectionMode(QListWidget.SingleSelection)
        self.list_materials.currentRowChanged.connect(self._on_material_row_changed)
        self.list_materials.setMinimumHeight(120)
        v_list.addWidget(self.list_materials)

        h_btns = QHBoxLayout()
        h_btns.setSpacing(6)
        h_btns.addWidget(self._icon_btn("material_add", "新增材料", self._add_material))
        h_btns.addWidget(self._icon_btn("material_copy", "复制当前材料", self._copy_material))
        h_btns.addWidget(self._icon_btn("material_del", "删除当前材料", self._del_material))
        h_btns.addStretch()
        v_list.addLayout(h_btns)
        layout.addWidget(grp_list, stretch=0)

        # ---------- 当前材料 ----------
        grp_detail = QGroupBox("当前材料")
        v_detail = QVBoxLayout(grp_detail)
        v_detail.setContentsMargins(6, 6, 6, 6)
        v_detail.setSpacing(6)

        # 名称行
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("名称:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("输入材料名称")
        self.edit_name.editingFinished.connect(self._on_name_edit_finished)
        h_name.addWidget(self.edit_name, 1)
        v_detail.addLayout(h_name)

        # 4 个 tab
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_basic(), "基本")
        self.tabs.addTab(self._build_tab_depth(), "深度效应")
        self.tabs.addTab(self._build_tab_dist(), "概率分布")
        self.tabs.addTab(self._build_tab_preview(), "预览")
        v_detail.addWidget(self.tabs)
        layout.addWidget(grp_detail, stretch=3)

        # ---------- 滚动容器 ----------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        self.setWidget(scroll)

    # ------------------------------------------------------------------
    # Tab 1: 基本
    # ------------------------------------------------------------------
    def _build_tab_basic(self):
        w = QWidget()
        f = QFormLayout(w)

        def mk_spin(lo, hi, val, step, suffix, decimals=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        self.spin_gamma_dry = mk_spin(5, 40, 19.0, 0.1, " kN/m³")
        self.spin_gamma_sat = mk_spin(5, 40, 21.0, 0.1, " kN/m³")
        self.spin_c = mk_spin(0, 500, 15.0, 1.0, " kPa")
        self.spin_phi = mk_spin(0, 55, 20.0, 0.5, " °")
        self.spin_phib = mk_spin(0, 50, 15.0, 0.5, " °")
        self.spin_cutoff = mk_spin(0, 1000, 100.0, 5.0, " kPa")

        f.addRow("天然重度 (γ):", self.spin_gamma_dry)
        f.addRow("饱和重度 (γsat):", self.spin_gamma_sat)
        f.addRow("有效黏聚力 (c'):", self.spin_c)
        f.addRow("有效摩擦角 (φ'):", self.spin_phi)

        self.grp_unsat = QGroupBox("非饱和基质吸力参数")
        fu = QFormLayout(self.grp_unsat)
        fu.addRow("吸力摩擦角 (φb):", self.spin_phib)
        fu.addRow("吸力截断上限:", self.spin_cutoff)
        f.addRow(self.grp_unsat)
        self.grp_unsat.setEnabled(False)

        for spin, key in ((self.spin_gamma_dry, "gamma_dist"),
                          (self.spin_gamma_sat, "gamma_sat_dist"),
                          (self.spin_c, "c_dist"),
                          (self.spin_phi, "phi_dist"),
                          (self.spin_phib, "phi_b_dist"),
                          (self.spin_cutoff, "suction_cutoff_dist")):
            spin.valueChanged.connect(
                lambda _v, k=key: self._sync_basic_to_dist(k))
            spin.editingFinished.connect(self._flush_to_material)
        return w

    # ------------------------------------------------------------------
    # Tab 2: 深度效应
    # ------------------------------------------------------------------
    def _build_tab_depth(self):
        w = QWidget()
        v = QVBoxLayout(w)

        self.chk_use_depth = QCheckBox("启用深度效应 (c, φ 随深度线性增长)")
        v.addWidget(self.chk_use_depth)

        f = QFormLayout()

        def mk_spin(lo, hi, val, step, suffix, decimals=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        self.spin_c_rate = mk_spin(0.0, 50.0, 0.0, 0.5, " kPa/m")
        self.spin_phi_rate = mk_spin(0.0, 5.0, 0.0, 0.1, " °/m")
        self.spin_depth_ref = mk_spin(-100.0, 200.0, 0.0, 0.5, " m")

        f.addRow("c' 随深度增长率:", self.spin_c_rate)
        f.addRow("φ' 随深度增长率:", self.spin_phi_rate)
        f.addRow("参考深度 (相对坡顶):", self.spin_depth_ref)
        v.addLayout(f)

        lbl_note = QLabel(
            "<i>注: 非饱和参数 (φb, 吸力截断) 与深度无直接关联, "
            "不参与深度效应。</i>")
        lbl_note.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        lbl_note.setWordWrap(True)
        v.addWidget(lbl_note)

        v.addWidget(QLabel("<b>深度剖面预览:</b>"))
        self.tbl_profile = QTableWidget(0, 3)
        self.tbl_profile.setHorizontalHeaderLabels(
            ["深度 z (m)", "c' (kPa)", "φ' (°)"])
        self.tbl_profile.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_profile.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_profile.setMaximumHeight(180)
        v.addWidget(self.tbl_profile)
        v.addStretch()

        self._depth_widgets = [self.spin_c_rate, self.spin_phi_rate,
                               self.spin_depth_ref]
        for x in self._depth_widgets:
            x.setEnabled(False)

        self.chk_use_depth.stateChanged.connect(self._on_depth_toggle)
        self.chk_use_depth.stateChanged.connect(
            lambda _v: self._flush_to_material())
        for s in (self.spin_c_rate, self.spin_phi_rate, self.spin_depth_ref):
            s.valueChanged.connect(self._refresh_depth_profile)
            s.editingFinished.connect(self._flush_to_material)
        return w

    def _on_depth_toggle(self, _=None):
        on = self.chk_use_depth.isChecked()
        for x in self._depth_widgets:
            x.setEnabled(on)
        self._refresh_depth_profile()

    def _refresh_depth_profile(self, *_):
        self.tbl_profile.setRowCount(0)
        use = self.chk_use_depth.isChecked()
        c0 = self.spin_c.value()
        phi0 = self.spin_phi.value()
        c_rate = self.spin_c_rate.value()
        phi_rate = self.spin_phi_rate.value()
        z_ref = self.spin_depth_ref.value()

        z_list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
        self.tbl_profile.setRowCount(len(z_list))
        for r, z in enumerate(z_list):
            if use:
                de = max(0.0, z - z_ref)
                c_val = c0 + c_rate * de
                phi_val = phi0 + phi_rate * de
            else:
                c_val = c0
                phi_val = phi0
            self.tbl_profile.setItem(r, 0, QTableWidgetItem(f"{z:.2f}"))
            self.tbl_profile.setItem(r, 1, QTableWidgetItem(f"{c_val:.2f}"))
            self.tbl_profile.setItem(r, 2, QTableWidgetItem(f"{phi_val:.2f}"))

    # ------------------------------------------------------------------
    # Tab 3: 概率分布
    # ------------------------------------------------------------------
    def _build_tab_dist(self):
        w = QWidget()
        v = QVBoxLayout(w)

        self.chk_use_random = QCheckBox("启用概率分布 (供可靠度分析抽样)")
        v.addWidget(self.chk_use_random)

        self.tbl_dist = QTableWidget(len(self._DIST_ROWS), 7)
        self.tbl_dist.setHorizontalHeaderLabels(
            ["参数", "单位", "分布类型", "均值", "COV", "下限", "上限"])
        hh = self.tbl_dist.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for c in (2, 3, 4, 5, 6):
            hh.setSectionResizeMode(c, QHeaderView.Stretch)
        self.tbl_dist.verticalHeader().setVisible(False)
        self.tbl_dist.setMinimumHeight(280)
        v.addWidget(self.tbl_dist)

        self._dist_widgets = {}

        for r, (key, name, unit, is_unsat, lo, hi) in enumerate(self._DIST_ROWS):
            self.tbl_dist.setItem(r, 0, QTableWidgetItem(name))
            self.tbl_dist.setItem(r, 1, QTableWidgetItem(unit))

            combo = QComboBox()
            for k, lbl in Distribution.KIND_LABELS.items():
                combo.addItem(lbl, k)
            self.tbl_dist.setCellWidget(r, 2, combo)

            spin_mean = QDoubleSpinBox()
            spin_mean.setRange(-1e6, 1e6)
            spin_mean.setDecimals(3)
            spin_mean.setSingleStep(0.5)
            spin_mean.setKeyboardTracking(False)
            self.tbl_dist.setCellWidget(r, 3, spin_mean)

            spin_cov = QDoubleSpinBox()
            spin_cov.setRange(0.0, 1.0)
            spin_cov.setDecimals(3)
            spin_cov.setSingleStep(0.01)
            spin_cov.setKeyboardTracking(False)
            self.tbl_dist.setCellWidget(r, 4, spin_cov)

            spin_lo = QDoubleSpinBox()
            spin_lo.setRange(-1e6, 1e6)
            spin_lo.setDecimals(3)
            spin_lo.setSingleStep(0.5)
            spin_lo.setValue(float(lo))
            spin_lo.setKeyboardTracking(False)
            self.tbl_dist.setCellWidget(r, 5, spin_lo)

            spin_hi = QDoubleSpinBox()
            spin_hi.setRange(-1e6, 1e6)
            spin_hi.setDecimals(3)
            spin_hi.setSingleStep(0.5)
            spin_hi.setValue(float(hi))
            spin_hi.setKeyboardTracking(False)
            self.tbl_dist.setCellWidget(r, 6, spin_hi)

            self._dist_widgets[key] = {
                "row": r, "combo": combo, "mean": spin_mean,
                "cov": spin_cov, "lo": spin_lo, "hi": spin_hi,
            }

            combo.currentIndexChanged.connect(
                lambda _i, k=key: self._on_dist_kind_changed(k))
            spin_mean.valueChanged.connect(
                lambda _v, k=key: self._on_dist_mean_changed(k))
            spin_cov.valueChanged.connect(
                lambda _v, k=key: self._on_dist_value_changed(k, "cov"))
            spin_lo.valueChanged.connect(
                lambda _v, k=key: self._on_dist_value_changed(k, "lo"))
            spin_hi.valueChanged.connect(
                lambda _v, k=key: self._on_dist_value_changed(k, "hi"))

        v.addStretch()
        self.tbl_dist.setEnabled(False)
        self.chk_use_random.stateChanged.connect(self._on_random_toggle)
        return w

    # ---------- 分布表联动 ----------
    @staticmethod
    def _set_spin(spin, value):
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _update_dist_row_enabled(self, key, kind):
        wdg = self._dist_widgets.get(key)
        if wdg is None:
            return
        if kind == Distribution.DET:
            wdg["mean"].setEnabled(True)
            wdg["cov"].setEnabled(False)
            wdg["lo"].setEnabled(False)
            wdg["hi"].setEnabled(False)
        elif kind in (Distribution.NORMAL, Distribution.LOGNORMAL):
            wdg["mean"].setEnabled(True)
            wdg["cov"].setEnabled(True)
            wdg["lo"].setEnabled(False)
            wdg["hi"].setEnabled(False)
        elif kind == Distribution.TRUNC_NORMAL:
            wdg["mean"].setEnabled(True)
            wdg["cov"].setEnabled(True)
            wdg["lo"].setEnabled(True)
            wdg["hi"].setEnabled(True)
        elif kind == Distribution.UNIFORM:
            wdg["mean"].setEnabled(False)
            wdg["cov"].setEnabled(False)
            wdg["lo"].setEnabled(True)
            wdg["hi"].setEnabled(True)

    def _on_dist_kind_changed(self, key):
        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            return
        d = getattr(self.materials_db[idx], key, None)
        wdg = self._dist_widgets.get(key)
        if not isinstance(d, Distribution) or wdg is None:
            return

        kind = wdg["combo"].currentData()
        d.kind = kind

        if kind == Distribution.DET:
            self._set_spin(wdg["cov"], 0.0)
            self._set_spin(wdg["lo"], d.mean)
            self._set_spin(wdg["hi"], d.mean)
            d.cov = 0.0
            d.lower = d.mean
            d.upper = d.mean

        elif kind in (Distribution.NORMAL, Distribution.LOGNORMAL):
            d.lower = None
            d.upper = None
            s2 = 2.0 * abs(d.std) if d.cov > 0 else 0.0
            self._set_spin(wdg["lo"], d.mean - s2)
            self._set_spin(wdg["hi"], d.mean + s2)

        elif kind == Distribution.TRUNC_NORMAL:
            lo = wdg["lo"].value()
            hi = wdg["hi"].value()
            if hi <= lo:
                hi = lo + 1.0
                self._set_spin(wdg["hi"], hi)
            if d.mean < lo:
                self._set_spin(wdg["mean"], lo)
                self._sync_dist_to_basic(key, lo)
                d.mean = lo
            if d.mean > hi:
                self._set_spin(wdg["mean"], hi)
                self._sync_dist_to_basic(key, hi)
                d.mean = hi
            d.lower = lo
            d.upper = hi

        elif kind == Distribution.UNIFORM:
            lo = wdg["lo"].value()
            hi = wdg["hi"].value()
            if hi <= lo:
                hi = lo + 1.0
                self._set_spin(wdg["hi"], hi)
            mid = 0.5 * (lo + hi)
            self._set_spin(wdg["mean"], mid)
            self._sync_dist_to_basic(key, mid)
            d.mean = mid
            d.lower = lo
            d.upper = hi
            d.cov = ((hi - lo) / (2 * np.sqrt(3) * mid)
                     if mid > 1e-9 else 0.0)
            self._set_spin(wdg["cov"], d.cov)

        self._update_dist_row_enabled(key, kind)
        self._refresh_preview_tab()
        if self._preview_mode == "field":
            self._refresh_spatial_view()

    def _on_dist_mean_changed(self, key):
        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            return
        d = getattr(self.materials_db[idx], key, None)
        wdg = self._dist_widgets.get(key)
        if not isinstance(d, Distribution) or wdg is None:
            return

        v = wdg["mean"].value()
        d.mean = v

        if d.kind == Distribution.TRUNC_NORMAL:
            lo = wdg["lo"].value()
            hi = wdg["hi"].value()
            if v < lo:
                self._set_spin(wdg["lo"], v)
                d.lower = v
            if v > hi:
                self._set_spin(wdg["hi"], v)
                d.upper = v

        if d.kind == Distribution.UNIFORM:
            return

        self._sync_dist_to_basic(key, v)
        self._refresh_preview_tab()
        if self._preview_mode == "field":
            self._refresh_spatial_view()

    def _on_dist_value_changed(self, key, field):
        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            return
        d = getattr(self.materials_db[idx], key, None)
        wdg = self._dist_widgets.get(key)
        if not isinstance(d, Distribution) or wdg is None:
            return

        if d.kind in (Distribution.DET,
                      Distribution.NORMAL,
                      Distribution.LOGNORMAL):
            if field == "cov":
                d.cov = wdg["cov"].value()
            if d.kind in (Distribution.NORMAL, Distribution.LOGNORMAL):
                s2 = 2.0 * abs(d.std)
                self._set_spin(wdg["lo"], d.mean - s2)
                self._set_spin(wdg["hi"], d.mean + s2)
            self._refresh_preview_tab()
            if self._preview_mode == "field":
                self._refresh_spatial_view()
            return

        lo = wdg["lo"].value()
        hi = wdg["hi"].value()
        mean = wdg["mean"].value()

        if hi < lo:
            if field == "lo":
                hi = lo
                self._set_spin(wdg["hi"], hi)
            else:
                lo = hi
                self._set_spin(wdg["lo"], lo)

        if mean < lo:
            if field == "lo":
                self._set_spin(wdg["mean"], lo)
                d.mean = lo
                mean = lo
                self._sync_dist_to_basic(key, lo)
            else:
                self._set_spin(wdg["lo"], mean)
                lo = mean
        if mean > hi:
            if field == "hi":
                self._set_spin(wdg["mean"], hi)
                d.mean = hi
                mean = hi
                self._sync_dist_to_basic(key, hi)
            else:
                self._set_spin(wdg["hi"], mean)
                hi = mean

        d.lower = lo
        d.upper = hi
        if field == "cov":
            d.cov = wdg["cov"].value()

        if d.kind == Distribution.UNIFORM:
            mid = 0.5 * (lo + hi)
            self._set_spin(wdg["mean"], mid)
            d.mean = mid
            self._sync_dist_to_basic(key, mid)
            d.cov = ((hi - lo) / (2 * np.sqrt(3) * mid)
                     if mid > 1e-9 else 0.0)
            self._set_spin(wdg["cov"], d.cov)

        self._refresh_preview_tab()
        if self._preview_mode == "field":
            self._refresh_spatial_view()

    def _sync_dist_to_basic(self, key, value):
        mapping = {
            "gamma_dist": self.spin_gamma_dry,
            "gamma_sat_dist": self.spin_gamma_sat,
            "c_dist": self.spin_c,
            "phi_dist": self.spin_phi,
            "phi_b_dist": self.spin_phib,
            "suction_cutoff_dist": self.spin_cutoff,
        }
        spin = mapping.get(key)
        if spin is None:
            return
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _sync_basic_to_dist(self, key):
        mapping = {
            "gamma_dist": self.spin_gamma_dry,
            "gamma_sat_dist": self.spin_gamma_sat,
            "c_dist": self.spin_c,
            "phi_dist": self.spin_phi,
            "phi_b_dist": self.spin_phib,
            "suction_cutoff_dist": self.spin_cutoff,
        }
        spin = mapping.get(key)
        wdg = self._dist_widgets.get(key)
        if spin is None or wdg is None:
            return

        v = spin.value()
        self._set_spin(wdg["mean"], v)

        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            return
        d = getattr(self.materials_db[idx], key, None)
        if not isinstance(d, Distribution):
            return
        d.mean = v

        if d.kind in (Distribution.TRUNC_NORMAL, Distribution.UNIFORM):
            lo = wdg["lo"].value()
            hi = wdg["hi"].value()
            if v < lo:
                self._set_spin(wdg["lo"], v)
                lo = v
            if v > hi:
                self._set_spin(wdg["hi"], v)
                hi = v
            d.lower = lo
            d.upper = hi

    def _on_random_toggle(self, _=None):
        on = self.chk_use_random.isChecked()
        self.tbl_dist.setEnabled(on)
        idx = self.current_material_index
        if 0 <= idx < len(self.materials_db):
            self.materials_db[idx].use_random_dist = on
        self._refresh_preview_tab()
        if self._preview_mode == "field":
            self._refresh_spatial_view()

    # ------------------------------------------------------------------
    # Tab 4: 预览
    # ------------------------------------------------------------------
    def _build_tab_preview(self):
        w = QWidget()
        v = QVBoxLayout(w)

        h_top = QHBoxLayout()
        h_top.addWidget(QLabel("视图:"))
        self.btn_view_curve = QPushButton("分布曲线")
        self.btn_view_curve.setCheckable(True)
        self.btn_view_curve.setChecked(True)
        self.btn_view_curve.clicked.connect(
            lambda: self._switch_preview_view("curve"))
        self.btn_view_field = QPushButton("空间云图")
        self.btn_view_field.setCheckable(True)
        self.btn_view_field.clicked.connect(
            lambda: self._switch_preview_view("field"))
        h_top.addWidget(self.btn_view_curve)
        h_top.addWidget(self.btn_view_field)
        h_top.addStretch()
        h_top.addWidget(self._icon_btn(
            "apply_surface", "冻结当前实现 → 参与后续计算", self._freeze_field))
        h_top.addWidget(self._icon_btn(
            "dist_export", "导出当前视图为 PNG", self._export_preview))
        v.addLayout(h_top)

        # 曲线模式控件
        self.w_curve_ctrl = QWidget()
        h_curve = QHBoxLayout(self.w_curve_ctrl)
        h_curve.setContentsMargins(0, 0, 0, 0)
        h_curve.addWidget(QLabel("参数:"))
        self.combo_preview = QComboBox()
        for key, name, unit, *_ in self._DIST_ROWS:
            self.combo_preview.addItem(f"{name}  [{unit}]", (key, unit))
        self.combo_preview.currentIndexChanged.connect(
            lambda _: self._refresh_preview_tab())
        h_curve.addWidget(self.combo_preview, 1)
        v.addWidget(self.w_curve_ctrl)

        # 云图模式控件 (无"模式"下拉, 云图 = 趋势 + 噪声)
        self.w_field_ctrl = QWidget()
        h_field = QHBoxLayout(self.w_field_ctrl)
        h_field.setContentsMargins(0, 0, 0, 0)
        h_field.addWidget(QLabel("参数:"))
        self.combo_field_param = QComboBox()
        for key, name, unit, *_ in self._DIST_ROWS:
            self.combo_field_param.addItem(f"{name}  [{unit}]", (key, unit))
        self.combo_field_param.currentIndexChanged.connect(
            lambda _: self._refresh_spatial_view())
        h_field.addWidget(self.combo_field_param, 1)
        h_field.addWidget(QLabel("θx:"))
        self.spin_theta_x = QDoubleSpinBox()
        self.spin_theta_x.setRange(0.5, 100.0)
        self.spin_theta_x.setValue(5.0)
        self.spin_theta_x.setSuffix(" m")
        self.spin_theta_x.valueChanged.connect(self._on_theta_changed_debounced)
        h_field.addWidget(self.spin_theta_x)
        h_field.addWidget(QLabel("θy:"))
        self.spin_theta_y = QDoubleSpinBox()
        self.spin_theta_y.setRange(0.5, 100.0)
        self.spin_theta_y.setValue(2.0)
        self.spin_theta_y.setSuffix(" m")
        self.spin_theta_y.valueChanged.connect(self._on_theta_changed_debounced)
        h_field.addWidget(self.spin_theta_y)
        h_field.addWidget(QLabel("种子:"))
        self.spin_field_seed = QSpinBox()
        self.spin_field_seed.setRange(0, 2_000_000_000)
        self.spin_field_seed.setValue(42)
        self.spin_field_seed.valueChanged.connect(
            lambda _: self._refresh_spatial_view())
        h_field.addWidget(self.spin_field_seed)
        v.addWidget(self.w_field_ctrl)

        # 云图作用域提示
        self.lbl_field_scope = QLabel("")
        self.lbl_field_scope.setStyleSheet(
            "color: #7f8c8d; font-size: 11px; font-style: italic;")
        self.lbl_field_scope.setWordWrap(True)
        self.lbl_field_scope.setVisible(False)
        v.addWidget(self.lbl_field_scope)

        self.dist_plot = DistributionPlotWidget()
        v.addWidget(self.dist_plot, stretch=1)
        self.spatial_plot = SpatialFieldWidget()
        self.spatial_plot.setVisible(False)
        v.addWidget(self.spatial_plot, stretch=1)

        self.w_field_ctrl.setVisible(False)
        return w

    def _switch_preview_view(self, mode):
        self._preview_mode = mode
        self.btn_view_curve.setChecked(mode == "curve")
        self.btn_view_field.setChecked(mode == "field")
        self.w_curve_ctrl.setVisible(mode == "curve")
        self.w_field_ctrl.setVisible(mode == "field")
        self.lbl_field_scope.setVisible(mode == "field")
        self.dist_plot.setVisible(mode == "curve")
        self.spatial_plot.setVisible(mode == "field")
        if mode == "curve":
            self._refresh_preview_tab()
        else:
            self._refresh_spatial_view()

    def _refresh_preview_tab(self):
        if self._preview_mode != "curve":
            return
        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            self.dist_plot.clear("无材料")
            return
        m = self.materials_db[idx]
        if not m.use_random_dist:
            self.dist_plot.clear("未启用概率分布 (请在上方勾选)")
            return
        key, unit = self.combo_preview.currentData()
        d = getattr(m, key, None)
        if not isinstance(d, Distribution):
            self.dist_plot.clear("分布对象缺失")
            return
        name = self.combo_preview.currentText().split("  [")[0]
        self.dist_plot.set_distribution(d, name, unit)

    def _export_preview(self):
        if self._preview_mode == "curve":
            self._export_dist_plot()
        else:
            self._export_spatial_plot()

    def _export_dist_plot(self):
        if self.dist_plot._dist is None:
            QMessageBox.information(self, "导出", "当前无可导出的分布曲线。")
            return
        name = self.combo_preview.currentText().split("  [")[0]
        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出分布图", f"dist_{name}.png",
            "PNG 图片 (*.png);;所有文件 (*.*)")
        if not fpath:
            return
        if not fpath.lower().endswith(".png"):
            fpath += ".png"
        if self.dist_plot.export_png(fpath):
            QMessageBox.information(self, "导出成功", f"分布图已保存至:\n{fpath}")
        else:
            QMessageBox.warning(self, "导出失败", "写入图片失败。")

    def _export_spatial_plot(self):
        try:
            pix = self.spatial_plot.grab()
        except Exception:
            QMessageBox.warning(self, "导出失败", "无法截取云图。")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出云图", "spatial_field.png",
            "PNG 图片 (*.png);;所有文件 (*.*)")
        if not fpath:
            return
        if not fpath.lower().endswith(".png"):
            fpath += ".png"
        if pix.save(fpath, "PNG"):
            QMessageBox.information(self, "导出成功", f"云图已保存至:\n{fpath}")
        else:
            QMessageBox.warning(self, "导出失败", "写入图片失败。")

    # ==================================================================
    # 空间云图
    # ==================================================================
    def set_geometry_context(self, ground_pts, layer_regions):
        self._ground_pts = list(ground_pts or [])
        self._layer_regions = list(layer_regions or [])
        self._grid_cache_key = None
        self._grid_cache = None
        if self._preview_mode == "field":
            self._refresh_spatial_view()

    def _on_theta_changed_debounced(self, _):
        self._theta_timer.start()

    def _build_region_masks(self, xs, ys, XX, YY):
        def _in_poly(xx, yy, poly):
            n = len(poly)
            if n < 3:
                return np.zeros_like(xx, dtype=bool)
            inside = np.zeros(xx.shape, dtype=bool)
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                cond = (yi > yy) != (yj > yy)
                denom = yj - yi
                denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
                x_cross = (xj - xi) * (yy - yi) / denom + xi
                inside ^= (cond & (xx < x_cross))
                j = i
            return inside

        masks = []
        XX_flat = XX.ravel()
        YY_flat = YY.ravel()
        for r in self._layer_regions:
            pts = r.get("points", [])
            if len(pts) < 3:
                masks.append(None)
                continue
            arr = np.asarray(pts, dtype=float)
            bx0, bx1 = arr[:, 0].min(), arr[:, 0].max()
            by0, by1 = arr[:, 1].min(), arr[:, 1].max()
            cand = ((XX_flat >= bx0) & (XX_flat <= bx1)
                    & (YY_flat >= by0) & (YY_flat <= by1))
            mask_flat = np.zeros_like(XX_flat, dtype=bool)
            if cand.any():
                inside_sub = _in_poly(XX_flat[cand], YY_flat[cand],
                                      [tuple(p) for p in arr])
                idx_cand = np.where(cand)[0]
                mask_flat[idx_cand[inside_sub]] = True
            masks.append(mask_flat.reshape(XX.shape))
        return masks

    @staticmethod
    def _effective_dist_stats(d):
        """分布的"生效"统计量, 兼容所有类型与上下限"""
        if d.kind == d.DET or d.cov <= 0.0:
            m = float(d.mean)
            return m, 0.0, m, m
        sample = d.sample(100000)
        eff_mean = float(sample.mean())
        eff_std = float(sample.std())
        lo = float(d.lower) if d.lower is not None else float(sample.min())
        hi = float(d.upper) if d.upper is not None else float(sample.max())
        return eff_mean, eff_std, lo, hi

    def _refresh_spatial_view(self):
        """云图 = 深度趋势 + 空间噪声 (二者叠加)"""
        if self._preview_mode != "field":
            return
        if not self._ground_pts or not self._layer_regions:
            self.spatial_plot.clear("未设置几何模型")
            self.lbl_field_scope.setText("")
            return
        if not self.materials_db:
            self.spatial_plot.clear("无材料")
            self.lbl_field_scope.setText("")
            return

        key, unit = self.combo_field_param.currentData()
        param_name = self.combo_field_param.currentText().split("  [")[0]

        # 作用域提示
        used_mats = set()
        for r in self._layer_regions:
            mi = int(r.get("material_index", 0))
            if 0 <= mi < len(self.materials_db):
                used_mats.add(mi)
        if used_mats:
            names = [f"{mi + 1}. {self.materials_db[mi].name}"
                     for mi in sorted(used_mats)]
            self.lbl_field_scope.setText(
                f"云图作用域: 坡体内实际引用的材料 = {' / '.join(names)}")
        else:
            self.lbl_field_scope.setText("云图作用域: 无有效材料引用")

        gx = [p[0] for p in self._ground_pts]
        gy = [p[1] for p in self._ground_pts]
        x_min, x_max = min(gx), max(gx)
        y_min, y_max = min(gy) - 8.0, max(gy) + 1.0
        nx, ny = 100, 60

        cache_key = (round(x_min, 4), round(x_max, 4),
                     round(y_min, 4), round(y_max, 4),
                     nx, ny, len(self._layer_regions))
        if self._grid_cache_key != cache_key:
            xs = np.linspace(x_min, x_max, nx)
            ys = np.linspace(y_min, y_max, ny)
            XX, YY = np.meshgrid(xs, ys)
            region_masks = self._build_region_masks(xs, ys, XX, YY)
            self._grid_cache = (xs, ys, XX, YY, region_masks)
            self._grid_cache_key = cache_key
        else:
            xs, ys, XX, YY, region_masks = self._grid_cache

        field = np.full_like(XX, np.nan, dtype=float)
        seed = int(self.spin_field_seed.value())
        theta_x = float(self.spin_theta_x.value())
        theta_y = float(self.spin_theta_y.value())
        y_top = max(gy)

        for r_idx, mask in enumerate(region_masks):
            if mask is None or not mask.any():
                continue
            r = self._layer_regions[r_idx]
            mat_idx = int(r.get("material_index", 0))
            if not (0 <= mat_idx < len(self.materials_db)):
                continue
            mat = self.materials_db[mat_idx]
            d = getattr(mat, key, None)
            if not isinstance(d, Distribution):
                continue

            eff_mean, eff_std, eff_lo, eff_hi = self._effective_dist_stats(d)

            # ---- 1. 深度趋势 (整个网格) ----
            if mat.use_depth_effect and key == "c_dist":
                Z = np.maximum(0.0, y_top - YY - mat.depth_ref)
                trend_map = mat.c_prime + mat.c_depth_rate * Z
            elif mat.use_depth_effect and key == "phi_dist":
                Z = np.maximum(0.0, y_top - YY - mat.depth_ref)
                trend_map = mat.phi_deg + mat.phi_depth_rate * Z
            else:
                trend_map = np.full_like(XX, eff_mean)

            # ---- 2. 空间噪声 (均值 0) ----
            noise_map = None
            if eff_std > 1e-12:
                try:
                    from core.random_field import generate_field_on_grid
                    region_seed = int(seed) * 10007 + r_idx * 7919
                    noise_map = generate_field_on_grid(
                        xs, ys, theta_x=theta_x, theta_y=theta_y,
                        mean=0.0, std=eff_std, seed=region_seed)
                except ImportError:
                    noise_map = None

            # ---- 3. 叠加 + clip ----
            field_r = trend_map.copy()
            if noise_map is not None:
                field_r = field_r + noise_map

            # 只有截断/均匀分布才 clip
            if d.kind in (Distribution.TRUNC_NORMAL, Distribution.UNIFORM):
                field_r = np.clip(field_r, eff_lo, eff_hi)

            field[mask] = field_r[mask]

        self._last_field = {
            "field": field.copy(), "xs": xs.copy(), "ys": ys.copy(),
            "key": key, "seed": seed,
            "theta_x": theta_x, "theta_y": theta_y,
        }
        self.spatial_plot.set_data(
            self._ground_pts, self._layer_regions,
            field, xs, ys, title=param_name, unit=unit)

    def _freeze_field(self):
        if self._last_field is None or self._grid_cache is None:
            QMessageBox.information(
                self, "冻结", "请先在'空间云图'模式下生成一张图再冻结。")
            return
        lf = self._last_field
        self._locked_field = {
            "xs": lf["xs"].copy(),
            "ys": lf["ys"].copy(),
            "seed": lf["seed"],
            "theta_x": lf["theta_x"],
            "theta_y": lf["theta_y"],
            "fields": {},
        }
        for key, *_ in self._DIST_ROWS:
            f = self._compute_field_for_key(key)
            if f is not None:
                self._locked_field["fields"][key] = f
        QMessageBox.information(
            self, "冻结完成",
            f"已冻结当前实现 (种子={lf['seed']})。\n"
            "后续稳定性分析将使用这份场。")

    def _compute_field_for_key(self, key):
        if self._grid_cache is None:
            return None
        xs, ys, XX, YY, region_masks = self._grid_cache
        field = np.full_like(XX, np.nan, dtype=float)
        seed = int(self.spin_field_seed.value())
        theta_x = float(self.spin_theta_x.value())
        theta_y = float(self.spin_theta_y.value())
        gx = [p[0] for p in self._ground_pts]
        gy = [p[1] for p in self._ground_pts]
        y_top = max(gy)

        for r_idx, mask in enumerate(region_masks):
            if mask is None or not mask.any():
                continue
            r = self._layer_regions[r_idx]
            mat_idx = int(r.get("material_index", 0))
            if not (0 <= mat_idx < len(self.materials_db)):
                continue
            mat = self.materials_db[mat_idx]
            d = getattr(mat, key, None)
            if not isinstance(d, Distribution):
                continue

            eff_mean, eff_std, eff_lo, eff_hi = self._effective_dist_stats(d)

            if mat.use_depth_effect and key == "c_dist":
                Z = np.maximum(0.0, y_top - YY - mat.depth_ref)
                trend_map = mat.c_prime + mat.c_depth_rate * Z
            elif mat.use_depth_effect and key == "phi_dist":
                Z = np.maximum(0.0, y_top - YY - mat.depth_ref)
                trend_map = mat.phi_deg + mat.phi_depth_rate * Z
            else:
                trend_map = np.full_like(XX, eff_mean)

            field_r = trend_map.copy()
            if eff_std > 1e-12:
                try:
                    from core.random_field import generate_field_on_grid
                    region_seed = int(seed) * 10007 + r_idx * 7919
                    noise_map = generate_field_on_grid(
                        xs, ys, theta_x=theta_x, theta_y=theta_y,
                        mean=0.0, std=eff_std, seed=region_seed)
                    field_r = field_r + noise_map
                except ImportError:
                    pass

            if d.kind in (Distribution.TRUNC_NORMAL, Distribution.UNIFORM):
                field_r = np.clip(field_r, eff_lo, eff_hi)

            field[mask] = field_r[mask]
        return field

    def get_locked_field(self):
        return self._locked_field

    def clear_locked_field(self):
        self._locked_field = None

    # ==================================================================
    # 实时写回 (基本 + 深度 tab)
    # ==================================================================
    def _flush_to_material(self):
        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            return
        m = self.materials_db[idx]

        m.gamma_dry = self.spin_gamma_dry.value()
        m.gamma_sat = self.spin_gamma_sat.value()
        m.c_prime = self.spin_c.value()
        m.phi_deg = self.spin_phi.value()
        m.phi_b_deg = self.spin_phib.value()
        m.suction_cutoff = self.spin_cutoff.value()

        m.use_depth_effect = self.chk_use_depth.isChecked()
        m.c_depth_rate = self.spin_c_rate.value()
        m.phi_depth_rate = self.spin_phi_rate.value()
        m.depth_ref = self.spin_depth_ref.value()

        self.materials_changed.emit()

    def _on_name_edit_finished(self):
        idx = self.current_material_index
        if not (0 <= idx < len(self.materials_db)):
            return
        m = self.materials_db[idx]

        raw = self.edit_name.text().strip()
        if not raw:
            self.edit_name.blockSignals(True)
            self.edit_name.setText(m.name)
            self.edit_name.blockSignals(False)
            return

        if raw == m.name:
            return

        # 冲突检测 (排除自身)
        new_name = self._unique_name(raw, exclude_idx=idx)

        m.name = new_name
        self.edit_name.blockSignals(True)
        self.edit_name.setText(m.name)
        self.edit_name.blockSignals(False)

        self._refresh_material_list()
        self.materials_changed.emit()

    # ==================================================================
    # 材料库操作
    # ==================================================================
    def _refresh_material_list(self):
        self.list_materials.blockSignals(True)
        self.list_materials.clear()
        for i, m in enumerate(self.materials_db):
            self.list_materials.addItem(f"{i + 1}. {m.name}")
        if 0 <= self.current_material_index < len(self.materials_db):
            self.list_materials.setCurrentRow(self.current_material_index)
        self.list_materials.blockSignals(False)

    def _on_material_row_changed(self, row):
        if row < 0 or row >= len(self.materials_db):
            return
        if row == self.current_material_index:
            return
        self.current_material_index = row
        self._load_material_params(row)

    def _load_material_params(self, idx):
        if not (0 <= idx < len(self.materials_db)):
            return
        m = self.materials_db[idx]

        self.edit_name.blockSignals(True)
        self.edit_name.setText(m.name)
        self.edit_name.blockSignals(False)

        for spin, val in ((self.spin_gamma_dry, m.gamma_dry),
                          (self.spin_gamma_sat, m.gamma_sat),
                          (self.spin_c, m.c_prime),
                          (self.spin_phi, m.phi_deg),
                          (self.spin_phib, m.phi_b_deg),
                          (self.spin_cutoff, m.suction_cutoff)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self.chk_use_depth.blockSignals(True)
        self.chk_use_depth.setChecked(m.use_depth_effect)
        self.chk_use_depth.blockSignals(False)
        self.spin_c_rate.blockSignals(True)
        self.spin_c_rate.setValue(m.c_depth_rate)
        self.spin_c_rate.blockSignals(False)
        self.spin_phi_rate.blockSignals(True)
        self.spin_phi_rate.setValue(m.phi_depth_rate)
        self.spin_phi_rate.blockSignals(False)
        self.spin_depth_ref.blockSignals(True)
        self.spin_depth_ref.setValue(m.depth_ref)
        self.spin_depth_ref.blockSignals(False)
        self._on_depth_toggle()

        self.chk_use_random.blockSignals(True)
        self.chk_use_random.setChecked(m.use_random_dist)
        self.chk_use_random.blockSignals(False)

        for key, wdg in self._dist_widgets.items():
            d = getattr(m, key, None)
            if not isinstance(d, Distribution):
                continue
            for x in (wdg["combo"], wdg["mean"], wdg["cov"],
                      wdg["lo"], wdg["hi"]):
                x.blockSignals(True)
            for i in range(wdg["combo"].count()):
                if wdg["combo"].itemData(i) == d.kind:
                    wdg["combo"].setCurrentIndex(i)
                    break
            wdg["mean"].setValue(d.mean)
            wdg["cov"].setValue(d.cov)

            if d.kind in (Distribution.TRUNC_NORMAL, Distribution.UNIFORM):
                lo_val = d.lower if d.lower is not None else d.mean - 2 * abs(d.std)
                hi_val = d.upper if d.upper is not None else d.mean + 2 * abs(d.std)
            else:
                s2 = 2.0 * abs(d.std)
                lo_val = d.mean - s2
                hi_val = d.mean + s2
            wdg["lo"].setValue(lo_val)
            wdg["hi"].setValue(hi_val)

            for x in (wdg["combo"], wdg["mean"], wdg["cov"],
                      wdg["lo"], wdg["hi"]):
                x.blockSignals(False)
            self._update_dist_row_enabled(key, d.kind)

        self.tbl_dist.setEnabled(m.use_random_dist)
        self.grp_unsat.setEnabled(m.is_unsaturated)

        self._refresh_depth_profile()
        self._refresh_preview_tab()
        if self._preview_mode == "field":
            self._refresh_spatial_view()

    def _add_material(self):
        name, ok = QInputDialog.getText(self, "新增材料", "请输入材料名称:")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            QMessageBox.warning(self, "提示", "名称不能为空。")
            return
        name = self._unique_name(name)

        self.materials_db.append(
            SoilMaterial(name, 19.0, 21.0, 15.0, 20.0, False, 15.0, 100.0))
        self.current_material_index = len(self.materials_db) - 1
        self._refresh_material_list()
        self._load_material_params(self.current_material_index)
        self.materials_changed.emit()

    def _copy_material(self):
        if not (0 <= self.current_material_index < len(self.materials_db)):
            return
        src = self.materials_db[self.current_material_index]
        new_mat = SoilMaterial.from_dict(src.to_dict())
        new_mat.name = self._unique_name(src.name)
        self.materials_db.append(new_mat)
        self.current_material_index = len(self.materials_db) - 1
        self._refresh_material_list()
        self._load_material_params(self.current_material_index)
        self.materials_changed.emit()

    def _del_material(self):
        if len(self.materials_db) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一个材料。")
            return
        if not (0 <= self.current_material_index < len(self.materials_db)):
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"删除材料 '{self.materials_db[self.current_material_index].name}'？\n"
            "引用了该材料的土层面将被自动重定向到第一个材料。",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.materials_db.pop(self.current_material_index)
        self.current_material_index = max(0, self.current_material_index - 1)
        self._refresh_material_list()
        self._load_material_params(self.current_material_index)
        self.materials_changed.emit()

    def _on_regime_changed(self, idx):
        is_unsat = (idx == 1)
        for m in self.materials_db:
            m.is_unsaturated = is_unsat
        self.grp_unsat.setEnabled(is_unsat)
        self.materials_changed.emit()

    # ==================================================================
    # 数据接口
    # ==================================================================
    def get_materials_list(self):
        return self.materials_db

    def to_dict(self):
        return {
            "active_regime": self.combo_regime.currentIndex(),
            "current_material": self.current_material_index,
            "layers": [m.to_dict() for m in self.materials_db],
            "field_seed": int(self.spin_field_seed.value()),
            "field_theta_x": float(self.spin_theta_x.value()),
            "field_theta_y": float(self.spin_theta_y.value()),
            "field_param_key": (self.combo_field_param.currentData()[0]
                                if self.combo_field_param.currentData()
                                else "c_dist"),
        }

    def from_dict(self, data):
        if not isinstance(data, dict):
            return
        layers = data.get("layers", [])
        if layers:
            self.materials_db = [SoilMaterial.from_dict(L) for L in layers]
        if not self.materials_db:
            self.materials_db = [SoilMaterial("默认土层")]

        regime = int(data.get("active_regime", 0))
        self.combo_regime.blockSignals(True)
        self.combo_regime.setCurrentIndex(regime)
        self.combo_regime.blockSignals(False)
        self.grp_unsat.setEnabled(regime == 1)

        self.current_material_index = int(data.get("current_material", 0))
        self.current_material_index = max(
            0, min(self.current_material_index, len(self.materials_db) - 1))

        if "field_seed" in data:
            self.spin_field_seed.blockSignals(True)
            self.spin_field_seed.setValue(int(data["field_seed"]))
            self.spin_field_seed.blockSignals(False)
        if "field_theta_x" in data:
            self.spin_theta_x.blockSignals(True)
            self.spin_theta_x.setValue(float(data["field_theta_x"]))
            self.spin_theta_x.blockSignals(False)
        if "field_theta_y" in data:
            self.spin_theta_y.blockSignals(True)
            self.spin_theta_y.setValue(float(data["field_theta_y"]))
            self.spin_theta_y.blockSignals(False)
        if "field_param_key" in data:
            k = str(data["field_param_key"])
            for i in range(self.combo_field_param.count()):
                if self.combo_field_param.itemData(i)[0] == k:
                    self.combo_field_param.setCurrentIndex(i)
                    break

        self._locked_field = None
        self._refresh_material_list()
        self._load_material_params(self.current_material_index)


class RainfallDockWidget(QDockWidget):
    """动态时序降雨过程与时间轴控制停靠窗"""
    time_step_changed = pyqtSignal(float)
    solve_time_series_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("动态时序降雨与入渗时间轴", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self._init_ui()

    def _init_ui(self):
        container = QWidget()
        layout = QHBoxLayout(container)

        grp_hyeto = QGroupBox("降雨强度-历时过程线 (Hyetograph)")
        v_h = QVBoxLayout(grp_hyeto)
        self.tbl_rain = QTableWidget(4, 2)
        self.tbl_rain.setHorizontalHeaderLabels(["历时时刻 (h)", "降雨强度 (mm/h)"])
        self.tbl_rain.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        default_hyeto = [(0.0, 10.0), (12.0, 35.0), (24.0, 60.0), (48.0, 5.0)]
        for r, (t, i_val) in enumerate(default_hyeto):
            self.tbl_rain.setItem(r, 0, QTableWidgetItem(str(t)))
            self.tbl_rain.setItem(r, 1, QTableWidgetItem(str(i_val)))
        v_h.addWidget(self.tbl_rain)
        layout.addWidget(grp_hyeto, stretch=3)

        grp_slider = QGroupBox("时变入渗求解与时间轴控制")
        v_s = QVBoxLayout(grp_slider)

        f_infil = QFormLayout()
        self.spin_ks = QDoubleSpinBox();
        self.spin_ks.setValue(15.0);
        self.spin_ks.setSuffix(" mm/h")
        self.spin_theta = QDoubleSpinBox();
        self.spin_theta.setValue(0.20);
        self.spin_theta.setSingleStep(0.05)
        f_infil.addRow("饱和渗透系数 (Ks):", self.spin_ks)
        f_infil.addRow("有效储水空隙增量 (Δθ):", self.spin_theta)
        v_s.addLayout(f_infil)

        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 48)
        self.slider_time.setValue(0)
        self.slider_time.valueChanged.connect(self._on_slider_moved)
        v_s.addWidget(self.slider_time)

        h_info = QHBoxLayout()
        self.lbl_time_val = QLabel("当前分析时刻: t = 0.0 h")
        self.lbl_time_val.setStyleSheet("font-weight: bold; color: #2980b9;")
        self.lbl_wet_depth = QLabel("计算湿润锋深度: zf = 0.00 m")
        self.lbl_wet_depth.setStyleSheet("font-weight: bold; color: #c0392b;")
        h_info.addWidget(self.lbl_time_val)
        h_info.addWidget(self.lbl_wet_depth)
        v_s.addLayout(h_info)

        btn_run_series = QPushButton("计算降雨全历时稳定性衰减曲线 Fs(t)")
        btn_run_series.setIcon(get_icon("rainfall_time"))
        btn_run_series.setStyleSheet("padding: 6px; font-weight: bold;")
        btn_run_series.clicked.connect(self.solve_time_series_requested.emit)
        v_s.addWidget(btn_run_series)

        layout.addWidget(grp_slider, stretch=4)
        self.setWidget(container)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series": [[float(t), float(i)] for t, i in self._get_time_series_data()],
            "ks_mm_h": self.spin_ks.value(),
            "delta_theta": self.spin_theta.value(),
            "current_time": self.slider_time.value(),
        }

    def from_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        series = data.get("series", [])
        if series:
            self.tbl_rain.setRowCount(len(series))
            for r, (t, i) in enumerate(series):
                self.tbl_rain.setItem(r, 0, QTableWidgetItem(str(t)))
                self.tbl_rain.setItem(r, 1, QTableWidgetItem(str(i)))

        self.spin_ks.setValue(float(data.get("ks_mm_h", 15.0)))
        self.spin_theta.setValue(float(data.get("delta_theta", 0.20)))
        self.slider_time.setValue(int(data.get("current_time", 0)))

    def _get_time_series_data(self) -> List[Tuple[float, float]]:
        series = []
        for r in range(self.tbl_rain.rowCount()):
            it = self.tbl_rain.item(r, 0)
            ii = self.tbl_rain.item(r, 1)
            if it and ii:
                try:
                    series.append((float(it.text()), float(ii.text())))
                except ValueError:
                    pass
        return sorted(series, key=lambda x: x[0]) if series else [(0.0, 0.0)]

    def get_rainfall_model(self) -> RainfallTimeSeries:
        series = self._get_time_series_data()
        return RainfallTimeSeries(series, self.spin_ks.value(), self.spin_theta.value())

    def _on_slider_moved(self, val):
        t_current = float(val)
        model = self.get_rainfall_model()
        depth = model.get_wetting_front_depth(t_current)
        self.lbl_time_val.setText(f"当前分析时刻: t = {t_current:.1f} h")
        self.lbl_wet_depth.setText(f"计算湿润锋深度: zf = {depth:.2f} m")
        self.time_step_changed.emit(t_current)

    def get_current_wetting_front_depth(self) -> float:
        t_current = float(self.slider_time.value())
        model = self.get_rainfall_model()
        return model.get_wetting_front_depth(t_current)


class LoadsDockWidget(QDockWidget):
    """荷载与抗震拟静力分析停靠窗"""
    loads_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("外荷载与抗震工况", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._init_ui()

    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        grp_load = QGroupBox("坡顶附加荷载")
        form_load = QFormLayout()
        self.spin_q = QDoubleSpinBox();
        self.spin_q.setRange(0.0, 500.0);
        self.spin_q.setValue(0.0);
        self.spin_q.setSuffix(" kPa")
        self.spin_qx1 = QDoubleSpinBox();
        self.spin_qx1.setRange(-50, 100);
        self.spin_qx1.setValue(5.0);
        self.spin_qx1.setSuffix(" m")
        self.spin_qx2 = QDoubleSpinBox();
        self.spin_qx2.setRange(-50, 100);
        self.spin_qx2.setValue(18.0);
        self.spin_qx2.setSuffix(" m")
        form_load.addRow("荷载强度 (q):", self.spin_q)
        form_load.addRow("分布起点 X1:", self.spin_qx1)
        form_load.addRow("分布终点 X2:", self.spin_qx2)
        grp_load.setLayout(form_load)
        layout.addWidget(grp_load)

        grp_seismic = QGroupBox("拟静力法地震惯性力")
        form_seismic = QFormLayout()
        self.spin_kh = QDoubleSpinBox();
        self.spin_kh.setRange(0.0, 0.4);
        self.spin_kh.setValue(0.0);
        self.spin_kh.setSingleStep(0.02)
        form_seismic.addRow("水平地震力系数 (kh):", self.spin_kh)
        grp_seismic.setLayout(form_seismic)
        layout.addWidget(grp_seismic)

        btn_apply = QPushButton("应用荷载设置")
        btn_apply.setIcon(get_icon("apply_surface"))
        btn_apply.clicked.connect(self.loads_changed.emit)
        layout.addWidget(btn_apply)

        layout.addStretch()
        self.setWidget(container)

    def get_surcharge_loads(self) -> List[Tuple[float, float, float]]:
        q = self.spin_q.value()
        if q > 0:
            return [(self.spin_qx1.value(), self.spin_qx2.value(), q)]
        return []

    def get_seismic_kh(self) -> float:
        return self.spin_kh.value()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q": self.spin_q.value(),
            "x1": self.spin_qx1.value(),
            "x2": self.spin_qx2.value(),
            "kh": self.spin_kh.value(),
        }

    def from_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        self.spin_q.setValue(float(data.get("q", 0.0)))
        self.spin_qx1.setValue(float(data.get("x1", 5.0)))
        self.spin_qx2.setValue(float(data.get("x2", 18.0)))
        self.spin_kh.setValue(float(data.get("kh", 0.0)))


class SearchDockWidget(QDockWidget):
    """支持圆弧与非圆弧折线双模式定义与全局寻优控制面板

    非圆弧模式支持两种行为:
      - 自动搜索 (chk_poly_auto_search 勾选): 表格里的点作为形状参考, DE 搜索最优
      - 手动试算 (chk_poly_auto_search 未勾选): 表格里的点就是最终滑面, 只算一次 Fs
    """
    calculate_requested = pyqtSignal()
    search_requested = pyqtSignal()
    search_stop_requested = pyqtSignal()
    # apply_searched_circle = pyqtSignal()
    preview_circle_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("滑面定义与全局稳定性寻优", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._init_ui()



    def _apply_compact_style(self):
        """让所有输入控件能被压缩, 避免内容被裁

        · QDoubleSpinBox / QComboBox / QLineEdit 忽略 sizeHint, 可缩到 50px
        · QLabel 允许裁剪, 不撑大布局
        """
        from PyQt5.QtWidgets import (QAbstractSpinBox, QComboBox,
                                      QLineEdit, QSizePolicy, QLabel)

        for w in self.findChildren(QAbstractSpinBox):
            w.setMinimumWidth(50)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for w in self.findChildren(QComboBox):
            w.setMinimumWidth(50)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for w in self.findChildren(QLineEdit):
            w.setMinimumWidth(50)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        # QTabWidget 内的各 tab: 允许压缩
        if hasattr(self, "tabs"):
            self.tabs.setMinimumWidth(0)
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if tab is not None:
                    tab.setMinimumWidth(0)
    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        # ---------- 1. 滑动面形态选择 ----------
        grp_shape = QGroupBox("1. 滑动面形态选择 (Slip Surface Shape)")
        f_shape = QFormLayout()
        self.combo_surface_type = QComboBox()
        self.combo_surface_type.addItems([
            "圆弧滑动面 (Circular)",
            "非圆弧多段折线面 (Polygonal / Non-Circular)"
        ])
        self.combo_surface_type.currentIndexChanged.connect(self._on_surface_type_changed)
        f_shape.addRow("滑面类型:", self.combo_surface_type)
        grp_shape.setLayout(f_shape)
        layout.addWidget(grp_shape)

        # ---------- 2. 圆弧几何参数 ----------
        self.grp_circle = QGroupBox("圆弧几何参数 (指定滑面)")
        form_circle = QFormLayout()
        self.spin_xc = QDoubleSpinBox();
        self.spin_xc.setRange(-200.0, 500.0);
        self.spin_xc.setValue(25.0);
        self.spin_xc.setSuffix(" m")
        self.spin_yc = QDoubleSpinBox();
        self.spin_yc.setRange(-200.0, 500.0);
        self.spin_yc.setValue(22.0);
        self.spin_yc.setSuffix(" m")
        self.spin_r = QDoubleSpinBox();
        self.spin_r.setRange(1.0, 500.0);
        self.spin_r.setValue(23.0);
        self.spin_r.setSuffix(" m")
        self.spin_slices = QSpinBox();
        self.spin_slices.setRange(10, 150);
        self.spin_slices.setValue(30);
        self.spin_slices.setSuffix(" 个")
        form_circle.addRow("滑弧圆心 Xc:", self.spin_xc)
        form_circle.addRow("滑弧圆心 Yc:", self.spin_yc)
        form_circle.addRow("滑弧半径 R:", self.spin_r)
        form_circle.addRow("切片土条数 N:", self.spin_slices)
        self.grp_circle.setLayout(form_circle)
        layout.addWidget(self.grp_circle)

        # ---------- 3. 非圆弧控制折点表 ----------
        self.grp_poly = QGroupBox("非圆弧控制折点表 (指定滑面)")
        l_poly = QVBoxLayout()

        # 3.1 双模式复选框
        self.chk_poly_auto_search = QCheckBox("启用自动搜索 (未勾选则仅按表中坐标试算)")
        self.chk_poly_auto_search.setChecked(True)
        self.chk_poly_auto_search.setToolTip(
            "✓ 勾选: 表格里的点作为形状参考, 算法在附近搜索最优滑面\n"
            "✗ 未勾选: 表格里的点就是最终滑面, 只算一次 Fs"
        )
        l_poly.addWidget(self.chk_poly_auto_search)

        # 3.2 折点表
        self.tbl_poly = QTableWidget(3, 2)
        self.tbl_poly.setHorizontalHeaderLabels(["X 坐标 (m)", "Y 高程 (m)"])
        default_poly = [(12.0, 15.0), (25.0, 4.0), (38.0, 0.0)]
        for r, (x, y) in enumerate(default_poly):
            self.tbl_poly.setItem(r, 0, QTableWidgetItem(str(x)))
            self.tbl_poly.setItem(r, 1, QTableWidgetItem(str(y)))
        self.tbl_poly.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_poly.itemChanged.connect(lambda _: self.preview_circle_changed.emit())
        l_poly.addWidget(self.tbl_poly)

        # 3.3 增删点按钮
        btn_poly_bar = QHBoxLayout()
        btn_add_p = QPushButton("添加折点")
        btn_add_p.clicked.connect(self._add_poly_point)
        btn_del_p = QPushButton("删除选中")
        btn_del_p.clicked.connect(self._del_poly_point)
        btn_poly_bar.addWidget(btn_add_p)
        btn_poly_bar.addWidget(btn_del_p)
        l_poly.addLayout(btn_poly_bar)

        self.grp_poly.setLayout(l_poly)
        self.grp_poly.setVisible(False)
        layout.addWidget(self.grp_poly)

        # ---------- 4. 预览按钮 ----------
        btn_preview = QPushButton("刷新并预览当前滑面网格")
        btn_preview.setIcon(get_icon("refresh"))
        btn_preview.clicked.connect(self.preview_circle_changed.emit)
        layout.addWidget(btn_preview)

        # ---------- 5. 选用力学模型 ----------
        grp_solver = QGroupBox("2. 选用求解模型 (Target Method)")
        f_solver = QFormLayout()
        self.combo_solver = QComboBox()
        self.combo_solver.addItems([
            "斯宾塞法 (Spencer 完全平衡法) [通用]",
            "摩根斯坦-普赖斯法 (Morgenstern-Price) [通用]",
            "简化让布法 (Simplified Janbu) [通用]",
            "简化毕肖普法 (Simplified Bishop) [仅圆弧]",
            "瑞典条分法 (Fellenius / Ordinary) [仅圆弧]"
        ])
        f_solver.addRow("力学模型:", self.combo_solver)
        grp_solver.setLayout(f_solver)
        layout.addWidget(grp_solver)

        # ---------- 6. 全局寻优设置 ----------
        self.grp_algo = QGroupBox("3. 最危险滑面全局寻优")
        self.form_algo = QFormLayout()
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "差分进化算法 (Differential Evolution, 推荐)",
            "粒子群优化算法 (PSO)",
            "模拟退火算法 (Simulated Annealing)",
            "单纯形搜索法 (Nelder-Mead)",
            "传统网格扫描法 (Grid Search)"
        ])
        self.combo_eval = QComboBox()
        self.combo_eval.addItems([
            "Spencer 完全平衡法",
            "Simplified Bishop (圆弧)",
            "Simplified Janbu"
        ])
        self.form_algo.addRow("寻优算法:", self.combo_algo)
        self.form_algo.addRow("寻优评价模型:", self.combo_eval)

        # 范围包络控件 (自适应圆弧/折线标签)
        self.lbl_b1 = QLabel("Xc 搜索范围 (m):")
        self.lbl_b2 = QLabel("Yc 搜索范围 (m):")
        self.lbl_b3 = QLabel("半径 R 范围 (m):")
        self.spin_b1_min = QDoubleSpinBox();
        self.spin_b1_min.setRange(-500, 500);
        self.spin_b1_min.setValue(10.0)
        self.spin_b1_max = QDoubleSpinBox();
        self.spin_b1_max.setRange(-500, 500);
        self.spin_b1_max.setValue(45.0)
        self.spin_b2_min = QDoubleSpinBox();
        self.spin_b2_min.setRange(-500, 500);
        self.spin_b2_min.setValue(15.0)
        self.spin_b2_max = QDoubleSpinBox();
        self.spin_b2_max.setRange(-500, 500);
        self.spin_b2_max.setValue(40.0)
        self.spin_b3_min = QDoubleSpinBox();
        self.spin_b3_min.setRange(-500, 500);
        self.spin_b3_min.setValue(10.0)
        self.spin_b3_max = QDoubleSpinBox();
        self.spin_b3_max.setRange(-500, 500);
        self.spin_b3_max.setValue(45.0)

        h_b1 = QHBoxLayout();
        h_b1.addWidget(self.spin_b1_min);
        h_b1.addWidget(QLabel("~"));
        h_b1.addWidget(self.spin_b1_max)
        h_b2 = QHBoxLayout();
        h_b2.addWidget(self.spin_b2_min);
        h_b2.addWidget(QLabel("~"));
        h_b2.addWidget(self.spin_b2_max)
        h_b3 = QHBoxLayout();
        h_b3.addWidget(self.spin_b3_min);
        h_b3.addWidget(QLabel("~"));
        h_b3.addWidget(self.spin_b3_max)

        self.form_algo.addRow(self.lbl_b1, h_b1)
        self.form_algo.addRow(self.lbl_b2, h_b2)
        self.form_algo.addRow(self.lbl_b3, h_b3)
        self.grp_algo.setLayout(self.form_algo)
        layout.addWidget(self.grp_algo)

        # ---------- 7. 控制按钮 + 进度条 + 结果 ----------
        h_btn_search = QHBoxLayout()
        self.btn_search = QPushButton("启动最危险滑面搜索")
        self.btn_search.setIcon(get_icon("search_surface"))
        self.btn_search.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 6px;")
        self.btn_search.clicked.connect(self.search_requested.emit)

        self.btn_stop = QPushButton("终止搜索")
        self.btn_stop.setIcon(get_icon("stop_search"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.search_stop_requested.emit)

        h_btn_search.addWidget(self.btn_search)
        h_btn_search.addWidget(self.btn_stop)
        layout.addLayout(h_btn_search)

        self.prog_bar = QProgressBar()
        self.prog_bar.setValue(0)
        layout.addWidget(self.prog_bar)

        self.lbl_best = QLabel("最危险滑面: 未搜索")
        self.lbl_best.setStyleSheet("font-weight: bold; color: #c0392b; font-size: 13px;")
        layout.addWidget(self.lbl_best)

        # self.btn_apply = QPushButton("应用最危险滑面至模型")
        # self.btn_apply.setIcon(get_icon("apply_surface"))
        # self.btn_apply.clicked.connect(self.apply_searched_circle.emit)
        # layout.addWidget(self.btn_apply)

        # ---------- 8. 定向执行分析主按钮 ----------
        self.btn_run_all = QPushButton("执行稳定性分析 (所选模型)")
        self.btn_run_all.setIcon(get_icon("run_solvers"))
        self.btn_run_all.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; font-size: 13px; padding: 8px;")
        self.btn_run_all.clicked.connect(self.calculate_requested.emit)
        layout.addWidget(self.btn_run_all)

        layout.addStretch()

        # ---------- 9. 滚动容器 ----------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(container)
        self.setWidget(scroll)

        # 允许 dock 缩到 220px
        self.setMinimumWidth(220)

        # ★ 让内部控件能被压缩
        self._apply_compact_style()
    # ------------------------------------------------------------------
    # 滑面类型切换
    # ------------------------------------------------------------------
    def _on_surface_type_changed(self, idx: int):
        is_circle = (idx == 0)

        # 1. 切换指定滑面参数面板
        if hasattr(self, "grp_circle"):
            self.grp_circle.setVisible(is_circle)
        if hasattr(self, "grp_poly"):
            self.grp_poly.setVisible(not is_circle)

        # 2. 寻优面板始终启用，仅动态更新标题
        if hasattr(self, "grp_algo"):
            self.grp_algo.setEnabled(True)
            if is_circle:
                self.grp_algo.setTitle("3. 最危险滑面全局寻优 (圆弧: Xc, Yc, R 搜索范围)")
            else:
                self.grp_algo.setTitle("3. 最危险滑面全局寻优 (非圆弧: 端点X, 底标高Y 搜索范围)")

        # 3. 求解模型适用性过滤
        if hasattr(self, "combo_solver"):
            for i in range(self.combo_solver.count()):
                txt = self.combo_solver.itemText(i)
                if "仅圆弧" in txt:
                    self.combo_solver.model().item(i).setEnabled(is_circle)
            if not is_circle and "仅圆弧" in self.combo_solver.currentText():
                self.combo_solver.setCurrentIndex(0)

        # 4. 双模式复选框仅对非圆弧可用
        if hasattr(self, "chk_poly_auto_search"):
            self.chk_poly_auto_search.setEnabled(not is_circle)

        # 5. 更新 bounds 标签
        if hasattr(self, "lbl_b1"):
            if is_circle:
                self.lbl_b1.setText("Xc 搜索范围 (m):")
                self.lbl_b2.setText("Yc 搜索范围 (m):")
                self.lbl_b3.setText("半径 R 范围 (m):")
            else:
                self.lbl_b1.setText("后缘 X 范围 (m):")
                self.lbl_b2.setText("滑底 Y 范围 (m):")
                self.lbl_b3.setText("坡脚 X 范围 (m):")

        # 6. 触发剖分与渲染
        self.preview_circle_changed.emit()

    # ------------------------------------------------------------------
    # 折点表操作
    # ------------------------------------------------------------------
    def _add_poly_point(self):
        row = self.tbl_poly.rowCount()
        self.tbl_poly.insertRow(row)
        self.tbl_poly.setItem(row, 0, QTableWidgetItem("30.0"))
        self.tbl_poly.setItem(row, 1, QTableWidgetItem("2.0"))
        self.preview_circle_changed.emit()

    def _del_poly_point(self):
        cur = self.tbl_poly.currentRow()
        if cur >= 0 and self.tbl_poly.rowCount() > 2:
            self.tbl_poly.removeRow(cur)
            self.preview_circle_changed.emit()

    def get_poly_points(self):
        """从折点表读取所有有效点 (按 X 升序)"""
        pts = []
        for r in range(self.tbl_poly.rowCount()):
            try:
                x_item = self.tbl_poly.item(r, 0)
                y_item = self.tbl_poly.item(r, 1)
                if x_item is None or y_item is None:
                    continue
                x = float(x_item.text())
                y = float(y_item.text())
                pts.append((x, y))
            except (ValueError, AttributeError):
                continue
        pts.sort(key=lambda p: p[0])
        return pts

    def set_poly_points(self, points):
        """把 [(x,y), ...] 回填到折点表"""
        self.tbl_poly.blockSignals(True)
        self.tbl_poly.setRowCount(len(points))
        for r, (x, y) in enumerate(points):
            self.tbl_poly.setItem(r, 0, QTableWidgetItem(f"{float(x):.3f}"))
            self.tbl_poly.setItem(r, 1, QTableWidgetItem(f"{float(y):.3f}"))
        self.tbl_poly.blockSignals(False)
        self.preview_circle_changed.emit()

    # ------------------------------------------------------------------
    # 双模式开关
    # ------------------------------------------------------------------
    def get_poly_auto_search(self) -> bool:
        return bool(self.chk_poly_auto_search.isChecked())

    def set_poly_auto_search(self, flag: bool):
        self.chk_poly_auto_search.setChecked(bool(flag))

    # ------------------------------------------------------------------
    # 供 MainWindow 调用的接口
    # ------------------------------------------------------------------
    def get_surface_type(self) -> str:
        return "circular" if self.combo_surface_type.currentIndex() == 0 else "polygonal"

    def get_circle_params(self):
        return (self.spin_xc.value(), self.spin_yc.value(),
                self.spin_r.value(), self.spin_slices.value())

    def set_circle_params(self, xc: float, yc: float, R: float):
        self.spin_xc.setValue(xc)
        self.spin_yc.setValue(yc)
        self.spin_r.setValue(R)

    def get_slip_surface(self):
        """获取当前界面的滑面对象"""
        from core.slip_surface import CircularSlipSurface, PolygonalSlipSurface
        if self.combo_surface_type.currentIndex() == 0:
            return CircularSlipSurface(
                self.spin_xc.value(), self.spin_yc.value(), self.spin_r.value()
            )
        else:
            pts = self.get_poly_points()
            if len(pts) < 2:
                pts = [(12.0, 15.0), (25.0, 4.0), (38.0, 0.0)]
            return PolygonalSlipSurface(pts)

    def get_selected_method_name(self) -> str:
        return self.combo_solver.currentText()

    def get_search_config(self):
        stype = self.get_surface_type()
        algo_name = self.combo_algo.currentText()

        txt = self.combo_eval.currentText()
        if "Janbu" in txt:
            eval_name = "Janbu"
        elif "Bishop" in txt:
            eval_name = "Bishop"
        elif "Spencer" in txt:
            eval_name = "Spencer"
        else:
            eval_name = "Spencer"

        bounds = [
            (self.spin_b1_min.value(), self.spin_b1_max.value()),
            (self.spin_b2_min.value(), self.spin_b2_max.value()),
            (self.spin_b3_min.value(), self.spin_b3_max.value()),
        ]
        ref_pts = self.get_poly_points()
        auto_search = self.get_poly_auto_search()
        return stype, algo_name, eval_name, bounds, ref_pts, auto_search

    # ------------------------------------------------------------------
    # 工程文件持久化
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "surface_type": self.get_surface_type(),
            "circle": {
                "xc": self.spin_xc.value(),
                "yc": self.spin_yc.value(),
                "R": self.spin_r.value(),
                "n_slices": self.spin_slices.value(),
            },
            "polygon": [list(p) for p in self.get_poly_points()],
            "poly_auto_search": self.get_poly_auto_search(),
            "search_bounds": {
                "b1": [self.spin_b1_min.value(), self.spin_b1_max.value()],
                "b2": [self.spin_b2_min.value(), self.spin_b2_max.value()],
                "b3": [self.spin_b3_min.value(), self.spin_b3_max.value()],
            },
            "search_algo": self.combo_algo.currentIndex(),
            "search_eval": self.combo_eval.currentIndex(),
            "solver_method": self.combo_solver.currentIndex(),
        }

    def from_dict(self, data):
        if not isinstance(data, dict):
            return

        # 滑面类型
        stype = data.get("surface_type", "circular")
        self.combo_surface_type.setCurrentIndex(0 if stype == "circular" else 1)

        # 圆弧参数
        c = data.get("circle", {}) or {}
        self.spin_xc.setValue(float(c.get("xc", 25.0)))
        self.spin_yc.setValue(float(c.get("yc", 22.0)))
        self.spin_r.setValue(float(c.get("R", 23.0)))
        self.spin_slices.setValue(int(c.get("n_slices", 30)))

        # 折线点
        poly = data.get("polygon", [])
        if poly:
            self.tbl_poly.blockSignals(True)
            self.tbl_poly.setRowCount(len(poly))
            for r, (x, y) in enumerate(poly):
                self.tbl_poly.setItem(r, 0, QTableWidgetItem(f"{float(x):.3f}"))
                self.tbl_poly.setItem(r, 1, QTableWidgetItem(f"{float(y):.3f}"))
            self.tbl_poly.blockSignals(False)

        # 双模式
        if "poly_auto_search" in data:
            self.set_poly_auto_search(bool(data["poly_auto_search"]))

        # 搜索包络
        b = data.get("search_bounds", {}) or {}
        for key, spin_min, spin_max in (
                ("b1", self.spin_b1_min, self.spin_b1_max),
                ("b2", self.spin_b2_min, self.spin_b2_max),
                ("b3", self.spin_b3_min, self.spin_b3_max),
        ):
            pair = b.get(key)
            if pair and len(pair) == 2:
                spin_min.setValue(float(pair[0]))
                spin_max.setValue(float(pair[1]))

        # 算法与模型
        if "search_algo" in data:
            self.combo_algo.setCurrentIndex(int(data["search_algo"]))
        if "search_eval" in data:
            self.combo_eval.setCurrentIndex(int(data["search_eval"]))
        if "solver_method" in data:
            self.combo_solver.setCurrentIndex(int(data["solver_method"]))

        # 让界面根据类型刷新
        self._on_surface_type_changed(self.combo_surface_type.currentIndex())


class ResultsDockWidget(QDockWidget):
    """计算结果、降雨时程与微元表格停靠窗"""

    def __init__(self, parent=None):
        super().__init__("计算成果与切片受力核查", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self._init_ui()

    def _init_ui(self):
        self.tabs = QTabWidget()

        self.tbl_summary = QTableWidget(5, 3)
        self.tbl_summary.setHorizontalHeaderLabels(["极限平衡求解方法", "稳定安全系数 (Fs)", "平衡条件与收敛状态"])
        self.tbl_summary.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl_summary, "经典 LEM 模型对比表")

        self.tbl_time_series = QTableWidget(0, 4)
        self.tbl_time_series.setHorizontalHeaderLabels(
            ["降雨历时 t (h)", "降雨强度 (mm/h)", "湿润锋深度 zf (m)", "安全系数 Fs (Bishop)"])
        self.tbl_time_series.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl_time_series, "降雨历时稳定性衰减 Fs(t)")

        self.tbl_slices = QTableWidget(0, 15)
        self.tbl_slices.setHorizontalHeaderLabels([
            "条号", "所属土层", "中点X(m)", "条宽b(m)", "高度h(m)",
            "土自重(kN)", "外载荷(kN)", "总竖力W(kN)", "地震力Fh(kN)",
            "孔压u(kPa)", "基质吸力(kPa)", "总黏聚力(kPa)", "摩擦角(°)", "底坡角α(°)", "底斜长l(m)"
        ])
        self.tbl_slices.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tabs.addTab(self.tbl_slices, "离散土条微元多物理量明细")

        self.setWidget(self.tabs)

    def display_summary(self, results):
        # 按 FS 升序排 (最危险的在最上面; None 排最后)
        sorted_results = sorted(
            results,
            key=lambda kv: (kv[1][0] is None,
                            kv[1][0] if kv[1][0] is not None else 999)
        )
        self.tbl_summary.setRowCount(len(sorted_results))
        for r, (name, (fs, note)) in enumerate(sorted_results):
            self.tbl_summary.setItem(r, 0, QTableWidgetItem(name))
            fs_text = f"{fs:.4f}" if fs is not None else "未收敛"
            item_fs = QTableWidgetItem(fs_text)
            item_fs.setTextAlignment(Qt.AlignCenter)
            self.tbl_summary.setItem(r, 1, item_fs)
            self.tbl_summary.setItem(r, 2, QTableWidgetItem(note))

    def display_time_series(self, time_results: List[Tuple[float, float, float, float]]):
        self.tbl_time_series.setRowCount(len(time_results))
        for r, (t, intensity, depth, fs) in enumerate(time_results):
            self.tbl_time_series.setItem(r, 0, QTableWidgetItem(f"{t:.1f}"))
            self.tbl_time_series.setItem(r, 1, QTableWidgetItem(f"{intensity:.1f}"))
            self.tbl_time_series.setItem(r, 2, QTableWidgetItem(f"{depth:.2f}"))
            item_fs = QTableWidgetItem(f"{fs:.4f}" if fs > 0 else "未收敛")
            item_fs.setTextAlignment(Qt.AlignCenter)
            self.tbl_time_series.setItem(r, 3, item_fs)
        self.tabs.setCurrentWidget(self.tbl_time_series)

    def display_slices(self, slices: Optional[List[Slice]]):
        if not slices:
            self.tbl_slices.setRowCount(0)
            return

        self.tbl_slices.setRowCount(len(slices))
        for r, s in enumerate(slices):
            self.tbl_slices.setItem(r, 0, QTableWidgetItem(str(s.index)))
            self.tbl_slices.setItem(r, 1, QTableWidgetItem(str(s.layer_name)))
            self.tbl_slices.setItem(r, 2, QTableWidgetItem(f"{s.xm:.2f}"))
            self.tbl_slices.setItem(r, 3, QTableWidgetItem(f"{s.b:.2f}"))
            self.tbl_slices.setItem(r, 4, QTableWidgetItem(f"{s.h:.2f}"))
            self.tbl_slices.setItem(r, 5, QTableWidgetItem(f"{s.W_soil:.2f}"))
            self.tbl_slices.setItem(r, 6, QTableWidgetItem(f"{s.q_load:.2f}"))
            self.tbl_slices.setItem(r, 7, QTableWidgetItem(f"{s.W:.2f}"))
            self.tbl_slices.setItem(r, 8, QTableWidgetItem(f"{s.Fh:.2f}"))
            self.tbl_slices.setItem(r, 9, QTableWidgetItem(f"{s.u:.2f}"))
            self.tbl_slices.setItem(r, 10, QTableWidgetItem(f"{s.suction:.2f}"))
            self.tbl_slices.setItem(r, 11, QTableWidgetItem(f"{s.c:.2f}"))
            self.tbl_slices.setItem(r, 12, QTableWidgetItem(f"{np.degrees(s.phi):.1f}"))
            self.tbl_slices.setItem(r, 13, QTableWidgetItem(f"{np.degrees(s.alpha):.2f}"))
            self.tbl_slices.setItem(r, 14, QTableWidgetItem(f"{s.l:.2f}"))

    def get_poly_points(self):
        """从折线表格读回所有点 (供保存使用)"""
        pts = []
        for r in range(self.tbl_poly.rowCount()):
            try:
                x = float(self.tbl_poly.item(r, 0).text())
                y = float(self.tbl_poly.item(r, 1).text())
                pts.append((x, y))
            except Exception:
                continue
        return pts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_type": self.get_surface_type(),
            "circle": {
                "xc": self.spin_xc.value(),
                "yc": self.spin_yc.value(),
                "R": self.spin_r.value(),
                "n_slices": self.spin_slices.value(),
            },
            "polygon": [list(p) for p in self.get_poly_points()],
            "search_bounds": {
                "b1": [self.spin_b1_min.value(), self.spin_b1_max.value()],
                "b2": [self.spin_b2_min.value(), self.spin_b2_max.value()],
                "b3": [self.spin_b3_min.value(), self.spin_b3_max.value()],
            },
            "search_algo": self.combo_algo.currentIndex(),
            "search_eval": self.combo_eval.currentIndex(),
            "solver_method": self.combo_solver.currentIndex(),
        }

    def from_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return

        # 滑面类型
        stype = data.get("surface_type", "circular")
        self.combo_surface_type.setCurrentIndex(0 if stype == "circular" else 1)

        # 圆弧参数
        c = data.get("circle", {}) or {}
        self.spin_xc.setValue(float(c.get("xc", 25.0)))
        self.spin_yc.setValue(float(c.get("yc", 22.0)))
        self.spin_r.setValue(float(c.get("R", 23.0)))
        self.spin_slices.setValue(int(c.get("n_slices", 30)))

        # 折线点
        poly = data.get("polygon", [])
        if poly:
            self.tbl_poly.blockSignals(True)
            self.tbl_poly.setRowCount(len(poly))
            for r, (x, y) in enumerate(poly):
                self.tbl_poly.setItem(r, 0, QTableWidgetItem(f"{float(x):.3f}"))
                self.tbl_poly.setItem(r, 1, QTableWidgetItem(f"{float(y):.3f}"))
            self.tbl_poly.blockSignals(False)

        # 搜索包络
        b = data.get("search_bounds", {}) or {}
        for key, spin_min, spin_max in (
                ("b1", self.spin_b1_min, self.spin_b1_max),
                ("b2", self.spin_b2_min, self.spin_b2_max),
                ("b3", self.spin_b3_min, self.spin_b3_max),
        ):
            pair = b.get(key, None)
            if pair and len(pair) == 2:
                spin_min.setValue(float(pair[0]))
                spin_max.setValue(float(pair[1]))

        # 算法与模型选择
        if "search_algo" in data:
            self.combo_algo.setCurrentIndex(int(data["search_algo"]))
        if "search_eval" in data:
            self.combo_eval.setCurrentIndex(int(data["search_eval"]))
        if "solver_method" in data:
            self.combo_solver.setCurrentIndex(int(data["solver_method"]))

        # 让界面根据类型刷新一次
        self._on_surface_type_changed(self.combo_surface_type.currentIndex())


class ReliabilityDockWidget(QDockWidget):
    """边坡可靠度指标与失效概率评价停靠窗 (CAD/CAE 风格)"""
    reliability_requested = pyqtSignal()
    reliability_stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("边坡可靠度与失效概率评价", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea | Qt.BottomDockWidgetArea)
        self._init_ui()

    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. 评价算法与抽样规模设置
        grp_method = QGroupBox("可靠度评价算法与抽样规模")
        f_method = QFormLayout()

        self.combo_rel_method = QComboBox()
        self.combo_rel_method.addItems([
            "蒙特卡洛模拟法 (MCS, 随机抽样)",
            "Rosenblueth 点估计法 (PEM, 快速核算)"
        ])
        self.combo_rel_method = QComboBox()
        self.combo_rel_method.addItems([
            "蒙特卡洛模拟法 (MCS, 随机抽样)",
            "Rosenblueth 点估计法 (PEM, 快速核算)"
        ])

        # ★ 补回抽样次数控件
        self.spin_n_sim = QSpinBox()
        self.spin_n_sim.setRange(100, 200000)
        self.spin_n_sim.setValue(5000)
        self.spin_n_sim.setSingleStep(1000)
        self.spin_n_sim.setSuffix(" 次")

        # self.lbl_inherit_model = QLabel("（自动继承滑面寻优面板）")
        # self.lbl_inherit_model.setStyleSheet("color: #7f8c8d; font-style: italic;")
        # f_method.addRow("求解模型:", self.lbl_inherit_model)
        # f_method.addRow("MCS 抽样次数:", self.spin_n_sim)

        self.lbl_inherit_model = QLabel("（自动继承滑面寻优面板）")
        self.lbl_inherit_model.setStyleSheet("color: #7f8c8d; font-style: italic;")
        f_method.addRow("求解模型:", self.lbl_inherit_model)
        f_method.addRow("MCS 抽样次数:", self.spin_n_sim)
        grp_method.setLayout(f_method)
        layout.addWidget(grp_method)

        # 2. 岩土力学参数概率统计特征
        grp_stats = QGroupBox("岩土力学参数概率统计特征")
        f_stats = QFormLayout()

        self.spin_cov_c = QDoubleSpinBox()
        self.spin_cov_c.setRange(0.01, 1.0)
        self.spin_cov_c.setValue(0.25)
        self.spin_cov_c.setSingleStep(0.05)

        self.combo_dist_c = QComboBox()
        self.combo_dist_c.addItems(["对数正态分布", "正态分布"])

        self.spin_cov_phi = QDoubleSpinBox()
        self.spin_cov_phi.setRange(0.01, 1.0)
        self.spin_cov_phi.setValue(0.15)
        self.spin_cov_phi.setSingleStep(0.05)

        self.combo_dist_phi = QComboBox()
        self.combo_dist_phi.addItems(["对数正态分布", "正态分布"])

        self.spin_rho = QDoubleSpinBox()
        self.spin_rho.setRange(-0.95, 0.95)
        self.spin_rho.setValue(-0.50)
        self.spin_rho.setSingleStep(0.05)

        self.spin_cov_gamma = QDoubleSpinBox()
        self.spin_cov_gamma.setRange(0.01, 0.5)
        self.spin_cov_gamma.setValue(0.08)
        self.spin_cov_gamma.setSingleStep(0.02)

        f_stats.addRow("黏聚力 c' 变异系数 (COV):", self.spin_cov_c)
        f_stats.addRow("黏聚力 c' 概率分布:", self.combo_dist_c)
        f_stats.addRow("摩擦角 φ' 变异系数 (COV):", self.spin_cov_phi)
        f_stats.addRow("摩擦角 φ' 概率分布:", self.combo_dist_phi)
        f_stats.addRow("c'-φ' 互相关系数 (ρ):", self.spin_rho)
        f_stats.addRow("天然重度 γ 变异系数 (COV):", self.spin_cov_gamma)
        grp_stats.setLayout(f_stats)
        layout.addWidget(grp_stats)

        # 3. 运行控制与实时进度条
        h_btn = QHBoxLayout()
        self.btn_run_rel = QPushButton("启动失效概率评价")
        ico_run = get_icon("run_solvers")
        if ico_run:
            self.btn_run_rel.setIcon(ico_run)
        self.btn_run_rel.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 6px;")
        self.btn_run_rel.clicked.connect(self.reliability_requested.emit)

        self.btn_stop_rel = QPushButton("终止评价")
        ico_stop = get_icon("stop_search")
        if ico_stop:
            self.btn_stop_rel.setIcon(ico_stop)
        self.btn_stop_rel.setEnabled(False)
        self.btn_stop_rel.setStyleSheet("padding: 6px;")
        self.btn_stop_rel.clicked.connect(self.reliability_stop_requested.emit)

        h_btn.addWidget(self.btn_run_rel)
        h_btn.addWidget(self.btn_stop_rel)
        layout.addLayout(h_btn)

        self.prog_bar = QProgressBar()
        self.prog_bar.setValue(0)
        layout.addWidget(self.prog_bar)

        # 4. 评价成果数据展示卡片
        grp_res = QGroupBox("边坡可靠度与风险成果指标")
        f_res = QFormLayout()

        self.lbl_pf = QLabel("未分析")
        self.lbl_pf.setStyleSheet("font-size: 15px; font-weight: bold; color: #c0392b;")
        self.lbl_beta = QLabel("未分析")
        self.lbl_beta.setStyleSheet("font-size: 15px; font-weight: bold; color: #2980b9;")
        self.lbl_fs_mean_std = QLabel("未分析")
        self.lbl_fs_range = QLabel("未分析")
        self.lbl_fail_counts = QLabel("未分析")

        f_res.addRow("失稳破坏概率 (Pf):", self.lbl_pf)
        f_res.addRow("可靠度指标 (β):", self.lbl_beta)
        f_res.addRow("安全系数均值与标准差:", self.lbl_fs_mean_std)
        f_res.addRow("抽样极值范围 [Min, Max]:", self.lbl_fs_range)
        f_res.addRow("失效破坏样本统计:", self.lbl_fail_counts)
        grp_res.setLayout(f_res)
        layout.addWidget(grp_res)

        layout.addStretch()
        self.setWidget(container)

    def get_config(self) -> Dict[str, Any]:
        """提取界面控件参数供后端求解器使用"""
        return {
            "method": "MCS" if "蒙特卡洛" in self.combo_rel_method.currentText() else "PEM",
            # "eval_method": "Bishop" if "Bishop" in self.combo_eval_method.currentText() else "Fellenius",
            "n_samples": self.spin_n_sim.value(),
            "cov_c": self.spin_cov_c.value(),
            "dist_c": self.combo_dist_c.currentText(),
            "cov_phi": self.spin_cov_phi.value(),
            "dist_phi": self.combo_dist_phi.currentText(),
            "rho": self.spin_rho.value(),
            "cov_gamma": self.spin_cov_gamma.value()
        }

    def display_results(self, res: Dict[str, Any]):
        """渲染呈现计算成果"""
        pf_pct = res.get("pf_percent", 0.0)
        beta = res.get("beta", 0.0)
        mean_fs = res.get("fs_mean", 0.0)
        std_fs = res.get("fs_std", 0.0)
        cov_fs = res.get("fs_cov", 0.0)

        self.lbl_pf.setText(f"{pf_pct:.2f}% (P_f = {res.get('pf', 0.0):.4e})")
        self.lbl_beta.setText(f"β = {beta:.3f}")
        self.lbl_fs_mean_std.setText(f"μ = {mean_fs:.3f}, σ = {std_fs:.3f} (COV={cov_fs:.2f})")

        if "fs_min" in res:
            self.lbl_fs_range.setText(f"[{res['fs_min']:.3f}, {res['fs_max']:.3f}]")
            self.lbl_fail_counts.setText(f"{res['failure_count']} 次失效 / {res['n_valid']} 次有效抽样")
        else:
            self.lbl_fs_range.setText("点估计法不提供极值样本")
            self.lbl_fail_counts.setText("解析点估计近似估算")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel_method": self.combo_rel_method.currentIndex(),
            # "eval_method": self.combo_eval_method.currentIndex(),
            "n_samples": self.spin_n_sim.value(),
            "cov_c": self.spin_cov_c.value(),
            "dist_c": self.combo_dist_c.currentIndex(),
            "cov_phi": self.spin_cov_phi.value(),
            "dist_phi": self.combo_dist_phi.currentIndex(),
            "rho": self.spin_rho.value(),
            "cov_gamma": self.spin_cov_gamma.value(),
        }

    def from_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        if "rel_method" in data:   self.combo_rel_method.setCurrentIndex(int(data["rel_method"]))
        # if "eval_method" in data:  self.combo_eval_method.setCurrentIndex(int(data["eval_method"]))
        if "n_samples" in data:    self.spin_n_sim.setValue(int(data["n_samples"]))
        if "cov_c" in data:        self.spin_cov_c.setValue(float(data["cov_c"]))
        if "dist_c" in data:       self.combo_dist_c.setCurrentIndex(int(data["dist_c"]))
        if "cov_phi" in data:      self.spin_cov_phi.setValue(float(data["cov_phi"]))
        if "dist_phi" in data:     self.combo_dist_phi.setCurrentIndex(int(data["dist_phi"]))
        if "rho" in data:          self.spin_rho.setValue(float(data["rho"]))
        if "cov_gamma" in data:    self.spin_cov_gamma.setValue(float(data["cov_gamma"]))


class ReinforcementDockWidget(QDockWidget):
    """边坡防治措施与支护构件管理停靠窗"""
    reinforcements_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("边坡防治措施与支护构件", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(360)
        self._init_ui()

    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        lbl_tip = QLabel(
            "配置边坡支护措施，用于计算加固后的稳定性安全系数。\n"
            "勾选下方开关后，支护反力将参与 LEM 求解。"
        )
        lbl_tip.setWordWrap(True)
        lbl_tip.setStyleSheet("color: #555; padding: 4px; font-size: 11px;")
        layout.addWidget(lbl_tip)

        self.tabs = QTabWidget()

        # ================ 锚索 Tab ================
        tab_anchor = QWidget()
        v_a = QVBoxLayout(tab_anchor)

        self.tbl_anchor = QTableWidget(0, 7)
        self.tbl_anchor.setHorizontalHeaderLabels([
            "编号", "头部 X (m)", "头部 Y (m)", "倾角 (°)",
            "长度 (m)", "预应力 (kN)", "间距 (m)"
        ])
        self.tbl_anchor.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_anchor.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_anchor.itemChanged.connect(lambda _: self.reinforcements_changed.emit())
        v_a.addWidget(self.tbl_anchor)

        h_a = QHBoxLayout()
        btn_add_a = QPushButton("添加锚索")
        btn_add_a.setIcon(get_icon("add_item"))
        btn_add_a.clicked.connect(self._add_anchor)

        btn_del_a = QPushButton("删除选中")
        btn_del_a.setIcon(get_icon("del_item"))
        btn_del_a.clicked.connect(lambda: self._del_row(self.tbl_anchor))

        h_a.addWidget(btn_add_a)
        h_a.addWidget(btn_del_a)
        v_a.addLayout(h_a)

        self.tabs.addTab(tab_anchor, "锚索 / 锚杆")

        # ================ 抗滑桩 Tab ================
        tab_pile = QWidget()
        v_p = QVBoxLayout(tab_pile)

        self.tbl_pile = QTableWidget(0, 6)
        self.tbl_pile.setHorizontalHeaderLabels([
            "编号", "桩位 X (m)", "桩顶 Y (m)", "桩底 Y (m)",
            "桩径 (m)", "桩间距 (m)"
        ])
        self.tbl_pile.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_pile.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_pile.itemChanged.connect(lambda _: self.reinforcements_changed.emit())
        v_p.addWidget(self.tbl_pile)

        h_p = QHBoxLayout()
        btn_add_p = QPushButton("添加桩排")
        btn_add_p.setIcon(get_icon("add_item"))
        btn_add_p.clicked.connect(self._add_pile)

        btn_del_p = QPushButton("删除选中")
        btn_del_p.setIcon(get_icon("del_item"))
        btn_del_p.clicked.connect(lambda: self._del_row(self.tbl_pile))

        h_p.addWidget(btn_add_p)
        h_p.addWidget(btn_del_p)
        v_p.addLayout(h_p)

        self.tabs.addTab(tab_pile, "抗滑桩")

        # ================ 挡土墙 Tab ================
        tab_wall = QWidget()
        v_w = QVBoxLayout(tab_wall)

        self.tbl_wall = QTableWidget(0, 7)
        self.tbl_wall.setHorizontalHeaderLabels([
            "编号", "墙位 X (m)", "墙顶 Y (m)", "墙底 Y (m)",
            "顶宽 (m)", "底宽 (m)", "墙前摩擦角 (°)"
        ])
        self.tbl_wall.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_wall.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_wall.itemChanged.connect(lambda _: self.reinforcements_changed.emit())
        v_w.addWidget(self.tbl_wall)

        h_w = QHBoxLayout()
        btn_add_w = QPushButton("添加挡土墙")
        btn_add_w.setIcon(get_icon("add_item"))
        btn_add_w.clicked.connect(self._add_wall)

        btn_del_w = QPushButton("删除选中")
        btn_del_w.setIcon(get_icon("del_item"))
        btn_del_w.clicked.connect(lambda: self._del_row(self.tbl_wall))

        h_w.addWidget(btn_add_w)
        h_w.addWidget(btn_del_w)
        v_w.addLayout(h_w)

        self.tabs.addTab(tab_wall, "挡土墙")

        layout.addWidget(self.tabs)

        # 全局开关
        self.chk_enable_all = QCheckBox("启用支护反力（参与稳定性计算）")
        self.chk_enable_all.setChecked(True)
        self.chk_enable_all.stateChanged.connect(
            lambda _: self.reinforcements_changed.emit()
        )
        layout.addWidget(self.chk_enable_all)

        # 应用按钮
        btn_apply = QPushButton("应用支护设置")
        btn_apply.setIcon(get_icon("apply_surface"))
        btn_apply.setStyleSheet(
            "background-color: #8e44ad; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        btn_apply.clicked.connect(self.reinforcements_changed.emit)
        layout.addWidget(btn_apply)

        layout.addStretch()

        # ================ 用 QScrollArea 包一层，内容能滚动 ================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        self.setWidget(scroll)

    # ---------------- 增删行 ----------------
    def _add_anchor(self):
        row = self.tbl_anchor.rowCount()
        self.tbl_anchor.insertRow(row)
        defaults = [
            str(row + 1), "20.0", "14.0", "15.0", "25.0", "500.0", "3.0"
        ]
        for c, v in enumerate(defaults):
            self.tbl_anchor.setItem(row, c, QTableWidgetItem(v))
        self.reinforcements_changed.emit()

    def _add_pile(self):
        row = self.tbl_pile.rowCount()
        self.tbl_pile.insertRow(row)
        defaults = [
            str(row + 1), "35.0", "5.0", "-10.0", "1.5", "4.0"
        ]
        for c, v in enumerate(defaults):
            self.tbl_pile.setItem(row, c, QTableWidgetItem(v))
        self.reinforcements_changed.emit()

    def _add_wall(self):
        row = self.tbl_wall.rowCount()
        self.tbl_wall.insertRow(row)
        defaults = [
            str(row + 1), "38.0", "2.0", "-5.0", "0.8", "1.5", "30.0"
        ]
        for c, v in enumerate(defaults):
            self.tbl_wall.setItem(row, c, QTableWidgetItem(v))
        self.reinforcements_changed.emit()

    def _del_row(self, table: QTableWidget):
        r = table.currentRow()
        if r < 0:
            return
        table.removeRow(r)
        for i in range(table.rowCount()):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
        self.reinforcements_changed.emit()

    # ---------------- 数据接口 ----------------
    def get_reinforcements(self):
        """返回所有支护构件对象列表"""
        from core.reinforcement import Anchor, PileRow, Wall

        if not self.chk_enable_all.isChecked():
            return []

        out = []

        # 锚索
        for r in range(self.tbl_anchor.rowCount()):
            try:
                x_head = float(self.tbl_anchor.item(r, 1).text())
                y_head = float(self.tbl_anchor.item(r, 2).text())
                angle = float(self.tbl_anchor.item(r, 3).text())
                length = float(self.tbl_anchor.item(r, 4).text())
                prestress = float(self.tbl_anchor.item(r, 5).text())
                spacing = float(self.tbl_anchor.item(r, 6).text())
                out.append(Anchor(
                    elem_id=r + 1,
                    x_head=x_head, y_head=y_head,
                    angle_deg=angle, length=length,
                    prestress_kN=prestress, spacing_m=spacing,
                ))
            except (ValueError, AttributeError):
                continue

        # 抗滑桩
        for r in range(self.tbl_pile.rowCount()):
            try:
                x = float(self.tbl_pile.item(r, 1).text())
                top = float(self.tbl_pile.item(r, 2).text())
                bottom = float(self.tbl_pile.item(r, 3).text())
                d = float(self.tbl_pile.item(r, 4).text())
                spacing = float(self.tbl_pile.item(r, 5).text())
                out.append(PileRow(
                    elem_id=r + 1,
                    x=x, top_elev=top, bottom_elev=bottom,
                    diameter=d, spacing=spacing,
                ))
            except (ValueError, AttributeError):
                continue

        # 挡土墙
        for r in range(self.tbl_wall.rowCount()):
            try:
                x = float(self.tbl_wall.item(r, 1).text())
                top = float(self.tbl_wall.item(r, 2).text())
                bottom = float(self.tbl_wall.item(r, 3).text())
                top_w = float(self.tbl_wall.item(r, 4).text())
                bot_w = float(self.tbl_wall.item(r, 5).text())
                phi_b = float(self.tbl_wall.item(r, 6).text())
                out.append(Wall(
                    elem_id=r + 1,
                    x=x, top_elev=top, bottom_elev=bottom,
                    top_width=top_w, bottom_width=bot_w,
                    passive_phi_deg=phi_b,
                ))
            except (ValueError, AttributeError):
                continue

        return out

    # ---------------- 持久化 ----------------
    def to_dict(self):
        from core.reinforcement import reinforcements_to_dict
        return {
            "enabled": self.chk_enable_all.isChecked(),
            "reinforcements": reinforcements_to_dict(self.get_reinforcements()),
        }

    def from_dict(self, data):
        if not isinstance(data, dict):
            return
        self.chk_enable_all.setChecked(bool(data.get("enabled", True)))

        self.tbl_anchor.setRowCount(0)
        self.tbl_pile.setRowCount(0)
        self.tbl_wall.setRowCount(0)

        for r in data.get("reinforcements", []) or []:
            kind = r.get("kind", "")
            if kind == "anchor":
                row = self.tbl_anchor.rowCount()
                self.tbl_anchor.insertRow(row)
                vals = [
                    str(r.get("id", row + 1)),
                    f"{r.get('x_head', 20.0):.2f}",
                    f"{r.get('y_head', 14.0):.2f}",
                    f"{r.get('angle_deg', 15.0):.2f}",
                    f"{r.get('length', 25.0):.2f}",
                    f"{r.get('prestress', 500.0):.2f}",
                    f"{r.get('spacing', 3.0):.2f}",
                ]
                for c, v in enumerate(vals):
                    self.tbl_anchor.setItem(row, c, QTableWidgetItem(v))
            elif kind == "pile":
                row = self.tbl_pile.rowCount()
                self.tbl_pile.insertRow(row)
                vals = [
                    str(r.get("id", row + 1)),
                    f"{r.get('x', 35.0):.2f}",
                    f"{r.get('top', 5.0):.2f}",
                    f"{r.get('bottom', -10.0):.2f}",
                    f"{r.get('diameter', 1.5):.2f}",
                    f"{r.get('spacing', 4.0):.2f}",
                ]
                for c, v in enumerate(vals):
                    self.tbl_pile.setItem(row, c, QTableWidgetItem(v))
            elif kind == "wall":
                row = self.tbl_wall.rowCount()
                self.tbl_wall.insertRow(row)
                vals = [
                    str(r.get("id", row + 1)),
                    f"{r.get('x', 38.0):.2f}",
                    f"{r.get('top', 2.0):.2f}",
                    f"{r.get('bottom', -5.0):.2f}",
                    f"{r.get('top_width', 0.8):.2f}",
                    f"{r.get('bottom_width', 1.5):.2f}",
                    f"{r.get('passive_phi', 30.0):.2f}",
                ]
                for c, v in enumerate(vals):
                    self.tbl_wall.setItem(row, c, QTableWidgetItem(v))

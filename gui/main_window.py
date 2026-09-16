# -*- coding: utf-8 -*-
"""
PyQt5 桌面端主窗口：集成标准菜单栏、CAD 矢量视口工具栏、可拆卸 QDockWidget 体系与工程图标
"""
from datetime import datetime
import os
import csv
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QMessageBox, QStatusBar,
    QFileDialog, QAction, QToolBar
)
from PyQt5.QtCore import Qt
import numpy as np

from core.geometry import SlopeGeometry
from core.slicing import create_slices
from core.solvers import (
    FelleniusSolver, BishopSolver, JanbuSolver,
    SpencerSolver, MorgensternPriceSolver
)
from core.dxf_io import export_model_dxf, import_dxf_polyline
from core.project_io import save_project_file, load_project_file
from core.threads import OptimizationWorker, ReliabilityWorker
from core.report_generator import generate_report
from core.reliability.monte_carlo import MonteCarloSimulator
from core.reliability.pem import run_point_estimate_analysis

from gui.canvas_qt import SlopeGraphicsView
from gui.icons import get_icon
from gui.dock_widgets import (
    GeometryDockWidget, MaterialDockWidget,
    RainfallDockWidget, LoadsDockWidget,
    SearchDockWidget, ResultsDockWidget, ReliabilityDockWidget,
    ReinforcementDockWidget,
)


class MainWindow(QMainWindow):
    """边坡极限平衡分析系统主窗口 (QDockWidget 模块化架构)"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("边坡极限平衡法稳定性分析平台 (LEM-Slope-Studio)")
        self.resize(1440, 920)

        self.current_slices = None
        self.slice_info = None
        self.current_geom = None
        self.searched_best_params = None
        self.current_project_file = None
        self._initial_fit_done = False
        self._search_worker = None
        self._last_search_diagnostics = None

        self._init_central_view()
        self._init_dock_widgets()
        self._init_menu_and_toolbars()
        self.on_params_changed()

    # ==================================================================
    # 中央视口
    # ==================================================================
    def _init_central_view(self):
        self.canvas = SlopeGraphicsView(self)
        self.setCentralWidget(self.canvas)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.canvas.drawing_status.connect(self.status_bar.showMessage)
        self.status_bar.showMessage("系统就绪，已载入标准多层边坡算例与智能寻优算法引擎。")

    # ==================================================================
    # 停靠窗口
    # ==================================================================
    def _init_dock_widgets(self):
        # ============================================================
        # 左侧：几何 / 材料 / 荷载  —— 全部为"建模输入"
        # ============================================================
        self.dock_geom = GeometryDockWidget(self)
        self.dock_geom.geometry_changed.connect(self.on_params_changed)

        self.dock_mat = MaterialDockWidget(self)
        self.dock_mat.materials_changed.connect(self.on_params_changed)
        self.dock_geom.set_region_material_names(
            [m.name for m in self.dock_mat.get_materials_list()]
        )
        self.dock_mat.materials_changed.connect(
            lambda: self.dock_geom.set_region_material_names(
                [m.name for m in self.dock_mat.get_materials_list()]
            )
        )

        self.dock_mat.materials_changed.connect(
            lambda: self.dock_geom.clamp_material_indices(
                len(self.dock_mat.get_materials_list())
            )
        )


        self.dock_loads = LoadsDockWidget(self)
        self.dock_loads.loads_changed.connect(self.on_params_changed)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_geom)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_mat)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_loads)

        # 画布 <-> 几何面板 交互
        self.dock_geom.polygon_draw_requested.connect(self.canvas.start_polygon_drawing)
        self.canvas.polygon_completed.connect(self.dock_geom.add_region)

        self.dock_geom.strata_draw_requested.connect(self.canvas.start_polyline_drawing)
        self.canvas.polyline_completed.connect(self.dock_geom.apply_cut_line)

        self.dock_geom.strata_validation_failed.connect(self.status_bar.showMessage)

        # ★ 新增: 图层树选中 → 画布高亮对应土层面
        self.dock_geom.regions_highlight_requested.connect(
            self.canvas.set_selected_regions
        )
        self.canvas.regions_selection_changed.connect(
            self.dock_geom.set_selected_regions
        )
        self.dock_geom.view_reset_requested.connect(
            lambda: self.canvas.fit_view_to_slope(margin_ratio=0.15)
        )

        self.tabifyDockWidget(self.dock_geom, self.dock_mat)
        self.tabifyDockWidget(self.dock_mat, self.dock_loads)
        self.dock_geom.raise_()

        # ============================================================
        # 右侧：滑面寻优 / 支护 / 可靠度  —— 全部为"分析控制"
        # ============================================================
        self.dock_search = SearchDockWidget(self)
        self.dock_search.preview_circle_changed.connect(self.on_params_changed)
        self.dock_search.calculate_requested.connect(self.on_calculate_requested)
        self.dock_search.search_requested.connect(self.on_search_requested)
        self.dock_search.search_stop_requested.connect(self.on_search_stop_requested)
        self.dock_search.combo_solver.currentIndexChanged.connect(
            lambda _: self._sync_reliability_inherit_label()
        )

        self.dock_reinf = ReinforcementDockWidget(self)
        self.dock_reinf.reinforcements_changed.connect(self.on_params_changed)

        self.dock_rel = ReliabilityDockWidget(self)
        self.dock_rel.reliability_requested.connect(self.on_reliability_requested)
        self.dock_rel.reliability_stop_requested.connect(self.on_reliability_stop_requested)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_search)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_reinf)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_rel)

        self.tabifyDockWidget(self.dock_search, self.dock_reinf)
        self.tabifyDockWidget(self.dock_reinf, self.dock_rel)
        self.dock_search.raise_()

        # ============================================================
        # 底部：降雨 / 成果
        # ============================================================
        self.dock_rain = RainfallDockWidget(self)
        self.dock_rain.time_step_changed.connect(self.on_time_step_changed)
        self.dock_rain.solve_time_series_requested.connect(self.on_solve_time_series)

        self.dock_results = ResultsDockWidget(self)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_rain)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_results)
        self.tabifyDockWidget(self.dock_rain, self.dock_results)
        self.dock_rain.raise_()

        # ============================================================
        # 面板尺寸与角落策略
        # ============================================================
        self.resizeDocks(
            [self.dock_geom, self.dock_mat, self.dock_loads,
             self.dock_search, self.dock_reinf, self.dock_rel],
            [340, 340, 340, 380, 380, 380],
            Qt.Horizontal
        )
        self.resizeDocks([self.dock_rain, self.dock_results], [180, 180], Qt.Vertical)

        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        self.setDockNestingEnabled(False)
        self.setDockOptions(
            QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks
        )

        self.setMinimumSize(1280, 720)

    # ==================================================================
    # 菜单栏 & 工具栏
    # ==================================================================
    def _init_menu_and_toolbars(self):
        menu_bar = self.menuBar()

        # ---------------- 文件 ----------------
        file_menu = menu_bar.addMenu("文件(&F)")

        act_new = QAction(get_icon("file_new"), "新建工程(&N)", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self.on_menu_new)
        file_menu.addAction(act_new)

        act_open = QAction(get_icon("file_open"), "打开工程(&O)...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.on_menu_open)
        file_menu.addAction(act_open)

        act_save = QAction(get_icon("file_save"), "保存工程(&S)", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.on_menu_save)
        file_menu.addAction(act_save)

        act_save_as = QAction(get_icon("file_save"), "工程另存为(&A)...", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self.on_menu_save_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_import_dxf = QAction(get_icon("import_dxf"), "导入 CAD 地表线 (DXF)...", self)
        act_import_dxf.triggered.connect(self.on_menu_import_dxf)
        file_menu.addAction(act_import_dxf)

        act_export_dxf = QAction(get_icon("export_dxf"), "分图层导出模型至 CAD (DXF)...", self)
        act_export_dxf.triggered.connect(self.on_menu_export_dxf)
        file_menu.addAction(act_export_dxf)

        act_export_csv = QAction(get_icon("export_csv"), "导出切片数据报表 (CSV)...", self)
        act_export_csv.triggered.connect(self.on_menu_export_csv)
        file_menu.addAction(act_export_csv)

        act_export_report = QAction(get_icon("export_csv"), "导出计算书 (Markdown)...", self)
        act_export_report.triggered.connect(self.on_menu_export_report)
        file_menu.addAction(act_export_report)

        file_menu.addSeparator()

        act_exit = QAction("退出(&X)", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ---------------- 视图 ----------------
        view_menu = menu_bar.addMenu("视图与窗口(&V)")

        act_zoom_fit = QAction(get_icon("zoom_extents"), "全屏居中复位 (Zoom Extents)", self)
        act_zoom_fit.setShortcut("F5")
        act_zoom_fit.triggered.connect(lambda: self.canvas.fit_view_to_slope(margin_ratio=0.15))
        view_menu.addAction(act_zoom_fit)

        view_menu.addSeparator()
        view_menu.addAction(self.dock_geom.toggleViewAction())
        view_menu.addAction(self.dock_mat.toggleViewAction())
        view_menu.addAction(self.dock_rain.toggleViewAction())
        view_menu.addAction(self.dock_loads.toggleViewAction())
        view_menu.addAction(self.dock_search.toggleViewAction())
        view_menu.addAction(self.dock_results.toggleViewAction())

        # ---------------- 分析计算 ----------------
        calc_menu = menu_bar.addMenu("分析计算(&A)")

        act_run_all = QAction(get_icon("run_solvers"), "运行全部 5 种经典模型求解", self)
        act_run_all.setShortcut("F6")
        act_run_all.triggered.connect(self.on_calculate_requested)
        calc_menu.addAction(act_run_all)

        act_start_search = QAction(get_icon("search_surface"), "启动最危险滑面搜索", self)
        act_start_search.triggered.connect(self.on_search_requested)
        calc_menu.addAction(act_start_search)

        act_time_series = QAction(get_icon("rainfall_time"), "计算降雨全历时稳定性衰减曲线 Fs(t)", self)
        act_time_series.triggered.connect(self.on_solve_time_series)
        calc_menu.addAction(act_time_series)

        # ---------------- 帮助 ----------------
        help_menu = menu_bar.addMenu("帮助(&H)")

        act_theory = QAction("极限平衡与非饱和理论说明...", self)
        act_theory.triggered.connect(self.on_menu_theory_help)
        help_menu.addAction(act_theory)

        act_about = QAction("关于 LEM-Slope-Studio...", self)
        act_about.triggered.connect(self.on_menu_about)
        help_menu.addAction(act_about)

        # ---------------- 工具栏 ----------------
        cad_toolbar = QToolBar("CAD 视口工具栏", self)
        cad_toolbar.addAction(act_zoom_fit)
        cad_toolbar.addSeparator()
        cad_toolbar.addAction(act_run_all)
        cad_toolbar.addAction(act_start_search)
        cad_toolbar.addAction(act_time_series)
        cad_toolbar.addSeparator()
        cad_toolbar.addAction(act_import_dxf)
        cad_toolbar.addAction(act_export_dxf)
        self.addToolBar(Qt.TopToolBarArea, cad_toolbar)

    def _sync_reliability_inherit_label(self):
        text = self.dock_search.get_selected_method_name()
        short = text.split()[0] if text else "—"
        self.dock_rel.lbl_inherit_model.setText(f"继承自滑面面板: {short}")

    @staticmethod
    def _make_field_lookup(locked):
        """把冻结场包装成一个可调用的查表函数"""
        if not locked or not locked.get("fields"):
            return None

        xs = locked["xs"]
        ys = locked["ys"]
        fields = locked["fields"]

        def _lookup(key, x, y):
            field = fields.get(key)
            if field is None:
                return None
            # 网格外 → None
            if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
                return None
            # 最近邻插值 (网格 100x60 足够密)
            ix = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
            iy = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
            v = field[iy, ix]
            if not np.isfinite(v):
                return None
            return float(v)

        return _lookup

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_fit_done:
            self._initial_fit_done = True
            self.canvas.fit_view_to_slope(margin_ratio=0.15)

    # ==================================================================
    # 参数变化 → 重新切片渲染
    # ==================================================================
    def on_params_changed(self):
        """统一读取全部 DockWidget 数据并重新切片渲染"""
        ground_pts = self.dock_geom.get_ground_points()
        if len(ground_pts) < 2:
            return

        water_pts = self.dock_geom.get_water_points()
        layer_regions = self.dock_geom.get_layer_regions()
        materials = self.dock_mat.get_materials_list()
        surcharge = self.dock_loads.get_surcharge_loads()
        kh = self.dock_loads.get_seismic_kh()
        rain_depth = self.dock_rain.get_current_wetting_front_depth()

        slip_surface = self.dock_search.get_slip_surface()
        xc, yc, R, n_slices = self.dock_search.get_circle_params()

        self.current_geom = SlopeGeometry(
            ground_coords=ground_pts,
            water_coords=water_pts,
            layer_regions=layer_regions,
            surcharge_loads=surcharge,
        )

        # ★ 构造 field_lookup (来自材料面板的冻结场)
        locked = self.dock_mat.get_locked_field()
        field_lookup = self._make_field_lookup(locked)

        self.current_slices, self.slice_info, msg = create_slices(
            geom=self.current_geom,
            materials=materials,
            slip_surface=slip_surface,
            n_slices=n_slices,
            rainfall_depth=rain_depth,
            kh=kh,
            field_lookup=field_lookup,
        )

        ground_x = [p[0] for p in ground_pts]
        ground_y = [p[1] for p in ground_pts]
        self.dock_mat.set_geometry_context(ground_pts, layer_regions)

        # ★ 新增: 把吸附参数传给画布
        self.canvas.set_snap_options(
            self.dock_geom.get_snap_tolerance(),
            self.dock_geom.get_snap_mode(),
        )

        # self.canvas.set_hatch_scale(self.dock_geom.get_hatch_scale())

        self.canvas.render_model(
            ground_x=ground_x,
            ground_y=ground_y,
            xc=xc, yc=yc, R=R,
            slices=self.current_slices,
            water_pts=water_pts,
            layer_regions=layer_regions,
            rainfall_depth=rain_depth,
            surcharge_loads=surcharge,
            slip_surface=slip_surface,
            reinforcements=self.dock_reinf.get_reinforcements(),
            materials=materials,  # ★ 新增: 让土层面 tooltip 能显示材料信息
            bottom_depth=self.dock_geom.get_bottom_depth(),
        )
        self.dock_results.display_slices(self.current_slices)
        self.status_bar.showMessage(msg)

    def on_time_step_changed(self, t: float):
        self.on_params_changed()
        self.status_bar.showMessage(f"已切换至降雨历时 t = {t:.1f} h，湿润锋深度已动态更新。")

    # ==================================================================
    # 降雨时程分析
    # ==================================================================
    def on_solve_time_series(self):
        ground_pts = self.dock_geom.get_ground_points()
        water_pts = self.dock_geom.get_water_points()
        materials = self.dock_mat.get_materials_list()
        surcharge = self.dock_loads.get_surcharge_loads()
        kh = self.dock_loads.get_seismic_kh()
        xc, yc, R, n_slices = self.dock_search.get_circle_params()

        geom = SlopeGeometry(
            ground_coords=ground_pts,
            water_coords=water_pts,
            layer_regions=self.dock_geom.get_layer_regions(),
            surcharge_loads=surcharge,
        )

        rain_model = self.dock_rain.get_rainfall_model()
        max_t = rain_model.get_max_time()

        if max_t <= 0.0:
            QMessageBox.warning(self, "提示", "降雨时序总历时为0，请在降雨面板中配置时间过程。")
            return

        time_steps = np.linspace(0.0, max_t, 13)
        time_results = []

        for t in time_steps:
            depth = rain_model.get_wetting_front_depth(t)
            intensity = rain_model.get_intensity_at(t)
            slices, info, _ = create_slices(
                geom=geom,
                materials=materials,
                xc=xc, yc=yc, R=R,
                n_slices=n_slices,
                rainfall_depth=depth,
                kh=kh,
            )
            if slices and info:
                b_solver = BishopSolver(slices, xc, yc, R, geom, info[2])
                fs, _ = b_solver.solve()
                fs_val = fs if fs is not None else 0.0
            else:
                fs_val = 0.0
            time_results.append((t, intensity, depth, fs_val))

        self.dock_results.display_time_series(time_results)
        self.status_bar.showMessage(
            f"降雨全历时 (0 ~ {max_t:.1f} h) 稳定性时程分析完成，已生成衰减表。"
        )

    # ==================================================================
    # 稳定性分析 (批量 5 种模型)
    # ==================================================================
    def on_calculate_requested(self):
        self.on_params_changed()
        if not self.current_slices or not self.slice_info:
            QMessageBox.warning(self, "几何异常",
                                "当前滑面未能在土体内部切出有效滑体，请调整几何输入。")
            return

        x_edges = self.slice_info[2]
        slip_surface = self.dock_search.get_slip_surface()
        stype = slip_surface.surface_type

        if stype == "circular":
            xc_ = slip_surface.xc
            yc_ = slip_surface.yc
            R_ = slip_surface.R
        else:
            xc_, yc_ = slip_surface.get_moment_center()
            R_ = max(
                float(np.hypot(s.xm - xc_, s.y_base - yc_))
                for s in self.current_slices
            )
            R_ = max(1.0, R_)

        model_classes = [
            ("Fellenius (瑞典条分法)", FelleniusSolver),
            ("Bishop (简化毕肖普)", BishopSolver),
            ("Janbu (简化让布)", JanbuSolver),
            ("Spencer (斯宾塞)", SpencerSolver),
            ("Morgenstern-Price (M-P)", MorgensternPriceSolver),
        ]

        results = []
        for name, cls in model_classes:
            if stype == "polygonal" and not getattr(cls, "supports_non_circular", False):
                continue
            try:
                solver = cls(
                    self.current_slices,
                    xc_, yc_, R_,
                    self.current_geom,
                    x_edges,
                    slip_surface=slip_surface,
                )
                fs, note = solver.solve()
            except Exception as e:
                fs, note = None, f"计算异常: {e}"
            results.append((name, (fs, note)))

        self.dock_results.display_summary(results)

        valid = [(n, fs) for n, (fs, _) in results if fs is not None]
        if valid:
            summary = " | ".join(f"{n.split()[0]}={fs:.3f}" for n, fs in valid)
            self.status_bar.showMessage(
                f"完成 {len(valid)}/{len(results)} 个模型: {summary}"
            )
        else:
            self.status_bar.showMessage("所有模型均未收敛，请检查输入参数。")

    # ==================================================================
    # 最危险滑面搜索
    # ==================================================================
    def on_search_requested(self):
        ground_pts = self.dock_geom.get_ground_points()
        water_pts = self.dock_geom.get_water_points()
        materials = self.dock_mat.get_materials_list()
        surcharge = self.dock_loads.get_surcharge_loads()
        kh = self.dock_loads.get_seismic_kh()
        rain_depth = self.dock_rain.get_current_wetting_front_depth()

        self.current_geom = SlopeGeometry(
            ground_coords=ground_pts,
            water_coords=water_pts,
            layer_regions=self.dock_geom.get_layer_regions(),
            surcharge_loads=surcharge,
        )

        stype, algo_name, eval_name, bounds, ref_pts, auto_search = \
            self.dock_search.get_search_config()

        mode_str = "自动搜索" if auto_search else "手动试算"
        self.status_bar.showMessage(
            f"正在启动 [{stype} | {mode_str}] 最危险滑面智能搜索..."
        )
        self.dock_search.btn_search.setEnabled(False)
        self.dock_search.btn_stop.setEnabled(True)
        self.dock_search.prog_bar.setValue(0)

        self._search_worker = OptimizationWorker(
            surface_type=stype,
            algo_name=algo_name,
            eval_name=eval_name,
            geom=self.current_geom,
            materials=materials,
            bounds=bounds,
            rainfall_depth=rain_depth,
            kh=kh,
            poly_ref_points=ref_pts,
            poly_auto_search=auto_search,
            parent=self,
        )
        self._search_worker.progress_updated.connect(self._on_search_progress)
        self._search_worker.search_finished.connect(self._on_search_finished)
        self._search_worker.search_failed.connect(self._on_search_failed)
        self._search_worker.search_diagnostics.connect(self._on_search_diagnostics)
        self._search_worker.start()

    def _on_search_diagnostics(self, diag: dict):
        self._last_search_diagnostics = diag
        total = diag.get("total_rejected", 0)
        if total > 0:
            self.status_bar.showMessage(
                f"搜索诊断已记录 ({total} 次拒绝) — 见计算书「附录」"
            )

    def _on_search_finished(self, best_fs: float, best_params: list, info_msg: str):
        self.dock_search.btn_search.setEnabled(True)
        self.dock_search.btn_stop.setEnabled(False)
        self.dock_search.prog_bar.setValue(100)
        self.searched_best_params = best_params

        if len(best_params) > 0 and isinstance(best_params[0], (list, tuple)):
            self.dock_search.lbl_best.setText(f"非圆弧最危险面: min Fs = {best_fs:.4f}")
            if hasattr(self.dock_search, "set_poly_points"):
                self.dock_search.set_poly_points(best_params)
        else:
            bx = float(best_params[0])
            by = float(best_params[1])
            br = float(best_params[2])
            self.dock_search.lbl_best.setText(
                f"最危险滑弧: min Fs = {best_fs:.4f} (Xc={bx:.1f}, Yc={by:.1f}, R={br:.1f})"
            )
            self.dock_search.set_circle_params(bx, by, br)

        self.on_params_changed()
        self.status_bar.showMessage(f"最危险滑面搜索完成: Fs = {best_fs:.4f}")

    def on_search_stop_requested(self):
        if self._search_worker and self._search_worker.isRunning():
            self._search_worker.stop()
            self.status_bar.showMessage("正在终止后台寻优任务...")

    def _on_search_progress(self, current: int, total: int, current_best: float):
        pct = int(current / max(1, total) * 100)
        self.dock_search.prog_bar.setValue(min(100, pct))
        self.dock_search.lbl_best.setText(f"寻优中... 当前最小 Fs = {current_best:.4f}")

    def _on_search_failed(self, err_msg: str):
        self.dock_search.btn_search.setEnabled(True)
        self.dock_search.btn_stop.setEnabled(False)
        self.status_bar.showMessage(err_msg)
        QMessageBox.information(self, "搜索提示", err_msg)

    # ==================================================================
    # 可靠度评价
    # ==================================================================
    @staticmethod
    def _method_text_to_solver_class(method_text: str):
        from core.solvers import (
            FelleniusSolver, BishopSolver, JanbuSolver,
            SpencerSolver, MorgensternPriceSolver,
        )
        if "Spencer" in method_text:
            return SpencerSolver
        if "Morgenstern" in method_text:
            return MorgensternPriceSolver
        if "Janbu" in method_text:
            return JanbuSolver
        if "Bishop" in method_text:
            return BishopSolver
        if "Fellenius" in method_text:
            return FelleniusSolver
        return BishopSolver

    def on_reliability_requested(self):
        self.on_params_changed()
        if not self.current_slices or not self.slice_info:
            QMessageBox.warning(self, "几何异常",
                                "请先设置并确保滑面在边坡内部切出有效滑动体。")
            return

        slip_surface = self.dock_search.get_slip_surface()
        method_text = self.dock_search.get_selected_method_name()
        solver_class = self._method_text_to_solver_class(method_text)

        stype = slip_surface.surface_type
        if stype == "polygonal" and not getattr(solver_class, "supports_non_circular", False):
            from core.solvers import JanbuSolver
            solver_class = JanbuSolver
            QMessageBox.information(
                self, "模型自动切换",
                f"当前滑面为非圆弧，所选模型 {method_text} 不支持非圆弧。\n"
                f"已自动切换到 Simplified Janbu 进行可靠度评价。"
            )

        cfg = self.dock_rel.get_config()
        ground_pts = self.dock_geom.get_ground_points()
        water_pts = self.dock_geom.get_water_points()
        materials = self.dock_mat.get_materials_list()
        surcharge = self.dock_loads.get_surcharge_loads()
        kh = self.dock_loads.get_seismic_kh()
        rain_depth = self.dock_rain.get_current_wetting_front_depth()

        geom = SlopeGeometry(
            ground_coords=ground_pts,
            water_coords=water_pts,
            layer_regions=self.dock_geom.get_layer_regions(),
            surcharge_loads=surcharge,
        )

        # ---------- PEM ----------
        if cfg["method"] == "PEM":
            res = run_point_estimate_analysis(
                geom=geom,
                materials=materials,
                slip_surface=slip_surface,
                solver_class=solver_class,
                cov_c=cfg["cov_c"],
                cov_phi=cfg["cov_phi"],
                rho_c_phi=cfg["rho"],
                rainfall_depth=rain_depth,
                kh=kh,
            )
            if res.get("success", False):
                self.dock_rel.display_results(res)
                self.status_bar.showMessage(
                    f"点估计法完成: Pf = {res['pf_percent']:.2f}%, "
                    f"β = {res['beta']:.2f}"
                )
            else:
                QMessageBox.warning(self, "计算失败",
                                    res.get("error", "点估计法未成功。"))

        # ---------- MCS ----------
        else:
            sim = MonteCarloSimulator(
                geom=geom,
                materials=materials,
                slip_surface=slip_surface,
                solver_class=solver_class,
                n_samples=cfg["n_samples"],
                rainfall_depth=rain_depth,
                kh=kh,
                cov_c=cfg["cov_c"],
                dist_c=cfg["dist_c"],
                cov_phi=cfg["cov_phi"],
                dist_phi=cfg["dist_phi"],
                rho_c_phi=cfg["rho"],
                cov_gamma=cfg["cov_gamma"],
            )

            self.dock_rel.btn_run_rel.setEnabled(False)
            self.dock_rel.btn_stop_rel.setEnabled(True)
            self.dock_rel.prog_bar.setValue(0)

            stype_cn = "圆弧" if stype == "circular" else "非圆弧"
            solver_short = method_text.split()[0]
            self.status_bar.showMessage(
                f"正在执行蒙特卡洛抽样 ({cfg['n_samples']} 次, "
                f"{stype_cn}滑面, 模型: {solver_short})..."
            )

            self._reliability_worker = ReliabilityWorker(sim, parent=self)
            self._reliability_worker.progress_updated.connect(self._on_rel_progress)
            self._reliability_worker.reliability_finished.connect(self._on_rel_finished)
            self._reliability_worker.reliability_failed.connect(self._on_rel_failed)
            self._reliability_worker.start()

    def on_reliability_stop_requested(self):
        if hasattr(self, "_reliability_worker") and self._reliability_worker \
                and self._reliability_worker.isRunning():
            self._reliability_worker.stop()
            self.status_bar.showMessage("正在终止可靠度抽样计算...")

    def _on_rel_progress(self, current: int, total: int, current_pf: float):
        pct = int(current / max(1, total) * 100)
        self.dock_rel.prog_bar.setValue(min(100, pct))
        self.dock_rel.lbl_pf.setText(f"{current_pf:.2f}% (抽样中...)")

    def _on_rel_finished(self, res: dict):
        self.dock_rel.btn_run_rel.setEnabled(True)
        self.dock_rel.btn_stop_rel.setEnabled(False)
        self.dock_rel.prog_bar.setValue(100)
        self.dock_rel.display_results(res)
        self.status_bar.showMessage(
            f"蒙特卡洛评价完成: Pf = {res['pf_percent']:.2f}%, β = {res['beta']:.3f}"
        )

    def _on_rel_failed(self, err_msg: str):
        self.dock_rel.btn_run_rel.setEnabled(True)
        self.dock_rel.btn_stop_rel.setEnabled(False)
        self.status_bar.showMessage(err_msg)
        QMessageBox.information(self, "可靠度评价提示", err_msg)

    # ==================================================================
    # 导出计算书
    # ==================================================================
    def on_menu_export_report(self):
        self.on_params_changed()
        if not self.current_slices or not self.slice_info:
            QMessageBox.warning(self, "无法导出", "当前无有效切片数据，请先设置滑面并刷新。")
            return

        try:
            self.on_calculate_requested()
        except Exception:
            pass

        method_text = self.dock_search.get_selected_method_name()
        method_short = method_text.split()[0]

        fs_current = None
        note_current = "—"
        try:
            row = self.dock_results.tbl_summary.rowCount()
            if row > 0:
                item = self.dock_results.tbl_summary.item(0, 1)
                if item and item.text() != "未收敛":
                    fs_current = float(item.text())
                item_note = self.dock_results.tbl_summary.item(0, 2)
                if item_note:
                    note_current = item_note.text()
        except Exception:
            pass

        solver_results = [(method_short, fs_current, note_current)]

        default_name = "计算书_" + datetime.now().strftime("%Y%m%d_%H%M") + ".md"
        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出计算书 (Markdown)", default_name,
            "Markdown 文件 (*.md);;所有文件 (*.*)"
        )
        if not fpath:
            return
        if not fpath.lower().endswith(".md"):
            fpath += ".md"

        ground_pts = self.dock_geom.get_ground_points()
        water_pts = self.dock_geom.get_water_points()
        layer_regions = self.dock_geom.get_layer_regions()
        materials = self.dock_mat.get_materials_list()
        surcharge = self.dock_loads.get_surcharge_loads()
        kh = self.dock_loads.get_seismic_kh()
        rain_depth = self.dock_rain.get_current_wetting_front_depth()
        slip_surface = self.dock_search.get_slip_surface()
        _, _, _, n_slices = self.dock_search.get_circle_params()

        try:
            md_text = generate_report(
                project_name=self.windowTitle().split("(")[0].strip() or "未命名工程",
                ground_pts=ground_pts,
                water_pts=water_pts,
                layer_regions=layer_regions,  # ★ 新参数
                materials=materials,
                surcharge_loads=surcharge,
                kh=kh,
                rainfall_depth=rain_depth,
                slip_surface=slip_surface,
                n_slices=n_slices,
                slices=self.current_slices,
                solver_results=solver_results,
                search_info=None,
                search_diagnostics=self._last_search_diagnostics,
            )

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(md_text)

            self.status_bar.showMessage(f"计算书已导出: {os.path.basename(fpath)}")
            QMessageBox.information(
                self, "导出成功",
                f"计算书 (Markdown) 已成功导出：\n{fpath}\n\n"
                f"可以使用 Typora、VS Code、Obsidian 等工具打开，\n"
                f"或导出为 PDF。"
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"生成计算书时发生异常:\n{e}")

    # ==================================================================
    # 新建 / 打开 / 保存
    # ==================================================================
    def on_menu_new(self):
        self._last_search_diagnostics = None
        default_ground = [(0.0, 15.0), (20.0, 15.0), (35.0, 0.0), (60.0, 0.0)]
        default_water = [(0.0, 12.0), (20.0, 12.0), (35.0, -1.0), (60.0, -1.0)]

        self.dock_geom.from_dict({
            "ground": default_ground,
            "water": default_water,
            "regions": [],  # ★ 空 → from_dict 自动生成默认大面
        })

        self.dock_mat.from_dict({
            "active_regime": 0,
            "current_layer": 0,
            "layers": [
                {"name": "第1层-上覆土", "gamma_dry": 19.0, "gamma_sat": 21.0,
                 "c_prime": 15.0, "phi_deg": 20.0, "is_unsaturated": False,
                 "phi_b_deg": 15.0, "suction_cutoff": 100.0},
                {"name": "第2层-下卧土", "gamma_dry": 22.0, "gamma_sat": 23.5,
                 "c_prime": 40.0, "phi_deg": 30.0, "is_unsaturated": False,
                 "phi_b_deg": 22.0, "suction_cutoff": 150.0},
                {"name": "第3层-基岩层", "gamma_dry": 25.0, "gamma_sat": 26.0,
                 "c_prime": 80.0, "phi_deg": 38.0, "is_unsaturated": False,
                 "phi_b_deg": 28.0, "suction_cutoff": 200.0},
            ],
        })

        self.dock_search.from_dict({
            "surface_type": "circular",
            "circle": {"xc": 25.0, "yc": 22.0, "R": 23.0, "n_slices": 30},
            "polygon": [(12.0, 15.0), (25.0, 4.0), (38.0, 0.0)],
        })

        self.dock_loads.from_dict({"q": 0.0, "x1": 5.0, "x2": 18.0, "kh": 0.0})
        self.dock_rain.from_dict({
            "series": [(0.0, 10.0), (12.0, 35.0), (24.0, 60.0), (48.0, 5.0)],
            "ks_mm_h": 15.0, "delta_theta": 0.20, "current_time": 0,
        })

        self.current_project_file = None
        self.searched_best_params = None

        self.on_params_changed()
        self.canvas.fit_view_to_slope(margin_ratio=0.15)
        self.status_bar.showMessage("已新建工程并恢复默认参数。")

    def on_menu_open(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "打开工程文件", "",
            "LEM 工程文件 (*.lem *.json);;所有文件 (*.*)"
        )
        if not fpath:
            return

        data = load_project_file(fpath)
        if not data:
            QMessageBox.critical(self, "打开失败", "工程文件损坏或格式不正确。")
            return

        try:
            if "geometry" in data:
                self.dock_geom.from_dict(data["geometry"])
            if "materials" in data:
                self.dock_mat.from_dict(data["materials"])
            if "environment" in data:
                env = data["environment"]
                self.dock_loads.from_dict(env)
                if "rainfall" in env:
                    self.dock_rain.from_dict(env["rainfall"])
            if "slip_surface" in data:
                self.dock_search.from_dict(data["slip_surface"])
            if "reliability" in data:
                self.dock_rel.from_dict(data["reliability"])

            self.current_project_file = fpath

            self.on_params_changed()
            self.canvas.fit_view_to_slope(margin_ratio=0.15)
            self.status_bar.showMessage(f"已成功载入工程: {os.path.basename(fpath)}")

        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"载入过程发生异常:\n{e}")

    def on_menu_save(self):
        if self.current_project_file:
            self._save_to_path(self.current_project_file)
        else:
            self.on_menu_save_as()

    def on_menu_save_as(self):
        fpath, _ = QFileDialog.getSaveFileName(
            self, "工程另存为", "slope_case.lem",
            "LEM 工程文件 (*.lem *.json);;所有文件 (*.*)"
        )
        if not fpath:
            return
        self._save_to_path(fpath)

    def _save_to_path(self, fpath: str):
        data = {
            "geometry": self.dock_geom.to_dict(),
            "materials": self.dock_mat.to_dict(),
            "environment": {
                **self.dock_loads.to_dict(),
                "rainfall": self.dock_rain.to_dict(),
            },
            "slip_surface": self.dock_search.to_dict(),
            "reliability": self.dock_rel.to_dict(),
        }
        if save_project_file(fpath, data):
            self.current_project_file = fpath
            self.status_bar.showMessage(f"工程已保存: {os.path.basename(fpath)}")
        else:
            QMessageBox.critical(self, "保存失败", "写入文件发生错误。")

    # ==================================================================
    # DXF / CSV 导入导出
    # ==================================================================
    def on_menu_import_dxf(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "导入 CAD DXF 地表线", "",
            "AutoCAD DXF 文件 (*.dxf);;所有文件 (*.*)"
        )
        if not fpath:
            return
        pts = import_dxf_polyline(fpath)
        if len(pts) >= 2:
            self.dock_geom.data_ground = pts
            self.dock_geom.tree.setCurrentItem(self.dock_geom.item_ground)
            self.dock_geom._load_table_data(pts)
            self.on_params_changed()
            self.canvas.fit_view_to_slope()
            self.status_bar.showMessage(f"成功从 DXF 导入 {len(pts)} 个地表顶点。")
            QMessageBox.information(self, "导入成功",
                                    f"已从 DXF 提取并载入 {len(pts)} 个连续地表坐标点。")
        else:
            QMessageBox.warning(self, "导入失败",
                                "未在 DXF 文件中找到有效的连续折线或线段实体。")

    def on_menu_export_dxf(self):
        fpath, _ = QFileDialog.getSaveFileName(
            self, "分图层导出模型至 CAD DXF", "slope_geometry.dxf",
            "AutoCAD DXF 文件 (*.dxf);;所有文件 (*.*)"
        )
        if not fpath:
            return

        ground_pts = self.dock_geom.get_ground_points()
        water_pts = self.dock_geom.get_water_points()
        layer_regions = self.dock_geom.get_layer_regions()

        # 导出地层分界线: 用每个面的边界折线代替 (暂只导出外边界)
        strata_out = []
        for region in layer_regions:
            pts = region.get("points", [])
            if len(pts) >= 2:
                strata_out.append(pts)

        ok = export_model_dxf(
            filepath=fpath,
            ground_pts=ground_pts,
            strata_lines=strata_out,  # 兼容旧签名
            water_pts=water_pts,
        )
        if ok:
            self.status_bar.showMessage(
                f"模型已成功分图层导出为 DXF: {os.path.basename(fpath)}"
            )
            QMessageBox.information(
                self, "导出成功",
                f"分图层 CAD DXF 导出完成：\n{fpath}\n"
                f"包含 0_GROUND_SURFACE、1_STRATA_LAYER 与 2_PHREATIC_WATER 图层。"
            )
        else:
            QMessageBox.critical(self, "导出失败", "DXF 文件写出发生错误。")

    def on_menu_export_csv(self):
        if not self.current_slices:
            QMessageBox.warning(self, "提示", "当前无有效切片数据可导出。")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出切片数据报表", "slope_slices_report.csv",
            "CSV 逗号分隔文件 (*.csv);;所有文件 (*.*)"
        )
        if not fpath:
            return

        try:
            with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "土条号", "所属地层", "中点X(m)", "条宽b(m)", "高度h(m)",
                    "土自重(kN)", "附加外载(kN)", "总竖力W(kN)", "地震力Fh(kN)",
                    "孔压u(kPa)", "基质吸力(kPa)", "总黏聚力(kPa)",
                    "摩擦角(度)", "底坡角(度)", "底斜长(m)"
                ])
                for s in self.current_slices:
                    writer.writerow([
                        s.index, s.layer_name,
                        round(s.xm, 3), round(s.b, 3), round(s.h, 3),
                        round(s.W_soil, 3), round(s.q_load, 3), round(s.W, 3),
                        round(s.Fh, 3),
                        round(s.u, 3), round(s.suction, 3), round(s.c, 3),
                        round(float(np.degrees(s.phi)), 3),
                        round(float(np.degrees(s.alpha)), 3),
                        round(s.l, 3),
                    ])
            self.status_bar.showMessage(f"切片数据报表已导出: {os.path.basename(fpath)}")
            QMessageBox.information(
                self, "导出成功",
                f"已成功将 {len(self.current_slices)} 个土条的完整数据导出至 CSV。"
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写入 CSV 报表时出错: {str(e)}")

    # ==================================================================
    # 帮助
    # ==================================================================
    def on_menu_theory_help(self):
        text = (
            "<h3>极限平衡法 (LEM) 理论模型体系说明</h3>"
            "<p>本软件实现了土力学经典的 5 大极限平衡计算模型，全面支持任意多层起伏地层、"
            "地下水浸润线、动态降雨时变入渗、非饱和吸力本构及地震拟静力荷载：</p>"
            "<ul>"
            "<li><b>Fellenius (瑞典条分法)</b>：力矩平衡显式解，适用于均质土初筛。</li>"
            "<li><b>Simplified Bishop (简化毕肖普法)</b>：考虑水平条间力，满足竖向力和力矩平衡。</li>"
            "<li><b>Simplified Janbu (简化让布法)</b>：满足条块力平衡，结合 d/L 引入 f0 经验修正。</li>"
            "<li><b>Spencer (斯宾塞法)</b>：完全平衡严密法，联立求解 (Fs, θ)。</li>"
            "<li><b>Morgenstern-Price (M-P 法 / GLE)</b>：广义严密极限平衡法，条间剪力假定为 "
            "X = λ·f(x)·E（半正弦波），求解 (Fs, λ)。</li>"
            "</ul>"
        )
        QMessageBox.about(self, "理论说明", text)

    def on_menu_about(self):
        text = (
            "<h3>边坡极限平衡法稳定性分析平台 (LEM-Slope-Studio)</h3>"
            "<p>基于 Python 3 与 PyQt5 架构开发，采用专业 QDockWidget 模块化界面布局。</p>"
            "<p>内置经典 LEM 模型、全局优化寻优算法、动态降雨历时入渗分析及分图层 AutoCAD DXF 交换。</p>"
        )
        QMessageBox.about(self, "关于系统", text)

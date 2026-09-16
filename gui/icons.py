# -*- coding: utf-8 -*-
"""
工程矢量图标管理模块 (PyQt5)
优先加载 assets/icons/ 目录下的矢量 SVG 图标；
若运行环境缺少 QtSvg 格式插件，自动切换为 QPainter 纯矢量程序绘制回退，确保图标在任何平台均可稳定显示。
"""
import os
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QPolygon
from PyQt5.QtCore import Qt, QPoint, QRect
import numpy as np

# 图标静态资源根目录
ICONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "icons"))


def get_icon(name: str) -> QIcon:
    """获取指定名称的矢量图标"""
    svg_path = os.path.join(ICONS_DIR, f"{name}.svg")
    if os.path.isfile(svg_path):
        icon = QIcon(svg_path)
        if not icon.isNull():
            return icon

    # 回退方案：使用 QPainter 在内存 QPixmap 上进行精确矢量绘制
    return _create_procedural_icon(name)


def _create_procedural_icon(name: str) -> QIcon:
    """纯代码矢量绘制备用图标 (24x24 像素)"""
    size = 24
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)

    if name == "zoom_extents":
        pen = QPen(QColor(44, 62, 80), 2)
        pen.setCosmetic(True)
        p.setPen(pen)
        d, s = 3, 5
        p.drawLine(d, d + s, d, d); p.drawLine(d, d, d + s, d)
        p.drawLine(size - d - s, d, size - d, d); p.drawLine(size - d, d, size - d, d + s)
        p.drawLine(d, size - d - s, d, size - d); p.drawLine(d, size - d, d + s, size - d)
        p.drawLine(size - d - s, size - d, size - d, size - d); p.drawLine(size - d, size - d, size - d, size - d - s)
        p.setBrush(QBrush(QColor(52, 152, 219)))
        p.drawEllipse(QPoint(12, 12), 2, 2)

    elif name == "run_solvers":
        p.setBrush(QBrush(QColor(39, 174, 96)))
        p.setPen(QPen(QColor(34, 153, 84), 1))
        poly = QPolygon([QPoint(6, 4), QPoint(20, 12), QPoint(6, 20)])
        p.drawPolygon(poly)

    elif name == "search_surface":
        p.setPen(QPen(QColor(211, 84, 0), 2))
        p.drawEllipse(QPoint(10, 10), 6, 6)
        p.drawLine(15, 15, 20, 20)
        p.setBrush(QBrush(QColor(230, 126, 34)))
        p.drawEllipse(QPoint(10, 10), 2, 2)

    elif name == "stop_search":
        p.setBrush(QBrush(QColor(192, 57, 43)))
        p.setPen(QPen(QColor(150, 45, 34), 1))
        p.drawRoundedRect(QRect(5, 5, 14, 14), 2, 2)

    elif name == "rainfall_time":
        pen = QPen(QColor(41, 128, 185), 2)
        p.setPen(pen)
        p.drawArc(QRect(4, 5, 16, 9), 0, 180 * 16)
        p.drawLine(7, 16, 5, 20)
        p.drawLine(12, 16, 10, 20)
        p.drawLine(17, 16, 15, 20)

    elif name == "file_new":
        p.setPen(QPen(QColor(44, 62, 80), 1.5))
        p.drawRect(5, 3, 14, 18)
        p.drawLine(12, 8, 12, 16)
        p.drawLine(8, 12, 16, 12)

    elif name == "file_open":
        p.setBrush(QBrush(QColor(241, 196, 15)))
        p.setPen(QPen(QColor(214, 137, 16), 1.5))
        poly = QPolygon([QPoint(3, 8), QPoint(9, 8), QPoint(11, 10), QPoint(21, 10), QPoint(21, 20), QPoint(3, 20)])
        p.drawPolygon(poly)

    elif name == "file_save":
        p.setBrush(QBrush(QColor(52, 152, 219)))
        p.setPen(QPen(QColor(41, 128, 185), 1.5))
        p.drawRect(4, 4, 16, 16)
        p.setBrush(QBrush(QColor(236, 240, 241)))
        p.drawRect(7, 4, 10, 6)

    elif name == "import_dxf":
        p.setPen(QPen(QColor(41, 128, 185), 2))
        p.drawLine(12, 4, 12, 15)
        p.drawLine(8, 11, 12, 15); p.drawLine(16, 11, 12, 15)
        p.drawLine(4, 19, 20, 19)

    elif name == "export_dxf":
        p.setPen(QPen(QColor(39, 174, 96), 2))
        p.drawLine(12, 15, 12, 4)
        p.drawLine(8, 8, 12, 4); p.drawLine(16, 8, 12, 4)
        p.drawLine(4, 19, 20, 19)

    elif name == "export_csv":
        p.setPen(QPen(QColor(44, 62, 80), 1.5))
        p.drawRect(4, 4, 16, 16)
        p.drawLine(4, 10, 20, 10); p.drawLine(4, 15, 20, 15)
        p.drawLine(10, 4, 10, 20); p.drawLine(15, 4, 15, 20)

    elif name == "add_item":
        p.setPen(QPen(QColor(39, 174, 96), 2))
        p.drawEllipse(QPoint(12, 12), 9, 9)
        p.drawLine(12, 7, 12, 17); p.drawLine(7, 12, 17, 12)

    elif name == "del_item":
        p.setPen(QPen(QColor(192, 57, 43), 2))
        p.drawEllipse(QPoint(12, 12), 9, 9)
        p.drawLine(7, 12, 17, 12)

    elif name == "refresh":
        p.setPen(QPen(QColor(41, 128, 185), 2))
        p.drawArc(QRect(4, 4, 16, 16), 45 * 16, 270 * 16)
        p.drawLine(16, 4, 20, 8); p.drawLine(20, 4, 20, 8)

    elif name == "apply_surface":
        p.setPen(QPen(QColor(39, 174, 96), 2.5))
        p.drawLine(4, 12, 9, 17); p.drawLine(9, 17, 20, 6)

    elif name == "merge_layers":
        # 左边土层面
        p.setBrush(QBrush(QColor(243, 223, 170)))
        p.setPen(QPen(QColor(44, 62, 80), 1.5))
        poly1 = QPolygon([QPoint(2, 7), QPoint(9, 5), QPoint(9, 19), QPoint(2, 17)])
        p.drawPolygon(poly1)
        # 右边土层面
        p.setBrush(QBrush(QColor(234, 212, 154)))
        p.drawPolygon(QPolygon([QPoint(15, 5), QPoint(22, 7), QPoint(22, 17), QPoint(15, 19)]))
        # 中间双向连接箭头
        p.setPen(QPen(QColor(41, 128, 185), 1.8))
        p.drawLine(9, 12, 15, 12)
        p.drawLine(9, 12, 10, 10)
        p.drawLine(9, 12, 10, 14)
        p.drawLine(15, 12, 14, 10)
        p.drawLine(15, 12, 14, 14)

    elif name == "material_add":
        p.setPen(QPen(QColor(39, 174, 96), 2))
        p.drawEllipse(QPoint(12, 12), 8, 8)
        p.drawLine(12, 8, 12, 16)
        p.drawLine(8, 12, 16, 12)
        # 小方块底座, 表示"添加到库"
        p.setBrush(QBrush(QColor(243, 223, 170)))
        p.setPen(QPen(QColor(44, 62, 80), 1))
        p.drawRect(3, 18, 6, 3)

    elif name == "material_copy":
        # 两张叠放的纸
        p.setBrush(QBrush(QColor(236, 240, 241)))
        p.setPen(QPen(QColor(44, 62, 80), 1.5))
        p.drawRect(4, 3, 12, 15)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.drawRect(8, 7, 12, 15)
        # 复制标记线
        p.setPen(QPen(QColor(52, 152, 219), 1.5))
        p.drawLine(11, 11, 17, 11)
        p.drawLine(11, 14, 17, 14)
        p.drawLine(11, 17, 15, 17)

    elif name == "material_del":
        # 垃圾桶
        p.setPen(QPen(QColor(192, 57, 43), 2))
        p.setBrush(QBrush(QColor(231, 76, 60)))
        # 桶身
        p.drawRect(6, 8, 12, 12)
        # 桶盖
        p.setBrush(QBrush(QColor(192, 57, 43)))
        p.drawRect(5, 5, 14, 3)
        # 提手
        p.setPen(QPen(QColor(150, 45, 34), 1.5))
        p.drawLine(9, 5, 11, 3)
        p.drawLine(11, 3, 13, 3)
        p.drawLine(13, 3, 15, 5)
        # 竖纹
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.2))
        p.drawLine(9, 10, 9, 18)
        p.drawLine(12, 10, 12, 18)
        p.drawLine(15, 10, 15, 18)

    elif name == "dist_preview":
        # 钟形曲线 (高斯 pdf)
        p.setPen(QPen(QColor(41, 128, 185), 2))
        path_points = []
        for xi in range(24):
            x = xi
            # 高斯钟形: 中心 12, 宽度 4
            y_pix = 20 - 14 * np.exp(-0.5 * ((xi - 12) / 3.5) ** 2)
            path_points.append(QPoint(x, int(y_pix)))
        for i in range(len(path_points) - 1):
            p.drawLine(path_points[i], path_points[i + 1])
        # 均值线
        p.setPen(QPen(QColor(192, 57, 43), 1.2, Qt.DashLine))
        p.drawLine(12, 4, 12, 21)

    else:
        p.setPen(QPen(QColor(127, 140, 141), 1))
        p.drawRect(4, 4, 16, 16)

    p.end()
    return QIcon(pix)
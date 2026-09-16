# -*- coding: utf-8 -*-
"""
边坡极限平衡法稳定性分析平台 (LEM-Slope-Studio)
应用程序入口
"""
import sys
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.icons import get_icon


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(get_icon("app_icon"))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
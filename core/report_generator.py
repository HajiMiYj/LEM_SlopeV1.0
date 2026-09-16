# -*- coding: utf-8 -*-
"""
计算书生成器 (Markdown 格式)

输出内容:
  1. 封面信息 (工程名、日期、软件版本)
  2. 工程概况 (地表几何、地层、材料)
  3. 荷载工况 (坡顶超载、地下水位、地震、降雨)
  4. 滑面定义 (圆弧 or 非圆弧)
  5. 稳定性分析结果 (各 LEM 方法 Fs)
  6. 土条微元明细表
  7. 结论与判定
"""
import datetime
from typing import List, Dict, Any, Optional


APP_NAME = "LEM-Slope-Studio"
APP_VERSION = "1.0"


def generate_report(
    project_name: str,
    ground_pts: List,
    water_pts: Optional[List],
    strata_lines: List,
    materials: List,
    surcharge_loads: List,
    kh: float,
    rainfall_depth: float,
    slip_surface,
    n_slices: int,
    slices: Optional[List],
    solver_results: List,
    search_info: Optional[Dict[str, Any]] = None,
    search_diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    """生成 Markdown 格式计算书"""

    lines = []

    # ========== 封面 ==========
    lines.append(f"# 边坡稳定性计算书")
    lines.append("")
    lines.append(f"**工程名称:** {project_name}")
    lines.append("")
    lines.append(f"**计算日期:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"**计算软件:** {APP_NAME} v{APP_VERSION}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 1. 工程概况 ==========
    lines.append("## 1. 工程概况")
    lines.append("")
    lines.append("### 1.1 地表轮廓线")
    lines.append("")
    lines.append("| 序号 | X (m) | Y (m) |")
    lines.append("| :--: | :---: | :---: |")
    for i, (x, y) in enumerate(ground_pts, 1):
        lines.append(f"| {i} | {float(x):.3f} | {float(y):.3f} |")
    lines.append("")

    if water_pts:
        lines.append("### 1.2 地下水位线")
        lines.append("")
        lines.append("| 序号 | X (m) | 水位 Y (m) |")
        lines.append("| :--: | :---: | :---: |")
        for i, (x, y) in enumerate(water_pts, 1):
            lines.append(f"| {i} | {float(x):.3f} | {float(y):.3f} |")
        lines.append("")
    else:
        lines.append("### 1.2 地下水位线")
        lines.append("")
        lines.append("> 未设置地下水位线 (按无水工况计算)。")
        lines.append("")

    if strata_lines:
        lines.append("### 1.3 地层分界线")
        lines.append("")
        for idx, line in enumerate(strata_lines, 1):
            lines.append(f"**第 {idx} 分界面:**")
            lines.append("")
            lines.append("| 序号 | X (m) | Y (m) |")
            lines.append("| :--: | :---: | :---: |")
            for i, (x, y) in enumerate(line, 1):
                lines.append(f"| {i} | {float(x):.3f} | {float(y):.3f} |")
            lines.append("")

    # ========== 2. 土体材料参数 ==========
    lines.append("## 2. 土体材料参数")
    lines.append("")
    lines.append("| 土层名称 | 天然重度 γ | 饱和重度 γsat | 有效黏聚力 c' | 有效内摩擦角 φ' | 非饱和 | φb | 吸力截断 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in materials:
        unsat = "是" if getattr(m, "is_unsaturated", False) else "否"
        phib = getattr(m, "phi_b_deg", 0.0)
        cutoff = getattr(m, "suction_cutoff", 0.0)
        lines.append(
            f"| {m.name} "
            f"| {m.gamma_dry:.2f} kN/m³ "
            f"| {m.gamma_sat:.2f} kN/m³ "
            f"| {m.c_prime:.2f} kPa "
            f"| {m.phi_deg:.2f}° "
            f"| {unsat} "
            f"| {phib:.2f}° "
            f"| {cutoff:.1f} kPa |"
        )
    lines.append("")

    # ========== 3. 荷载工况 ==========
    lines.append("## 3. 荷载工况")
    lines.append("")
    if surcharge_loads:
        lines.append("### 3.1 坡顶超载")
        lines.append("")
        lines.append("| 序号 | 起始 X (m) | 终止 X (m) | 荷载强度 q (kPa) |")
        lines.append("| :--: | :---: | :---: | :---: |")
        for i, (x1, x2, q) in enumerate(surcharge_loads, 1):
            lines.append(f"| {i} | {float(x1):.2f} | {float(x2):.2f} | {float(q):.2f} |")
        lines.append("")
    else:
        lines.append("### 3.1 坡顶超载")
        lines.append("")
        lines.append("> 无坡顶超载。")
        lines.append("")

    lines.append("### 3.2 地震拟静力工况")
    lines.append("")
    lines.append(f"- 水平地震加速度系数 kh = **{kh:.3f}**")
    lines.append(f"- 惯性力 Fh = kh · W (作用方向水平指向坡外)")
    lines.append("")

    lines.append("### 3.3 降雨入渗工况")
    lines.append("")
    if rainfall_depth > 0:
        lines.append(f"- 湿润锋深度 = **{rainfall_depth:.2f} m**")
        lines.append(f"- 湿润锋以上土体按饱和重度 γsat 计算")
        lines.append(f"- 湿润锋以下土体保持天然状态，基质吸力保留")
    else:
        lines.append("> 无降雨入渗工况 (湿润锋深度 = 0)。")
    lines.append("")

    # ========== 4. 滑面定义 ==========
    lines.append("## 4. 滑动面定义")
    lines.append("")
    stype = getattr(slip_surface, "surface_type", "circular")
    if stype == "circular":
        lines.append("### 4.1 滑面类型：圆弧滑动面")
        lines.append("")
        lines.append(f"- 圆心 X₀ = **{slip_surface.xc:.3f} m**")
        lines.append(f"- 圆心 Y₀ = **{slip_surface.yc:.3f} m**")
        lines.append(f"- 半径 R = **{slip_surface.R:.3f} m**")
        lines.append("")
    else:
        lines.append("### 4.1 滑面类型：非圆弧多段折线滑动面")
        lines.append("")
        lines.append("| 折点号 | X (m) | Y (m) |")
        lines.append("| :--: | :---: | :---: |")
        for i, (x, y) in enumerate(slip_surface.points, 1):
            lines.append(f"| {i} | {float(x):.3f} | {float(y):.3f} |")
        lines.append("")

    lines.append(f"### 4.2 切片离散信息")
    lines.append("")
    lines.append(f"- 土条数量：**{n_slices}**")
    lines.append("")

    # ========== 5. 计算结果 ==========
    lines.append("## 5. 稳定性分析结果")
    lines.append("")
    lines.append("| 计算方法 | 稳定安全系数 Fs | 收敛状态 |")
    lines.append("| :--- | :---: | :--- |")
    for method_name, fs, note in solver_results:
        fs_str = f"{fs:.4f}" if fs is not None else "未收敛"
        lines.append(f"| {method_name} | **{fs_str}** | {note} |")
    lines.append("")

    if search_info:
        lines.append("### 5.1 最危险滑面搜索")
        lines.append("")
        lines.append(f"- 搜索算法：{search_info.get('algo', '—')}")
        lines.append(f"- 评价模型：{search_info.get('eval', '—')}")
        lines.append(f"- 最小安全系数：**{search_info.get('min_fs', '—')}**")
        lines.append(f"- 收敛状态：{search_info.get('status', '—')}")
        lines.append("")

    # ========== 6. 土条明细 ==========
    if slices:
        lines.append("## 6. 土条微元明细表")
        lines.append("")
        lines.append("| # | 土层 | 中点 X (m) | 宽度 b (m) | 高度 h (m) "
                     "| 土自重 (kN) | 超载 (kN) | 总竖力 W (kN) "
                     "| 地震力 Fh (kN) | 孔压 u (kPa) | 吸力 (kPa) "
                     "| 黏聚力 (kPa) | 摩擦角 (°) | 底坡角 α (°) | 底斜长 l (m) |")
        lines.append("| :--: | :--- | :---: | :---: | :---: | :---: | :---: | :---: "
                     "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        import math
        for s in slices:
            lines.append(
                f"| {s.index} | {s.layer_name} "
                f"| {s.xm:.3f} | {s.b:.3f} | {s.h:.3f} "
                f"| {s.W_soil:.2f} | {s.q_load:.2f} | {s.W:.2f} "
                f"| {s.Fh:.2f} | {s.u:.2f} | {s.suction:.2f} "
                f"| {s.c:.2f} | {math.degrees(s.phi):.2f} "
                f"| {math.degrees(s.alpha):.2f} | {s.l:.3f} |"
            )
        lines.append("")

    # ========== 7. 结论 ==========
    lines.append("## 7. 结论")
    lines.append("")

    # 取最保守的 FS (最大值代表最不利？不，最小值才是不利)
    valid_fs = [fs for _, fs, _ in solver_results if fs is not None]
    if valid_fs:
        min_fs = min(valid_fs)
        max_fs = max(valid_fs)
        lines.append(f"- 各方法计算最小 Fs = **{min_fs:.4f}**，最大 Fs = **{max_fs:.4f}**")
        if min_fs >= 1.35:
            verdict = "**稳定** （最小 Fs ≥ 1.35，满足《建筑边坡工程技术规范》GB50330 一级边坡要求）"
        elif min_fs >= 1.25:
            verdict = "**基本稳定** （最小 Fs ≥ 1.25，满足二级边坡要求）"
        elif min_fs >= 1.15:
            verdict = "**欠稳定** （最小 Fs ≥ 1.15，满足三级边坡要求）"
        elif min_fs >= 1.05:
            verdict = "**临界状态** （最小 Fs ≥ 1.05，需加强支护）"
        else:
            verdict = "**不稳定** （最小 Fs < 1.05，边坡处于失稳状态，需立即采取工程措施）"
        lines.append(f"- 综合判定：{verdict}")
    else:
        lines.append("> 所有计算方法均未收敛，无法给出结论。请检查输入参数。")
    lines.append("")
    
    
        # ========== 附录 A: 搜索诊断 ==========
    if search_diagnostics:
        lines.append("---")
        lines.append("")
        lines.append("## 附录 A：最危险滑面搜索诊断")
        lines.append("")

        mode = search_diagnostics.get("mode", "unknown")

        if mode == "auto_search":
            lines.append("### A.1 搜索参数")
            lines.append("")
            lines.append(f"- 参考控制点数：**{search_diagnostics.get('n_ref', '—')}**"
                         f"（中间点 {search_diagnostics.get('n_mid', '—')} 个）")
            lines.append(f"- 搜索维度：{search_diagnostics.get('dim', '—')}")
            lines.append(f"- 种群规模：{search_diagnostics.get('popsize', '—')}")
            lines.append(f"- 最大迭代：{search_diagnostics.get('maxiter', '—')}")
            lines.append("")

            lines.append("### A.2 收敛状态")
            lines.append("")
            success = search_diagnostics.get("de_success")
            lines.append(f"- 收敛：{'✓ 成功' if success else '✗ 未完全收敛'}")
            lines.append(f"- 消息：`{search_diagnostics.get('de_message', '—')}`")
            lines.append("")

            b = search_diagnostics.get("bounds_final", {})
            if b:
                xi = b.get("x_in", [0, 0])
                xo = b.get("x_out", [0, 0])
                yb = b.get("y_bot", [0, 0])
                lines.append("### A.3 生效搜索范围")
                lines.append("")
                lines.append(f"- 后缘 X ∈ [{xi[0]:.2f}, {xi[1]:.2f}] m")
                lines.append(f"- 坡脚 X ∈ [{xo[0]:.2f}, {xo[1]:.2f}] m")
                lines.append(f"- 滑底 Y ∈ [{yb[0]:.2f}, {yb[1]:.2f}] m")
                lines.append("")

            stats = search_diagnostics.get("reject_stats", {})
            total = search_diagnostics.get("total_rejected", 0)
            if stats:
                lines.append("### A.4 候选滑面拒绝统计")
                lines.append("")
                lines.append(f"DE 搜索期间累计拒绝 **{total}** 次候选解，分类如下：")
                lines.append("")
                lines.append("| 拒绝原因 | 次数 | 占比 |")
                lines.append("| :--- | :---: | :---: |")
                for k, v in sorted(stats.items(), key=lambda x: -x[1]):
                    pct = v / max(1, total) * 100
                    lines.append(f"| {k} | {v} | {pct:.1f}% |")
                lines.append("")

        elif mode == "manual":
            lines.append("### A.1 手动试算")
            lines.append("")
            lines.append(f"- 控制点数：{search_diagnostics.get('n_points', '—')}")
            lines.append(f"- 中间点数：{search_diagnostics.get('n_mid', '—')}")
            lines.append("")
            lines.append("> 手动试算模式未启动全局搜索。")
            lines.append("")

        elif mode == "circular":
            lines.append("### A.1 圆弧搜索")
            lines.append("")
            lines.append(f"- 搜索算法：{search_diagnostics.get('algo', '—')}")
            lines.append(f"- 评价模型：{search_diagnostics.get('eval', '—')}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*本计算书由 {APP_NAME} 自动生成，计算结果仅供参考，实际工程设计须经注册岩土工程师复核。*")
    lines.append("")

    return "\n".join(lines)
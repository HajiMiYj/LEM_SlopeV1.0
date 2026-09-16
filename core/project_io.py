# -*- coding: utf-8 -*-
"""
工程文件序列化与数据存储模块
采用 V2 结构化格式，仅支持 V2，不再兼容 V1 旧文件
"""
import json
import os
from typing import Dict, Any, Optional


CURRENT_PROJECT_VERSION = 2
APP_SIGNATURE = "LEM-Slope-Studio"


def save_project_file(filepath: str, project_data: Dict[str, Any]) -> bool:
    """保存工程配置文件 (自动注入版本号与应用签名)"""
    try:
        payload = dict(project_data)
        payload["version"] = CURRENT_PROJECT_VERSION
        payload["app"] = APP_SIGNATURE

        parent = os.path.dirname(os.path.abspath(filepath))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[project_io] 保存失败: {e}")
        return False


def load_project_file(filepath: str) -> Optional[Dict[str, Any]]:
    """加载工程配置文件 (仅接受 V2 格式)"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[project_io] 读取失败: {e}")
        return None

    if not isinstance(data, dict):
        return None
    if data.get("version") != CURRENT_PROJECT_VERSION:
        print(f"[project_io] 不支持的工程版本: {data.get('version')}")
        return None
    return data
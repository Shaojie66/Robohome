"""把运行时的家写回规范文件（入库后更新台账用）。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import Home


def home_to_dict(home: Home) -> dict:
    return {
        "name": home.name,
        "rooms": [asdict(r) for r in home.rooms.values()],
        "storage": [asdict(s) for s in home.storage.values()],
        "fixtures": [asdict(f) for f in home.fixtures.values()],
        # 没有标签的容器不写 label，规范文件保持简洁
        "containers": [{k: v for k, v in asdict(c).items() if k != "label" or v} for c in home.containers.values()],
        "items": [asdict(i) for i in home.items.values()],
        "appliances": [asdict(a) for a in home.appliances.values()],
    }


def dumps_compact(value, width: int = 210, indent: int = 0) -> str:
    """缩进格式的 JSON，但放得下一行的对象、列表就写成一行（一条记录一行），方便人读和改。"""
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or len(flat) + indent <= width or not value:
        return flat
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(value, dict):
        body = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {dumps_compact(v, width, indent + 2)}" for k, v in value.items()]
        return "{\n" + ",\n".join(body) + f"\n{pad}}}"
    body = [inner + dumps_compact(v, width, indent + 2) for v in value]
    return "[\n" + ",\n".join(body) + f"\n{pad}]"


def save_home(home: Home, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps_compact(home_to_dict(home)) + "\n")

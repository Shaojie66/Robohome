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
        "containers": [asdict(c) for c in home.containers.values()],
        "items": [asdict(i) for i in home.items.values()],
        "appliances": [asdict(a) for a in home.appliances.values()],
    }


def save_home(home: Home, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(home_to_dict(home), fh, ensure_ascii=False, indent=2)
        fh.write("\n")

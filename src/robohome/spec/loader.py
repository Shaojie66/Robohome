"""读取和校验家用规范文件（JSON）。"""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    Appliance,
    Container,
    Fixture,
    Home,
    Item,
    Pose,
    Room,
    StorageArea,
)


class SpecError(ValueError):
    """规范文件有错：引用了不存在的 id、容器超出槽位等。"""


def _pose(d: dict | None) -> Pose | None:
    if d is None:
        return None
    return Pose(position=tuple(d["position"]), yaw_deg=d.get("yaw_deg", 0.0))


def _by_id(objs: list) -> dict:
    out = {}
    for o in objs:
        if o.id in out:
            raise SpecError(f"重复的 id: {o.id}")
        out[o.id] = o
    return out


def parse_home(data: dict) -> Home:
    rooms = _by_id(
        [
            Room(
                id=r["id"],
                name=r["name"],
                waypoints={k: _pose(v) for k, v in r.get("waypoints", {}).items()},
                forbidden_waypoints=r.get("forbidden_waypoints", []),
            )
            for r in data["rooms"]
        ]
    )
    fixtures = _by_id(
        [
            Fixture(
                id=f["id"],
                name=f["name"],
                kind=f["kind"],
                room=f["room"],
                waypoint=f["waypoint"],
                handle_pose=_pose(f["handle_pose"]),
                state=f.get("state", "closed"),
                encloses=f.get("encloses"),
            )
            for f in data.get("fixtures", [])
        ]
    )
    storage = _by_id([StorageArea(**s) for s in data.get("storage", [])])
    containers = _by_id(
        [
            Container(
                id=c["id"],
                name=c["name"],
                size_m=tuple(c["size_m"]),
                capacity_l=c["capacity_l"],
                location=c["location"],
                slot=c.get("slot", 0),
                pose=_pose(c.get("pose")),
                microwave_safe=c.get("microwave_safe", False),
            )
            for c in data.get("containers", [])
        ]
    )
    items = _by_id([Item(**i) for i in data.get("items", [])])
    appliances = _by_id([Appliance(**a) for a in data.get("appliances", [])])

    home = Home(
        name=data["name"],
        rooms=rooms,
        fixtures=fixtures,
        storage=storage,
        containers=containers,
        items=items,
        appliances=appliances,
    )
    validate(home)
    return home


def load_home(path: str | Path) -> Home:
    with open(path, encoding="utf-8") as fh:
        return parse_home(json.load(fh))


def validate(home: Home) -> None:
    """检查所有引用都能对上，保证机器人拿到的是一份自洽的家。"""
    errors: list[str] = []

    def waypoint_ok(room: str, wp: str, owner: str) -> None:
        if room not in home.rooms:
            errors.append(f"{owner}: 房间 {room} 不存在")
        elif wp not in home.rooms[room].waypoints:
            errors.append(f"{owner}: 房间 {room} 没有点位 {wp}")

    for f in home.fixtures.values():
        waypoint_ok(f.room, f.waypoint, f"部件 {f.id}")
        if f.encloses and f.encloses not in home.storage and f.encloses not in home.appliances:
            errors.append(f"部件 {f.id}: 围住的区域 {f.encloses} 不存在")

    for s in home.storage.values():
        if s.room not in home.rooms:
            errors.append(f"存放区 {s.id}: 房间 {s.room} 不存在")

    used: dict[tuple[str, int], str] = {}
    for c in home.containers.values():
        if c.location in home.storage:
            if not 0 <= c.slot < home.storage[c.location].slots:
                errors.append(f"容器 {c.id}: 槽位 {c.slot} 超出 {c.location} 的范围")
        elif c.location not in home.appliances:
            errors.append(f"容器 {c.id}: 位置 {c.location} 不存在")
        key = (c.location, c.slot)
        if key in used:
            errors.append(f"容器 {c.id} 和 {used[key]} 占了同一个槽位")
        used[key] = c.id

    for i in home.items.values():
        if i.container not in home.containers:
            errors.append(f"物品 {i.id}: 容器 {i.container} 不存在")

    for a in home.appliances.values():
        waypoint_ok(a.room, a.waypoint, f"家电 {a.id}")
        if a.door and a.door not in home.fixtures:
            errors.append(f"家电 {a.id}: 门 {a.door} 不存在")

    if errors:
        raise SpecError("规范文件校验失败：\n  " + "\n  ".join(errors))

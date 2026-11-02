"""读取和校验家用规范文件（JSON）。"""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    Appliance,
    Container,
    Fixture,
    Geometry,
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


def _geometry(d: dict | None) -> Geometry | None:
    if d is None:
        return None
    layout = d.get("layout")
    return Geometry(
        position=tuple(d["position"]),
        size_m=tuple(d["size_m"]),
        yaw_deg=d.get("yaw_deg", 0.0),
        layout=tuple(layout) if layout is not None else None,
    )


def _with_geometry(d: dict) -> dict:
    return {**d, "geometry": _geometry(d.get("geometry"))}


def _raw(data: dict, container_id: str) -> dict:
    return next((c for c in data.get("containers", []) if c.get("id") == container_id), {})


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
                hinge=f.get("hinge"),
            )
            for f in data.get("fixtures", [])
        ]
    )
    storage = _by_id([StorageArea(**_with_geometry(s)) for s in data.get("storage", [])])
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
                kind=c.get("kind", "box"),
                # 旧格式用 microwave_safe: true，等价于 safe_for: ["microwave"]
                safe_for=sorted(set(c.get("safe_for", [])) | ({"microwave"} if c.get("microwave_safe") else set())),
                home=c.get("home"),
                home_slot=c.get("home_slot"),
            )
            for c in data.get("containers", [])
        ]
    )
    # 没写原位的容器，建家时放在哪个存放区，原位就是哪里（台面和家电里的不算）
    for c in containers.values():
        area = storage.get(c.location)
        if c.home is None and area is not None and area.kind == "storage" and "home" not in _raw(data, c.id):
            c.home, c.home_slot = c.location, c.slot
    items = _by_id([Item(**i) for i in data.get("items", [])])
    appliances = _by_id([Appliance(**_with_geometry(a)) for a in data.get("appliances", [])])

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
        if s.waypoint:
            waypoint_ok(s.room, s.waypoint, f"存放区 {s.id}")
        elif home.fixture_enclosing(s.id) is None:
            errors.append(f"存放区 {s.id}: 没有门也没有写 waypoint，机器人不知道站在哪里取放")

    def slots_of(location: str) -> int | None:
        if location in home.storage:
            return home.storage[location].slots
        if location in home.appliances:
            return home.appliances[location].slots
        return None

    used: dict[tuple[str, int], str] = {}
    for c in home.containers.values():
        n = slots_of(c.location)
        if n is None:
            errors.append(f"容器 {c.id}: 位置 {c.location} 不存在")
        elif not 0 <= c.slot < n:
            errors.append(f"容器 {c.id}: 槽位 {c.slot} 超出 {c.location} 的范围")
        key = (c.location, c.slot)
        if key in used:
            errors.append(f"容器 {c.id} 和 {used[key]} 占了同一个槽位")
        used[key] = c.id
        if c.home is not None:
            if c.home not in home.storage:
                errors.append(f"容器 {c.id}: 原位 {c.home} 不是存放区")
            elif c.home_slot is not None and not 0 <= c.home_slot < home.storage[c.home].slots:
                errors.append(f"容器 {c.id}: 原位槽位 {c.home_slot} 超出 {c.home} 的范围")

    for i in home.items.values():
        if i.container not in home.containers and i.container not in home.appliances:
            errors.append(f"物品 {i.id}: 容器 {i.container} 不存在")
        for action, params in (i.cook or {}).items():
            able = [a for a in home.appliances.values() if action in a.capabilities]
            if not able:
                errors.append(f"物品 {i.id}: 没有家电能做 {action}")
            elif not isinstance(params, dict):
                errors.append(f"物品 {i.id}: {action} 的参数要写成对象")
            elif not any(params.get("minutes", 0) <= (a.max_minutes or 10**6) and params.get("temp_c", 0) <= (a.max_temp_c or 10**6) for a in able):
                errors.append(f"物品 {i.id}: {action} 的时间或温度超出家电上限")

    for owner, g, n in [(f"存放区 {s.id}", s.geometry, s.slots) for s in home.storage.values()] + [
        (f"家电 {a.id}", a.geometry, a.slots) for a in home.appliances.values()
    ]:
        if g is None:
            continue
        if len(g.position) != 3 or len(g.size_m) != 3 or min(g.size_m) <= 0:
            errors.append(f"{owner}: geometry 要写三维位置和三个正的尺寸")
        elif g.layout is not None and (len(g.layout) != 3 or min(g.layout) < 1 or g.layout[0] * g.layout[1] * g.layout[2] < n):
            errors.append(f"{owner}: layout {list(g.layout)} 排不下 {n} 个槽位")

    for f in home.fixtures.values():
        if f.hinge not in (None, "left", "right", "bottom"):
            errors.append(f"部件 {f.id}: hinge 只能是 left / right / bottom")

    for a in home.appliances.values():
        waypoint_ok(a.room, a.waypoint, f"家电 {a.id}")
        if a.door and a.door not in home.fixtures:
            errors.append(f"家电 {a.id}: 门 {a.door} 不存在")

    if errors:
        raise SpecError("规范文件校验失败：\n  " + "\n  ".join(errors))

"""从规范文件算出仿真要用的几何：每个存放区的局部坐标系、槽位中心、门轴和把手，以及建家时的几何检查。

这一层只用规范数据和简单的数学，不依赖 MuJoCo，所以几何检查不装仿真也能跑。

局部坐标系：原点在内部底面中心，x 朝宽度方向，y 朝里（开口在 y = -深/2 那一面），z 朝上。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..spec.models import Fixture, Geometry, Home

WALL = 0.02  # 柜壁厚度
DOOR_GAP = 0.005  # 门板和柜体之间留的缝


@dataclass(frozen=True)
class Reach:
    """机器人手能够到的范围。硬件还没定，先按常见人形机器人估一个，可以按实际机型改。"""

    horizontal_m: float = 1.1  # 手离底盘中心的水平距离
    min_z: float = 0.05  # 最低能够到的高度（蹲下）
    max_z: float = 1.95  # 最高能够到的高度


DEFAULT_REACH = Reach()


@dataclass(frozen=True)
class Area:
    """一块能放容器的地方：存放区，或者家电的内部（炸篮、洗碗机的碗架）。"""

    id: str
    name: str
    geometry: Geometry
    slots: int
    waypoint: str | None
    fixture: Fixture | None  # 围着它的门或抽屉
    style: str  # cabinet 有柜壁和门 / drawer 东西放在抽屉里，抽屉拉出来再取放 / surface 台面
    is_appliance: bool = False

    @property
    def layout(self) -> tuple[int, int, int]:
        return self.geometry.layout or (max(1, self.slots), 1, 1)

    @property
    def cell(self) -> tuple[float, float, float]:
        cols, rows, levels = self.layout
        w, d, h = self.geometry.size_m
        return w / cols, d / rows, h / levels

    def slot_local(self, slot: int) -> tuple[float, float, float]:
        """槽位底面中心在局部坐标系里的位置。0 号在最外面一行、最下面一层的最左边。"""
        cols, rows, _ = self.layout
        w, d, _ = self.geometry.size_m
        cw, cd, ch = self.cell
        level, rest = divmod(slot, cols * rows)
        row, col = divmod(rest, cols)
        return (-w / 2 + (col + 0.5) * cw, -d / 2 + (row + 0.5) * cd, level * ch)

    @property
    def approach(self) -> str:
        """手从哪个方向伸进去：抽屉和台面从上面，柜子从正面。"""
        return "front" if self.style == "cabinet" else "top"

    @property
    def drawer_travel(self) -> float:
        return round(max(0.15, 0.8 * self.geometry.size_m[1]), 3)


def areas(home: Home) -> dict[str, Area]:
    """所有写了三维尺寸、能放容器的地方。"""
    out: dict[str, Area] = {}
    for s in home.storage.values():
        if s.geometry is None:
            continue
        fixture = home.fixture_enclosing(s.id)
        out[s.id] = Area(s.id, s.name, s.geometry, s.slots, s.waypoint, fixture, _style(fixture, s.kind == "surface"))
    for a in home.appliances.values():
        if a.geometry is None:
            continue
        fixture = home.fixtures.get(a.door) if a.door else home.fixture_enclosing(a.id)
        out[a.id] = Area(a.id, a.name, a.geometry, a.slots, a.waypoint, fixture, _style(fixture, fixture is None), is_appliance=True)
    return out


def _style(fixture: Fixture | None, surface: bool) -> str:
    if fixture is None:
        return "surface" if surface else "cabinet"
    return "drawer" if fixture.kind == "drawer" else "cabinet"


# ---------- 坐标变换 ----------


def rotate_z(v: tuple[float, float, float], yaw_rad: float) -> tuple[float, float, float]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def to_world(area: Area, local: tuple[float, float, float]) -> tuple[float, float, float]:
    """抽屉关着时，局部坐标换成世界坐标。"""
    x, y, z = rotate_z(local, math.radians(area.geometry.yaw_deg))
    px, py, pz = area.geometry.position
    return (px + x, py + y, pz + z)


def to_local(area: Area, world: tuple[float, float, float]) -> tuple[float, float, float]:
    px, py, pz = area.geometry.position
    return rotate_z((world[0] - px, world[1] - py, world[2] - pz), -math.radians(area.geometry.yaw_deg))


def hinge_side(area: Area) -> str:
    """门轴在哪边：规范里写了就用，没写就放在离把手远的一边。"""
    f = area.fixture
    if f is None or f.kind == "drawer":
        return "none"
    if f.hinge:
        return f.hinge
    return "left" if to_local(area, f.handle_pose.position)[0] >= 0 else "right"


# ---------- 建家时的几何检查 ----------


def _horizontal(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def check_geometry(home: Home, reach: Reach = DEFAULT_REACH) -> list[str]:
    """检查规范文件里的三维数据能不能让机器人真的做事：容器放得进槽位、把手和槽位够得着、把手在门上。

    返回发现的问题，空列表表示没问题。
    """
    problems: list[str] = []
    found = areas(home)
    room_waypoints = {wp: pose for r in home.rooms.values() for wp, pose in r.waypoints.items()}

    used = {c.location for c in home.containers.values()} | {c.home for c in home.containers.values() if c.home}
    for loc in sorted(used - set(found)):
        if loc in home.storage or loc in home.appliances:
            name = (home.storage.get(loc) or home.appliances.get(loc)).name
            problems.append(f"{name}（{loc}）放着容器，但没有写三维尺寸（geometry），仿真搭不出来")

    for area in found.values():
        cols, rows, levels = area.layout
        if cols * rows * levels < area.slots:
            problems.append(f"{area.name}：排布 {cols}×{rows}×{levels} 放不下 {area.slots} 个槽位")
        stand = _stand_point(home, area)
        if stand is None:
            problems.append(f"{area.name}：不知道机器人站在哪里操作（门的点位或 waypoint）")
            continue
        drawer_offset = area.drawer_travel if area.style == "drawer" else 0.0
        for slot in range(area.slots):
            x, y, z = area.slot_local(slot)
            target = to_world(area, (x, y - drawer_offset, z))
            dist = _horizontal(target, room_waypoints[stand].position)
            if dist > reach.horizontal_m or not reach.min_z <= target[2] <= reach.max_z:
                problems.append(
                    f"{area.name}槽位 {slot}：离点位 {stand} 水平 {dist:.2f} 米、高 {target[2]:.2f} 米，手够不着"
                    f"（范围是水平 {reach.horizontal_m} 米以内、高 {reach.min_z} 到 {reach.max_z} 米）"
                )
                break  # 一个存放区只报第一个够不着的槽位
        if area.fixture is not None:
            problems += _check_handle(area, room_waypoints, reach)

    cw_cache = {a.id: a.cell for a in found.values()}
    for c in home.containers.values():
        for loc in {c.location, c.home} - {None}:
            if loc not in cw_cache:
                continue
            area = found[loc]
            cw, cd, ch = cw_cache[loc]
            sx, sy, sz = c.size_m
            fits_flat = (sx <= cw and sy <= cd) or (sy <= cw and sx <= cd)
            tall_ok = area.style == "surface" or sz <= ch - 0.04  # 上面留点缝给手
            if not (fits_flat and tall_ok):
                problems.append(
                    f"{c.name}（{sx}×{sy}×{sz} 米）放不进{area.name}的槽位（每格 {cw:.2f}×{cd:.2f}×{ch:.2f} 米）"
                )
    return problems


def _stand_point(home: Home, area: Area) -> str | None:
    if area.fixture is not None:
        return area.fixture.waypoint
    return area.waypoint or (home.appliances[area.id].waypoint if area.is_appliance else None)


def _check_handle(area: Area, waypoints: dict, reach: Reach) -> list[str]:
    f = area.fixture
    out = []
    hx, hy, hz = to_local(area, f.handle_pose.position)
    w, d, h = area.geometry.size_m
    front = -d / 2 - WALL
    if abs(hy - front) > 0.1 or abs(hx) > w / 2 + WALL + 0.05 or not -WALL - 0.05 <= hz <= h + WALL + 0.05:
        out.append(f"{f.name}的把手（{f.handle_pose.position}）不在{area.name}的正面上，检查把手坐标或存放区尺寸")
    pos = f.handle_pose.position
    dist = _horizontal(pos, waypoints[f.waypoint].position)
    if dist > reach.horizontal_m or not reach.min_z <= pos[2] <= reach.max_z:
        out.append(f"{f.name}的把手离点位 {f.waypoint} 水平 {dist:.2f} 米、高 {pos[2]:.2f} 米，手够不着")
    return out

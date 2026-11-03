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
HAND_ABOVE_TOP = 0.012  # 磁吸板贴在容器顶上，手的位置比容器顶高这么多
RELEASE_GAP = 0.01  # 放下时容器离底面这么高再松手


@dataclass(frozen=True)
class Reach:
    """机器人手能够到的范围：每个高度上，手离底盘中心（两脚中间）水平最远多少米。

    profile 是 (高度, 最远水平距离) 的表，高度从低到高，中间按直线插值；比表里最低的低、最高的高都够不着。
    """

    name: str
    profile: tuple[tuple[float, float], ...]
    source: str = ""

    @property
    def min_z(self) -> float:
        return self.profile[0][0]

    @property
    def max_z(self) -> float:
        return self.profile[-1][0]

    def max_horizontal(self, z: float) -> float:
        """这个高度上手最远能伸到离底盘中心多远；够不着这个高度时返回 0。"""
        pts = self.profile
        if not pts[0][0] - 1e-9 <= z <= pts[-1][0] + 1e-9:
            return 0.0
        for (z0, r0), (z1, r1) in zip(pts, pts[1:]):
            if z <= z1:
                return r0 if z1 == z0 else r0 + (r1 - r0) * (z - z0) / (z1 - z0)
        return pts[-1][1]

    def reaches(self, horizontal: float, z: float) -> bool:
        return horizontal <= self.max_horizontal(z) + 1e-6

    def limit_text(self, z: float) -> str:
        r = self.max_horizontal(z)
        who = self.name + (" " if self.name[-1].isascii() else "")
        if r <= 0:
            return f"{who}的手能伸到的高度是 {self.min_z:g} 到 {self.max_z:g} 米"
        return f"{who}在这个高度手最远伸到水平 {r:.2f} 米"

    @classmethod
    def box(cls, horizontal_m: float, min_z: float, max_z: float, name: str = "机器人") -> Reach:
        """各个高度一样远的简单范围（之前的估算就是这样）。"""
        return cls(name, ((min_z, horizontal_m), (max_z, horizontal_m)))


# 宇树 G1（29 自由度）：用 MuJoCo Menagerie 里的公开模型（unitree_g1/g1.xml）算出来，重算见 scripts/g1_reach.py。
# 姿态限制：两脚踩平不动，蹲、弯腰、转腰、伸手都在关节限位内，上身前倾不超过 0.5 弧度，
# 手里拿 1 公斤东西时重心仍落在两脚之间（留 2 厘米余量）；手的位置取掌心。
# 每个高度取算出来的最远距离再减 5 厘米，留给手腕朝向和关节别顶死。没算全身动态平衡。
G1_REACH = Reach(
    "宇树 G1",
    (
        (0.05, 0.51), (0.15, 0.59), (0.25, 0.64), (0.35, 0.67), (0.45, 0.68), (0.55, 0.70), (0.65, 0.71),
        (0.75, 0.73), (0.85, 0.74), (0.95, 0.74), (1.05, 0.72), (1.15, 0.69), (1.25, 0.64), (1.35, 0.57),
        (1.45, 0.47), (1.55, 0.28),
    ),
    source="MuJoCo Menagerie unitree_g1（由宇树公开的 g1_29dof_rev_1_0 模型整理），scripts/g1_reach.py",
)

# 第一阶段的估算：常见人形机器人，水平 1.1 米、高 0.05 到 1.95 米。比 G1 宽松得多，留着做对比
GENERIC_REACH = Reach.box(1.1, 0.05, 1.95, name="估算的人形机器人")

DEFAULT_REACH = G1_REACH


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
        out[s.id] = Area(s.id, s.name, s.geometry, s.slots, s.waypoint, fixture, _style(fixture, s.kind in ("surface", "intake")))
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
    forbidden = {wp for r in home.rooms.values() for wp in r.forbidden_waypoints}

    used = {c.location for c in home.containers.values()} | {c.home for c in home.containers.values() if c.home}
    for loc in sorted(used - set(found)):
        if loc in home.storage or loc in home.appliances:
            name = (home.storage.get(loc) or home.appliances.get(loc)).name
            problems.append(f"{name}（{loc}）放着容器，但没有写三维尺寸（geometry），仿真搭不出来")

    for area in found.values():
        # 取放时手在容器顶上，按会放在这里的最高的容器算手要伸到的高度
        tallest = max((c.size_m[2] for c in home.containers.values() if area.id in (c.location, c.home)), default=None)
        lift = 0.0 if tallest is None else tallest + HAND_ABOVE_TOP + RELEASE_GAP
        cols, rows, levels = area.layout
        if cols * rows * levels < area.slots:
            problems.append(f"{area.name}：排布 {cols}×{rows}×{levels} 放不下 {area.slots} 个槽位")
        stand = _stand_point(home, area)
        if stand is None:
            problems.append(f"{area.name}：不知道机器人站在哪里操作（门的点位或 waypoint）")
            continue
        if stand in forbidden:
            continue  # 机器人不去禁区（例如灶台前），那里的东西本来就不归它
        drawer_offset = area.drawer_travel if area.style == "drawer" else 0.0
        for slot in range(area.slots):
            x, y, z = area.slot_local(slot)
            target = to_world(area, (x, y - drawer_offset, z + lift))
            dist = _horizontal(target, room_waypoints[stand].position)
            if not reach.reaches(dist, target[2]):
                problems.append(
                    f"{area.name}槽位 {slot}：取放时手要伸到离点位 {stand} 水平 {dist:.2f} 米、高 {target[2]:.2f} 米，手够不着"
                    f"（{reach.limit_text(target[2])}）"
                )
                break  # 一个存放区只报第一个够不着的槽位
        if area.fixture is not None:
            problems += _check_handle(area, room_waypoints, reach)

    from .floor import check_floor  # floor 用到这里的 areas，放在函数里引入免得互相引用

    problems += check_floor(home)

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
    for what, pos in (("", f.handle_pose.position), ("开到底时", handle_when_open(area))):
        dist = _horizontal(pos, waypoints[f.waypoint].position)
        if not reach.reaches(dist, pos[2]):
            out.append(
                f"{f.name}的把手{what}离点位 {f.waypoint} 水平 {dist:.2f} 米、高 {pos[2]:.2f} 米，手够不着（{reach.limit_text(pos[2])}）"
            )
            break
    return out


DOOR_OPEN_RAD = {"left": 1.75, "right": 1.75, "bottom": 1.5}  # 侧开门开 100 度，下翻门放平


def handle_when_open(area: Area) -> tuple[float, float, float]:
    """门或抽屉完全打开时把手在哪（世界坐标）。拉门的手一路跟着把手走，开到底时离机器人最远。"""
    f = area.fixture
    hx, hy, hz = to_local(area, f.handle_pose.position)
    w, d, h = area.geometry.size_m
    t = WALL
    side = hinge_side(area)
    if side == "none":
        return to_world(area, (hx, hy - area.drawer_travel, hz))
    front_y = -d / 2 - t - DOOR_GAP - t / 2
    angle = DOOR_OPEN_RAD[side]
    if side == "bottom":  # 绕门底的横轴往外翻
        ry, rz = hy - front_y, hz + t
        c, s_ = math.cos(angle), math.sin(angle)
        return to_world(area, (hx, front_y + c * ry - s_ * rz, -t + s_ * ry + c * rz))
    W = w + 2 * t
    ox = -W / 2 if side == "left" else W / 2
    a = -angle if side == "left" else angle  # 往外开：左边门轴顺时针转，右边逆时针
    rx, ry = hx - ox, hy - front_y
    c, s_ = math.cos(a), math.sin(a)
    return to_world(area, (ox + c * rx - s_ * ry, front_y + s_ * rx + c * ry, hz))

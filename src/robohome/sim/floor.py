"""地面地图：家具在地上占的地方、机器人站在哪、怎么走过去。

只用规范数据和简单数学，不依赖 MuJoCo，建家时的几何检查也用它。仿真里再把激光看到的障碍加进来。

- 障碍：每个写了三维尺寸的存放区和家电占的地面（含柜壁）、拉开着的抽屉、开着的门、禁区点位周围一圈，
  还有柜子背后的墙和小家电下面的台面（见 layout.decor）
- 走路：机器人按一个圆（Footprint.radius）在格子地图上找路，再把折线拉直
- 站位：机器人在点位上干活时脚尖可以离家具更近（Footprint.front），所以点位本身可以在"走路的圆"碰到家具的地方；
  走进去和退出来都沿着点位的朝向直着走
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from ..spec.models import Home, Pose
from .layout import DOOR_GAP, WALL, Area, areas, decor, hinge_side

RESOLUTION = 0.05  # 格子边长
FORBIDDEN_RADIUS = 0.3  # 禁区点位周围多大一圈不能进（大约一个灶台宽）
APPROACH_MAX = 0.6  # 从点位最多往后退这么远，找一个能转身的地方
SEEN_POINT = 0.03  # 激光看到的一个点按这么大的一小块算（相邻两条激光在两米外差不多隔这么远）
_BUCKET = 0.5


@dataclass(frozen=True)
class Footprint:
    """机器人在地上占多大。默认按宇树 G1：肩宽约 0.45 米、脚长约 0.17 米。"""

    radius: float = 0.25  # 走路、转身时占的圆
    front: float = 0.15  # 站定干活时，底盘中心到脚尖
    back: float = 0.12
    half_width: float = 0.23

    def covers(self, pose: tuple[float, float, float], x: float, y: float, pad: float = 0.0) -> bool:
        """站在 pose 时，地上的点 (x, y) 在不在身体占的长方形里。"""
        px, py, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = x - px, y - py
        f, side = c * dx + s * dy, -s * dx + c * dy
        return -self.back - pad <= f <= self.front + pad and abs(side) <= self.half_width + pad

    def corners(self, pose: tuple[float, float, float]) -> list[tuple[float, float]]:
        """站定时占的长方形边上的一圈点（底盘中心 x、y，朝向弧度）。"""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        pts = []
        for f in _steps(-self.back, self.front):
            for side in (-self.half_width, self.half_width):
                pts.append((x + c * f - s * side, y + s * f + c * side))
        for side in _steps(-self.half_width, self.half_width):
            for f in (-self.back, self.front):
                pts.append((x + c * f - s * side, y + s * f + c * side))
        return pts


G1_FOOTPRINT = Footprint()


def _steps(a: float, b: float, step: float = 0.04) -> list[float]:
    n = max(1, math.ceil((b - a) / step))
    return [a + (b - a) * i / n for i in range(n + 1)]


@dataclass(frozen=True)
class Box2D:
    """地上的一块长方形障碍。"""

    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float
    what: str

    def distance(self, x: float, y: float) -> float:
        """点到长方形的距离；在里面是 0。"""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.cx, y - self.cy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        ox, oy = max(abs(lx) - self.hx, 0.0), max(abs(ly) - self.hy, 0.0)
        return math.hypot(ox, oy)


def _local_box(area: Area, x0: float, x1: float, y0: float, y1: float, what: str) -> Box2D:
    """存放区局部坐标里的一块（x、y 范围）换成地上的长方形。"""
    g = area.geometry
    yaw = math.radians(g.yaw_deg)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return Box2D(g.position[0] + c * cx - s * cy, g.position[1] + s * cx + c * cy, (x1 - x0) / 2, (y1 - y0) / 2, yaw, what)


def area_box(area: Area) -> Box2D:
    w, d, _ = area.geometry.size_m
    t = WALL
    front = -d / 2 - t - (DOOR_GAP + t if area.fixture is not None else 0.0)  # 关着的门板也算进去
    return _local_box(area, -w / 2 - t, w / 2 + t, front, d / 2 + t, area.name)


def open_box(area: Area) -> Box2D | None:
    """部件开着时在地上多占的地方：拉开的抽屉，或者开着的门板。门开到一半按全开算。"""
    f = area.fixture
    if f is None:
        return None
    w, d, h = area.geometry.size_m
    t = WALL
    front = -d / 2 - t - DOOR_GAP - t
    side = hinge_side(area)
    if side == "none":  # 抽屉整个往外拉出 drawer_travel
        return _local_box(area, -w / 2 - t, w / 2 + t, front - area.drawer_travel, front, f"拉开的{f.name}")
    W, H = w + 2 * t, h + 2 * t
    if side == "bottom":  # 下翻门放平后伸出去一个门高
        if area.geometry.position[2] - t > 0.5:
            return None  # 离地很高的下翻门不挡路
        return _local_box(area, -W / 2, W / 2, front - H, front, f"开着的{f.name}")
    if area.geometry.position[2] - t > 1.2:
        return None  # 吊柜的门开着也不挡地上的路
    # 侧开门开到 100 度，门板几乎垂直于柜面，伸出去一个门宽
    x = -W / 2 if side == "left" else W / 2
    return _local_box(area, min(x, x - math.copysign(t, x)), max(x, x - math.copysign(t, x)), front - W, front, f"开着的{f.name}")


@dataclass
class FloorMap:
    """格子地图。clearance[i][j] 是格子中心到最近障碍的距离。"""

    boxes: list[Box2D]
    zones: list[tuple[float, float, float, str]]  # 禁区：圆心、半径、名字
    x0: float
    y0: float
    nx: int
    ny: int
    clearance: list[list[float]] = field(default_factory=list)
    seen: list[tuple[float, float]] = field(default_factory=list)  # 仿真里激光看到的点
    _buckets: dict[tuple[int, int], list[tuple[float, float]]] = field(default_factory=dict)

    @classmethod
    def of(cls, home: Home, open_fixtures: set[str] | frozenset[str] = frozenset(), margin: float = 1.0) -> FloorMap:
        found = areas(home)
        boxes = [area_box(a) for a in found.values()]
        boxes += [b for a in found.values() if a.fixture is not None and a.fixture.id in open_fixtures if (b := open_box(a))]
        for d in decor(found):
            x0, x1, y0, y1 = d.x0, d.x1, d.y0, d.y1
            if d.name == "墙":  # 墙后面走不过去：把墙往两头和后面延长到地图外
                x0, x1, y1 = x0 - 2 * margin, x1 + 2 * margin, y1 + 2 * margin
            boxes.append(Box2D((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 0.0, d.name))
        zones = []
        for r in home.rooms.values():
            for wp in r.forbidden_waypoints:
                if wp in r.waypoints:
                    p = r.waypoints[wp].position
                    zones.append((p[0], p[1], FORBIDDEN_RADIUS, f"禁区 {wp}"))
        inside = [b for b in boxes if b.what != "墙"]
        xs = [b.cx for b in inside] + [p.position[0] for r in home.rooms.values() for p in r.waypoints.values()]
        ys = [b.cy for b in inside] + [p.position[1] for r in home.rooms.values() for p in r.waypoints.values()]
        x0, y0 = min(xs) - margin, min(ys) - margin
        nx = math.ceil((max(xs) + margin - x0) / RESOLUTION) + 1
        ny = math.ceil((max(ys) + margin - y0) / RESOLUTION) + 1
        fm = cls(boxes, zones, x0, y0, nx, ny)
        fm.clearance = [[fm._static_clearance(*fm.center(i, j)) for j in range(ny)] for i in range(nx)]
        return fm

    # ---------- 距离 ----------

    def center(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + i * RESOLUTION, self.y0 + j * RESOLUTION

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.x0) / RESOLUTION), round((y - self.y0) / RESOLUTION)

    def _static_clearance(self, x: float, y: float) -> float:
        d = min((b.distance(x, y) for b in self.boxes), default=9.0)
        for zx, zy, r, _ in self.zones:
            d = min(d, max(0.0, math.hypot(x - zx, y - zy) - r))
        return d

    def static_distance(self, x: float, y: float) -> float:
        """点到规范里已知的障碍（家具、墙、禁区）的距离，不算激光看到的。"""
        return self._static_clearance(x, y)

    def _near_seen(self, x: float, y: float) -> list[tuple[float, float]]:
        bx, by = math.floor(x / _BUCKET), math.floor(y / _BUCKET)
        return [p for i in (-1, 0, 1) for j in (-1, 0, 1) for p in self._buckets.get((bx + i, by + j), ())]

    def distance(self, x: float, y: float) -> float:
        """点到最近障碍（含激光看到的）的距离。激光点只算半米以内的，再远的不影响走路和站位。"""
        d = self._static_clearance(x, y)
        for px, py in self._near_seen(x, y):
            d = min(d, max(0.0, math.hypot(x - px, y - py) - SEEN_POINT))
        return d

    def nearest(self, x: float, y: float) -> str:
        """离这个点最近的障碍是什么，用来写原因。"""
        best, what = 9.0, "障碍"
        for b in self.boxes:
            if (dd := b.distance(x, y)) < best:
                best, what = dd, b.what
        for zx, zy, r, name in self.zones:
            if (dd := max(0.0, math.hypot(x - zx, y - zy) - r)) < best:
                best, what = dd, name
        for px, py in self._near_seen(x, y):
            if (dd := max(0.0, math.hypot(x - px, y - py) - SEEN_POINT)) < best:
                best, what = dd, "地上有东西"
        return what

    def add_seen(self, points: list[tuple[float, float]], influence: float = _BUCKET) -> int:
        """加进仿真里看到的障碍点（离已有的点不到 2 厘米的不重复加），更新附近格子的距离。返回新加了几个。"""
        k = math.ceil(influence / RESOLUTION)
        added = 0
        for px, py in points:
            if any(math.hypot(px - qx, py - qy) < 0.02 for qx, qy in self._near_seen(px, py)):
                continue
            added += 1
            self.seen.append((px, py))
            self._buckets.setdefault((math.floor(px / _BUCKET), math.floor(py / _BUCKET)), []).append((px, py))
            ci, cj = self.cell(px, py)
            for i in range(max(0, ci - k), min(self.nx, ci + k + 1)):
                for j in range(max(0, cj - k), min(self.ny, cj + k + 1)):
                    x, y = self.center(i, j)
                    d = max(0.0, math.hypot(x - px, y - py) - SEEN_POINT)
                    if d < self.clearance[i][j]:
                        self.clearance[i][j] = d
        return added

    def seen_blocks(self, pts: list[tuple[float, float]], radius: float) -> bool:
        """沿这条折线走，身体（半径 radius 的圆）会不会碰到激光看到的东西。"""
        for a, b in zip(pts, pts[1:]):
            n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / RESOLUTION))
            for k in range(n + 1):
                x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
                if any(math.hypot(x - px, y - py) < radius + SEEN_POINT for px, py in self._near_seen(x, y)):
                    return True
        return False

    def _free(self, i: int, j: int, radius: float) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and self.clearance[i][j] > radius

    def segment_clear(self, a: tuple[float, float], b: tuple[float, float], radius: float) -> bool:
        n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (RESOLUTION / 2)))
        return all(
            self.distance(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) > radius for k in range(n + 1)
        )

    # ---------- 站位 ----------

    def stand_problem(self, pose: tuple[float, float, float], fp: Footprint = G1_FOOTPRINT) -> str | None:
        """站在这里时身体会不会碰到障碍；碰到就返回碰到了什么。"""
        for x, y in fp.corners(pose):
            if self._static_clearance(x, y) < 1e-6:
                return self.nearest(x, y)
        for x, y in self._near_seen(pose[0], pose[1]):
            if fp.covers(pose, x, y, pad=SEEN_POINT):
                return "地上有东西"
        return None

    def approach(self, pose: tuple[float, float, float], fp: Footprint = G1_FOOTPRINT) -> tuple[float, float] | None:
        """从点位沿朝向往后退，找到第一个能转身的地方；退的路上身体碰到东西，或者退 APPROACH_MAX 米还不行，返回 None。"""
        x, y, yaw = pose
        for k in range(round(APPROACH_MAX / RESOLUTION) + 1):
            d = k * RESOLUTION
            p = (x - d * math.cos(yaw), y - d * math.sin(yaw))
            if self.stand_problem((p[0], p[1], yaw), fp) is not None:
                return None
            if self.distance(*p) > fp.radius:
                return p
        return None

    # ---------- 找路 ----------

    def route(
        self, start: tuple[float, float, float], goal: tuple[float, float, float], fp: Footprint = G1_FOOTPRINT
    ) -> tuple[list[tuple[float, float]] | None, str]:
        """从 start 走到 goal（底盘中心 x、y、朝向）的折线；走不过去返回 (None, 原因)。"""
        if (hit := self.stand_problem(goal, fp)) is not None:
            return None, f"点位上站不下：{hit}"
        a = self.approach(start, fp) if self.distance(start[0], start[1]) <= fp.radius else (start[0], start[1])
        b = self.approach(goal, fp)
        if a is None:
            return None, f"退不出来：{self.nearest(start[0], start[1])}"
        if b is None:
            return None, f"走不到点位跟前：{self.nearest(goal[0], goal[1])}"
        middle = self._astar(a, b, fp.radius)
        if middle is None:
            return None, "找不到能过去的路"
        pts = [(start[0], start[1]), *middle, (goal[0], goal[1])]
        out = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
                out.append(p)
        return out, ""

    def _astar(self, a: tuple[float, float], b: tuple[float, float], radius: float) -> list[tuple[float, float]] | None:
        sa, sb = self.cell(*a), self.cell(*b)
        if not (self._free(*sa, radius) or self.distance(*a) > radius) or not (self._free(*sb, radius) or self.distance(*b) > radius):
            return None
        moves = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        best = {sa: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(0.0, sa)]
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == sb:
                break
            g = best[cur]
            for di, dj, cost in moves:
                nxt = (cur[0] + di, cur[1] + dj)
                if nxt != sb and not self._free(*nxt, radius):
                    continue
                ng = g + cost
                if ng < best.get(nxt, 1e18):
                    best[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(heap, (ng + math.hypot(sb[0] - nxt[0], sb[1] - nxt[1]), nxt))
        if sb not in best:
            return None
        cells = [sb]
        while cells[-1] != sa:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [a, *(self.center(*c) for c in cells[1:-1]), b]
        return self._shortcut(pts, radius)

    def _shortcut(self, pts: list[tuple[float, float]], radius: float) -> list[tuple[float, float]]:
        """能直着走就直着走：去掉中间不必要的拐点。"""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_clear(pts[i], pts[j], radius):
                j -= 1
            out.append(pts[j])
            i = j
        return out


def pose_of(p: Pose) -> tuple[float, float, float]:
    return p.position[0], p.position[1], math.radians(p.yaw_deg)


def path_length(pts: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def check_floor(home: Home, fp: Footprint = G1_FOOTPRINT) -> list[str]:
    """建家时检查地上：每个干活的点位站得下（在这里拉开的抽屉、开着的门不碰到身体），从入口都走得到。"""
    problems: list[str] = []
    found = areas(home)
    waypoints = {wp: pose for r in home.rooms.values() for wp, pose in r.waypoints.items()}
    forbidden = {wp for r in home.rooms.values() for wp in r.forbidden_waypoints}
    used: dict[str, set[str]] = {}  # 点位 → 在这里打开的部件
    for f in home.fixtures.values():
        used.setdefault(f.waypoint, set()).add(f.id)
    for s in home.storage.values():
        if s.waypoint:
            used.setdefault(s.waypoint, set())
    closed = FloorMap.of(home)
    for wp, fixtures in sorted(used.items()):
        if wp in forbidden or wp not in waypoints:
            continue
        pose = pose_of(waypoints[wp])
        if (hit := closed.stand_problem(pose, fp)) is not None:
            problems.append(f"点位 {wp}：机器人站在这里会碰到{hit}")
            continue
        for fid in sorted(fixtures):
            area = next((a for a in found.values() if a.fixture is not None and a.fixture.id == fid), None)
            if area is None:
                continue
            opened = FloorMap.of(home, open_fixtures={fid})
            if (hit := opened.stand_problem(pose, fp)) is not None:
                problems.append(f"点位 {wp}：{hit}会碰到站在这里的机器人，往后挪一点")
    start = waypoints.get("entrance") or next(iter(waypoints.values()), None)
    if start is not None:
        for wp in sorted(used):
            if wp in forbidden or wp not in waypoints or waypoints[wp] is start:
                continue
            _, why = closed.route(pose_of(start), pose_of(waypoints[wp]), fp)
            if why:
                problems.append(f"从入口走不到点位 {wp}：{why}")
    return problems

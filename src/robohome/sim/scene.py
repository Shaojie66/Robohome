"""把规范文件里的家搭成 MuJoCo 模型（MJCF）。

- 柜子：底、顶、两侧、背板和隔板是不动的；门是带转轴的门板，由一个力气有限的"手"（位置执行器）去拉
- 抽屉：带滑轨的托盘，东西放在托盘里，拉开时跟着一起出来
- 台面：一块桌板
- 容器：有重量、会摩擦、会掉落的刚体
- 机器人：底盘和手都按给定轨迹移动（运动学控制），身体先不参与碰撞；手是一块磁吸板，吸住的容器跟着手走

门卡住用转轴上很大的摩擦来模拟：手的力气拉不动，门就真的打不开。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from ..spec.models import Container, Home
from .layout import DOOR_GAP, DOOR_OPEN_RAD, WALL, Area, areas, hinge_side, to_local, to_world
DOOR_FORCE = 25.0  # 拉门的最大力矩（N·m）
DRAWER_FORCE = 90.0  # 拉抽屉的最大力（N）
STUCK_FRICTION = 5000.0  # 卡住的门轴摩擦，远大于手的力气

COLORS = {
    "frozen": "0.78 0.88 0.97 1",
    "chilled": "0.86 0.93 0.97 1",
    "ambient": "0.80 0.70 0.58 1",
    "surface": "0.62 0.45 0.30 1",
    "intake": "0.85 0.80 0.70 1",
    "appliance": "0.30 0.32 0.36 1",
    "door": "0.93 0.93 0.92 1",
    "appliance_door": "0.20 0.21 0.24 1",
}
CONTAINER_COLORS = {
    "box": "0.35 0.62 0.86 1",
    "jar": "0.95 0.75 0.30 1",
    "bowl": "0.96 0.96 0.94 1",
    "cup": "0.55 0.80 0.62 1",
    "object": "0.80 0.45 0.45 1",
    "parcel": "0.76 0.60 0.40 1",  # 牛皮纸袋
}


@dataclass
class SceneInfo:
    """搭模型时记下的名字和参数，执行动作时按名字找到对应的物体。"""

    areas: dict[str, Area]
    frame_body: dict[str, str] = field(default_factory=dict)  # 存放区 id → 它的局部坐标系所在的物体
    door_open: dict[str, float] = field(default_factory=dict)  # 部件 id → 开到多少（弧度或米）
    door_area: dict[str, str] = field(default_factory=dict)  # 部件 id → 它围着的存放区
    containers: list[str] = field(default_factory=list)  # 场景里实际有的容器（台账里有、现实中不在的不放）
    unknown: dict[str, list[str]] = field(default_factory=dict)  # 存放区 → 台账里没有的东西的物体名
    pour_into: list[str] = field(default_factory=list)  # 能倒东西进去的家电


def _f(v) -> str:
    return " ".join(f"{x:.4f}" for x in v)


def _yaw_quat(deg: float) -> str:
    r = math.radians(deg) / 2
    return _f((math.cos(r), 0, 0, math.sin(r)))


def build_mjcf(
    home: Home,
    missing: set[str] | frozenset[str] = frozenset(),
    stuck: set[str] | frozenset[str] = frozenset(),
    unknown: dict[str, list[str]] | None = None,
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[str, SceneInfo]:
    """生成 MJCF 文本。missing 里的容器不放进场景（台账有、现实没有），stuck 里的门转轴摩擦极大。"""
    info = SceneInfo(areas=areas(home))
    world: list[str] = []
    actuators: list[str] = []

    for area in info.areas.values():
        world.append(_area_xml(home, area, info, stuck, actuators))

    for c in home.containers.values():
        if c.id in missing or c.location not in info.areas:
            continue
        world.append(_container_xml(c, info.areas[c.location], c.slot))
        info.containers.append(c.id)

    for loc, things in (unknown or {}).items():
        area = info.areas.get(loc)
        if area is None:
            continue
        taken = {c.slot for c in home.containers.values() if c.location == loc}
        free = [s for s in range(area.slots) if s not in taken] or [0]
        for n, _ in enumerate(things):
            name = f"unknown_{loc}_{n}"
            x, y, z = area.slot_local(free[n % len(free)])
            # 台账里没有的东西：一个灰色的袋子，放在空槽位上
            world.append(
                f'<body name="{name}" pos="{_f(to_world(area, (x, y, z + 0.05)))}">'
                f'<freejoint/><geom type="ellipsoid" size="0.09 0.07 0.05" rgba="0.45 0.45 0.45 1" mass="0.3" class="item"/></body>'
            )
            info.unknown.setdefault(loc, []).append(name)

    world.append(_decor_xml(info))
    world.append(_robot_xml(start))
    xml = f"""<mujoco model={quoteattr(home.name)}>
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="960" offheight="640"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" rgb2="0.78 0.80 0.84" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.84 0.80" rgb2="0.80 0.78 0.74" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0" specular="0.1"/>
  </asset>
  <default>
    <geom friction="0.9 0.01 0.001" solref="0.005 1" condim="3"/>
    <joint damping="2" armature="0.01"/>
    <default class="visual"><geom contype="0" conaffinity="0" group="1"/></default>
    <default class="moving"><geom contype="4" conaffinity="4"/></default>
    <default class="item"><geom contype="5" conaffinity="5"/></default>
  </default>
  <worldbody>
    <light pos="2.5 0 4" dir="0 0.3 -1" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" castshadow="true"/>
    <light pos="2.5 3 3" dir="0 -0.3 -1" diffuse="0.2 0.2 0.2" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="6 5 0.1" pos="2.5 1.5 0" material="floor"/>
    {chr(10).join(world)}
  </worldbody>
  <actuator>
    {chr(10).join(actuators)}
  </actuator>
</mujoco>"""
    return xml, info


def _decor_xml(info: SceneInfo) -> str:
    """不在规范里、只为了看着像厨房的东西：柜子背后的墙，和放微波炉、空气炸锅的台面。"""
    boxes = []
    for a in info.areas.values():
        if a.style == "surface":
            continue
        corners = [to_world(a, (sx * (a.geometry.size_m[0] / 2 + WALL), sy * (a.geometry.size_m[1] / 2 + WALL), 0)) for sx in (-1, 1) for sy in (-1, 1)]
        boxes.append((a, min(c[0] for c in corners), max(c[0] for c in corners), min(c[1] for c in corners), max(c[1] for c in corners)))
    if not boxes:
        return ""
    out = []
    back = max(b[4] for b in boxes) + 0.03
    x0, x1 = min(b[1] for b in boxes) - 0.3, max(b[2] for b in boxes) + 0.3
    out.append(_box(None, ((x0 + x1) / 2, back + 0.02, 1.15), ((x1 - x0) / 2, 0.02, 1.15), "0.93 0.91 0.87 1", 'class="visual"'))
    # 底面离地 0.3 到 1.2 米、下面又没有别的柜子的（微波炉、空气炸锅），给它们垫一个台面
    supported = []
    for a, ax0, ax1, ay0, ay1 in boxes:
        bottom = a.geometry.position[2] - WALL
        below = any(
            o is not a and o.geometry.position[2] < bottom and ox0 < ax1 and ax0 < ox1 and oy0 < ay1 and ay0 < oy1
            for o, ox0, ox1, oy0, oy1 in boxes
        )
        if 0.3 <= bottom <= 1.2 and not below:
            supported.append((bottom, ax0, ax1, ay0, ay1))
    if supported:
        top = min(b[0] for b in supported)
        cx0, cx1 = min(b[1] for b in supported) - 0.1, max(b[2] for b in supported) + 0.1
        cy0, cy1 = min(b[3] for b in supported) - 0.05, back
        out.append(_box(None, ((cx0 + cx1) / 2, (cy0 + cy1) / 2, top / 2), ((cx1 - cx0) / 2, (cy1 - cy0) / 2, top / 2), "0.74 0.66 0.56 1", 'class="visual"'))
    return "".join(out)


def _box(name: str | None, center, half, rgba: str, extra: str = "") -> str:
    n = f'name="{name}" ' if name else ""
    return f'<geom {n}type="box" pos="{_f(center)}" size="{_f(half)}" rgba="{rgba}" {extra}/>'


def _area_xml(home: Home, area: Area, info: SceneInfo, stuck, actuators: list[str]) -> str:
    g = area.geometry
    w, d, h = g.size_m
    t = WALL
    body = f"area_{area.id}"
    info.frame_body[area.id] = body
    if area.is_appliance:
        color = COLORS["appliance"]
    elif area.style == "surface":
        color = COLORS["intake" if home.storage[area.id].kind == "intake" else "surface"]
    else:
        color = COLORS[home.storage[area.id].temperature]
    parts: list[str] = []

    if area.style == "surface":
        if area.is_appliance:  # 灶台这类没有门的家电：一个实心台子，顶面就是放东西的地方
            parts.append(_box(f"shell_{area.id}", (0, 0, -g.position[2] / 2), (w / 2, d / 2, g.position[2] / 2), color))
        else:
            parts.append(_box(f"shell_{area.id}", (0, 0, -t / 2), (w / 2, d / 2, t / 2), color))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    leg = (sx * (w / 2 - 0.05), sy * (d / 2 - 0.05), -g.position[2] / 2)
                    parts.append(_box(None, leg, (0.025, 0.025, g.position[2] / 2 - t / 2), color, 'class="visual"'))
    else:
        bottom = -t / 2 - (t + 0.004 if area.style == "drawer" else 0)  # 抽屉托盘下面留出空隙
        parts += [
            _box(f"shell_{area.id}", (0, 0, bottom), (w / 2 + t, d / 2 + t, t / 2), color),
            _box(None, (0, 0, h + t / 2), (w / 2 + t, d / 2 + t, t / 2), color),
            _box(None, (-(w / 2 + t / 2), 0, h / 2), (t / 2, d / 2 + t, h / 2 + t), color),
            _box(None, (w / 2 + t / 2, 0, h / 2), (t / 2, d / 2 + t, h / 2 + t), color),
            _box(None, (0, d / 2 + t / 2, h / 2), (w / 2 + t, t / 2, h / 2 + t), color),
        ]
        if area.style == "cabinet":
            _, _, levels = area.layout
            ch = h / levels
            for k in range(1, levels):
                parts.append(_box(None, (0, 0, k * ch - t / 2), (w / 2, d / 2, t / 2), color))
        if area.is_appliance:
            parts.append(
                f'<geom name="led_{area.id}" type="sphere" size="0.015" pos="{_f((w / 2 - 0.03, -d / 2 - t - 0.006, h + t))}" '
                f'rgba="0.2 0.8 0.3 1" class="visual"/>'
            )
            if home.appliances[area.id].accepts_pour:
                info.pour_into.append(area.id)

    if area.fixture is not None:
        parts.append(_fixture_xml(area, info, stuck, actuators))

    pos = g.position
    return f'<body name="{body}" pos="{_f(pos)}" quat="{_yaw_quat(g.yaw_deg)}">{"".join(parts)}</body>'


def _fixture_xml(area: Area, info: SceneInfo, stuck, actuators: list[str]) -> str:
    f = area.fixture
    w, d, h = area.geometry.size_m
    t = WALL
    side = hinge_side(area)
    fid = f.id
    info.door_area[fid] = area.id
    friction = f'frictionloss="{STUCK_FRICTION}"' if fid in stuck else ""
    door_rgba = COLORS["appliance_door"] if area.is_appliance else COLORS["door"]
    hx, hy, hz = to_local(area, f.handle_pose.position)
    W, H = w + 2 * t, h + 2 * t
    front_y = -d / 2 - t - DOOR_GAP - t / 2  # 门板中心所在的 y

    moving = 'class="moving"'
    if side == "none":  # 抽屉：托盘 + 面板，沿 -y 拉出
        travel = area.drawer_travel
        info.door_open[fid] = travel
        info.frame_body[area.id] = f"fx_{fid}"
        wall_h = min(h * 0.45, 0.12)
        inner = [
            _box(None, (0, 0, -t / 2), (w / 2 - 0.004, d / 2, t / 2), door_rgba, moving),
            _box(None, (-(w / 2 - 0.004 - t / 2), 0, wall_h / 2), (t / 2, d / 2, wall_h / 2), door_rgba, moving),
            _box(None, (w / 2 - 0.004 - t / 2, 0, wall_h / 2), (t / 2, d / 2, wall_h / 2), door_rgba, moving),
            _box(None, (0, d / 2 - t / 2, wall_h / 2), (w / 2 - 0.004, t / 2, wall_h / 2), door_rgba, moving),
            _box(f"panel_{fid}", (0, front_y, h / 2), (W / 2, t / 2, H / 2), door_rgba, f'{moving} density="300"'),
            (
                f'<geom type="capsule" fromto="{_f((hx - 0.08, hy - 0.02, hz))} {_f((hx + 0.08, hy - 0.02, hz))}" '
                f'size="0.012" rgba="0.55 0.55 0.58 1" class="visual"/>'
            ),
            f'<site name="handle_{fid}" pos="{_f((hx, hy, hz))}" size="0.01"/>',
        ]
        if area.id in info.pour_into:
            inner.append(
                f'<geom name="food_{area.id}" type="box" pos="0 0 0.02" size="{_f((w / 2 - 0.03, d / 2 - 0.03, 0.02))}" '
                f'rgba="0.85 0.55 0.20 0" class="visual"/>'
            )
        actuators.append(
            f'<position name="a_{fid}" joint="j_{fid}" kp="600" forcerange="-{DRAWER_FORCE} {DRAWER_FORCE}" ctrlrange="0 {travel}"/>'
        )
        return (
            f'<body name="fx_{fid}">'
            f'<joint name="j_{fid}" type="slide" axis="0 -1 0" range="0 {travel}" damping="40" {friction}/>'
            f'{"".join(inner)}</body>'
        )

    angle = DOOR_OPEN_RAD[side]
    info.door_open[fid] = angle
    if side == "bottom":
        origin = (0, front_y, -t)
        axis = "1 0 0"
        panel_center = (0, 0, H / 2)
    else:
        sign = -1 if side == "left" else 1  # 左边的门轴在 -x
        origin = (sign * W / 2, front_y, -t)
        axis = "0 0 -1" if side == "left" else "0 0 1"  # 正方向都是往外开
        panel_center = (-sign * W / 2, 0, H / 2)
    handle = (hx - origin[0], hy - origin[1], hz - origin[2])
    actuators.append(
        f'<position name="a_{fid}" joint="j_{fid}" kp="80" forcerange="-{DOOR_FORCE} {DOOR_FORCE}" ctrlrange="0 {angle}"/>'
    )
    grip = (
        f'<geom type="capsule" fromto="{_f((handle[0], handle[1] - 0.02, handle[2] - 0.06))} '
        f'{_f((handle[0], handle[1] - 0.02, handle[2] + 0.06))}" size="0.012" rgba="0.55 0.55 0.58 1" class="visual"/>'
        if side != "bottom"
        else f'<geom type="capsule" fromto="{_f((handle[0] - 0.1, handle[1] - 0.02, handle[2]))} '
        f'{_f((handle[0] + 0.1, handle[1] - 0.02, handle[2]))}" size="0.012" rgba="0.55 0.55 0.58 1" class="visual"/>'
    )
    panel = _box(f"panel_{fid}", panel_center, (W / 2, t / 2, H / 2), door_rgba, f'{moving} density="250"')
    return (
        f'<body name="fx_{fid}" pos="{_f(origin)}">'
        f'<joint name="j_{fid}" type="hinge" axis="{axis}" range="0 {angle}" {friction}/>'
        f"{panel}{grip}"
        f'<site name="handle_{fid}" pos="{_f(handle)}" size="0.01"/>'
        f"</body>"
    )


def container_shape(c: Container) -> tuple[str, str]:
    """容器的 MuJoCo 形状：保鲜盒是长方体，罐子、杯子、碗是圆柱。返回 (type, size)。"""
    sx, sy, sz = c.size_m
    if c.kind in ("jar", "cup", "bowl"):
        return "cylinder", _f((min(sx, sy) / 2, sz / 2))
    return "box", _f((sx / 2, sy / 2, sz / 2))


def _container_xml(c: Container, area: Area, slot: int) -> str:
    x, y, z = area.slot_local(slot)
    pos = to_world(area, (x, y, z + c.size_m[2] / 2 + 0.002))
    shape, size = container_shape(c)
    mass = 0.5 if c.kind == "parcel" else 0.2 + 0.15 * c.capacity_l
    rgba = CONTAINER_COLORS.get(c.kind, CONTAINER_COLORS["box"])
    lid = ""
    if c.kind in ("box", "jar"):  # 盖子颜色深一点，看得出是保鲜盒
        top = c.size_m[2] / 2
        lid_size = _f((c.size_m[0] / 2 + 0.003, c.size_m[1] / 2 + 0.003, 0.008)) if shape == "box" else _f((min(c.size_m[:2]) / 2 + 0.003, 0.008))
        lid = f'<geom type="{shape}" pos="0 0 {top - 0.006:.4f}" size="{lid_size}" rgba="0.2 0.35 0.6 1" class="visual"/>'
    return (
        f'<body name="c_{c.id}" pos="{_f(pos)}" quat="{_yaw_quat(area.geometry.yaw_deg)}">'
        f'<freejoint name="fj_{c.id}"/><geom name="g_{c.id}" type="{shape}" size="{size}" rgba="{rgba}" mass="{mass:.3f}" class="item"/>'
        f"{lid}</body>"
    )


def _robot_xml(start: tuple[float, float, float]) -> str:
    x, y, yaw = start
    return f"""<body name="robot" mocap="true" pos="{x:.3f} {y:.3f} 0" quat="{_yaw_quat(yaw)}">
      <geom type="cylinder" size="0.22 0.05" pos="0 0 0.05" rgba="0.25 0.27 0.30 1" class="visual"/>
      <geom type="capsule" fromto="0 -0.09 0.1 0 -0.09 0.78" size="0.055" rgba="0.82 0.84 0.86 1" class="visual"/>
      <geom type="capsule" fromto="0 0.09 0.1 0 0.09 0.78" size="0.055" rgba="0.82 0.84 0.86 1" class="visual"/>
      <geom type="box" size="0.11 0.18 0.24" pos="0 0 1.04" rgba="0.90 0.91 0.92 1" class="visual"/>
      <geom type="sphere" size="0.11" pos="0 0 1.42" rgba="0.20 0.22 0.25 1" class="visual"/>
      <geom type="box" size="0.02 0.07 0.025" pos="0.1 0 1.44" rgba="0.30 0.75 0.95 1" class="visual"/>
      <site name="shoulder_l" pos="0 0.2 1.22" size="0.01"/>
      <site name="shoulder_r" pos="0 -0.2 1.22" size="0.01"/>
    </body>
    <body name="hand" mocap="true" pos="{x:.3f} {y:.3f} 0.9">
      <geom type="box" size="0.05 0.05 0.012" rgba="0.95 0.55 0.15 1" class="visual"/>
    </body>"""

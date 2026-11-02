"""仿真里的机器人身体：技能库调用它，它在 MuJoCo 里真的做一遍，再按物理结果告诉技能发生了什么。

和模拟实现的区别：
- 门卡住、东西不在、半路滑落这些故障不再由技能"按设定报错"，而是物理上真的发生：
  门轴摩擦太大拉不开、到了槽位手下面是空的、松手后容器掉到地上
- 放下东西后看它实际停在哪个槽位，没放稳会如实报告
- 拍照看的是仿真里实际在那儿的东西，再和台账对比

简化：底盘和手按轨迹移动，不做腿和手臂的动力学；机器人身体和手里拿着的东西不参与碰撞。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..skills.library import SkillFailed
from ..world import FLOOR, Photo, World
from .layout import DEFAULT_REACH, Area, Reach
from .scene import build_mjcf

STEP = 0.002
CARRY_LIFT = 0.03  # 手里拿着东西在柜子里平移时，离槽位底面多抬高一点
TOP_CLEARANCE = 0.22  # 从上面取放时，先停在上方多高


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


class SimBody:
    """一个家对应一个仿真。用 SimBody.attach(world) 接到世界上，之后技能会通过它执行。"""

    def __init__(
        self,
        world: World,
        reach: Reach = DEFAULT_REACH,
        recorder=None,
    ):
        home = world.home
        self.home = home
        self.reach = reach
        self.recorder = recorder
        self.waypoints = {wp: pose for r in home.rooms.values() for wp, pose in r.waypoints.items()}
        start = self.waypoints[world.robot.waypoint]
        self.base = np.array([start.position[0], start.position[1], math.radians(start.yaw_deg)])
        # 故障在这里变成物理条件，之后由物理去"发现"
        self.slippery = set(world.slippery)
        xml, info = build_mjcf(
            home,
            missing=set(world.missing_containers),
            stuck=set(world.stuck_fixtures),
            unknown=dict(world.unknown_objects),
            start=(self.base[0], self.base[1], start.yaw_deg),
        )
        self.xml, self.info = xml, info
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.held: str | None = None
        self.held_rel: tuple[np.ndarray, np.ndarray] | None = None
        self.caption = ""
        self.frame_count = 0
        centers = [a.geometry.position for a in info.areas.values()]
        self.scene_center = (
            (min(c[0] for c in centers) + max(c[0] for c in centers)) / 2,
            (min(c[1] for c in centers) + max(c[1] for c in centers)) / 2,
        )
        self._robot_mocap = self.model.body_mocapid[self.model.body("robot").id]
        self._hand_mocap = self.model.body_mocapid[self.model.body("hand").id]
        for f in home.fixtures.values():  # 规范里本来就开着的门
            if f.state == "open" and f.id in info.door_open:
                self._set_fixture(f.id, info.door_open[f.id])
        self._place_hand(self._rest_hand())
        mujoco.mj_forward(self.model, self.data)
        self._settle(0.6)

    @classmethod
    def attach(cls, world: World, reach: Reach = DEFAULT_REACH, recorder=None) -> SimBody:
        """接到世界上。注入的故障交给物理去体现，世界里的故障标记清掉，免得模拟实现替物理"报错"。"""
        body = cls(world, reach, recorder)
        world.missing_containers.clear()
        world.stuck_fixtures.clear()
        world.slippery.clear()
        world.unknown_objects.clear()
        world.body = body
        return body

    # ---------- 低层：步进、坐标 ----------

    def _step(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / STEP))):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            self.frame_count += 1
            if self.recorder is not None:
                self.recorder.tick(self)

    def _settle(self, seconds: float = 0.4) -> None:
        self._step(seconds)

    def _body_id(self, name: str) -> int:
        return self.model.body(name).id

    def _frame(self, area: Area) -> tuple[np.ndarray, np.ndarray]:
        b = self._body_id(self.info.frame_body[area.id])
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def _to_world(self, area: Area, local) -> np.ndarray:
        pos, rot = self._frame(area)
        return pos + rot @ np.asarray(local, dtype=float)

    def _to_local(self, area: Area, world_pos) -> np.ndarray:
        pos, rot = self._frame(area)
        return rot.T @ (np.asarray(world_pos, dtype=float) - pos)

    def _container_pos(self, cid: str) -> np.ndarray:
        return self.data.xpos[self._body_id(f"c_{cid}")].copy()

    def _forward(self) -> np.ndarray:
        return np.array([math.cos(self.base[2]), math.sin(self.base[2]), 0.0])

    def _rest_hand(self) -> np.ndarray:
        fwd = self._forward()
        left = np.array([-fwd[1], fwd[0], 0.0])
        return np.array([self.base[0], self.base[1], 0.0]) + 0.3 * fwd - 0.12 * left + np.array([0, 0, 0.95])

    def _hand(self) -> np.ndarray:
        return self.data.mocap_pos[self._hand_mocap].copy()

    def _place_hand(self, pos, quat=None) -> None:
        self.data.mocap_pos[self._hand_mocap] = pos
        self.data.mocap_quat[self._hand_mocap] = _yaw_quat(self.base[2]) if quat is None else quat

    def _place_base(self) -> None:
        self.data.mocap_pos[self._robot_mocap] = [self.base[0], self.base[1], 0]
        self.data.mocap_quat[self._robot_mocap] = _yaw_quat(self.base[2])

    def _hold(self) -> None:
        """手里吸住的容器跟着手走（磁吸板，看作刚性连接）。"""
        if self.held is None:
            return
        rel_pos, rel_quat = self.held_rel
        hand_pos = self.data.mocap_pos[self._hand_mocap]
        hand_quat = self.data.mocap_quat[self._hand_mocap]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, hand_quat)
        pos = hand_pos + rot.reshape(3, 3) @ rel_pos
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, hand_quat, rel_quat)
        j = self.model.joint(f"fj_{self.held}")
        adr, vadr = j.qposadr[0], j.dofadr[0]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qvel[vadr : vadr + 6] = 0

    def _set_collision(self, cid: str, on: bool) -> None:
        g = self.model.geom(f"g_{cid}").id
        self.model.geom_contype[g] = 5 if on else 0
        self.model.geom_conaffinity[g] = 5 if on else 0

    def _grab(self, cid: str) -> None:
        hand_pos, hand_quat = self._hand(), self.data.mocap_quat[self._hand_mocap].copy()
        b = self._body_id(f"c_{cid}")
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, hand_quat)
        rel_pos = rot.reshape(3, 3).T @ (self.data.xpos[b] - hand_pos)
        inv, rel_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, hand_quat)
        mujoco.mju_mulQuat(rel_quat, inv, self.data.xquat[b])
        self.held, self.held_rel = cid, (rel_pos, rel_quat)
        self._set_collision(cid, False)

    def _release(self) -> None:
        if self.held is None:
            return
        cid, self.held = self.held, None
        self._set_collision(cid, True)
        j = self.model.joint(f"fj_{cid}")
        self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0

    # ---------- 动作轨迹 ----------

    def _move_hand(self, target, duration: float = 0.6, quat=None) -> None:
        start = self._hand()
        start_q = self.data.mocap_quat[self._hand_mocap].copy()
        end_q = _yaw_quat(self.base[2]) if quat is None else np.asarray(quat, dtype=float)
        target = np.asarray(target, dtype=float)
        n = max(1, round(duration / STEP))
        for i in range(1, n + 1):
            u = _smooth(i / n)
            q = start_q + (end_q - start_q) * u
            q = q / np.linalg.norm(q)
            self._place_hand(start + (target - start) * u, q)
            self._step(STEP)

    def _check_reach(self, target, what: str) -> None:
        dist = math.hypot(target[0] - self.base[0], target[1] - self.base[1])
        if dist > self.reach.horizontal_m + 1e-6 or not self.reach.min_z <= target[2] <= self.reach.max_z:
            raise SkillFailed(f"手够不着{what}：水平 {dist:.2f} 米、高 {target[2]:.2f} 米（仿真）")

    def _approach_points(self, area: Area, slot_local, top_z: float) -> tuple[np.ndarray, np.ndarray]:
        """手伸进去之前停的点，和最终取放的点（都是手的位置）。top_z 是手要贴的高度（局部坐标）。"""
        x, y, _ = slot_local
        at = self._to_world(area, (x, y, top_z))
        if area.approach == "front":
            d = area.geometry.size_m[1]
            pre = self._to_world(area, (x, -d / 2 - 0.2, top_z + CARRY_LIFT))
        else:
            pre = at + np.array([0, 0, TOP_CLEARANCE])
        return pre, at

    # ---------- 感知 ----------

    def locate(self, cid: str) -> tuple[str, int]:
        """容器现在实际在哪个存放区的哪个槽位；哪儿都不在就算掉在地上。"""
        return self._locate_point(self._container_pos(cid))

    def _locate_point(self, pos) -> tuple[str, int]:
        for area in self.info.areas.values():
            local = self._to_local(area, pos)
            w, d, h = area.geometry.size_m
            top = h if area.style != "surface" else 0.6
            if abs(local[0]) <= w / 2 + 0.01 and abs(local[1]) <= d / 2 + 0.01 and -0.02 <= local[2] <= top:
                return area.id, self._nearest_slot(area, local)
        return FLOOR, 0

    @staticmethod
    def _nearest_slot(area: Area, local) -> int:
        _, _, ch = area.cell
        level = int(max(0, local[2] - 0.005) // ch) if area.style == "cabinet" else 0
        best, best_d = 0, 1e9
        for s in range(area.slots):
            x, y, z = area.slot_local(s)
            if area.style == "cabinet" and abs(z - level * ch) > 1e-6:
                continue
            dist = math.hypot(local[0] - x, local[1] - y)
            if dist < best_d:
                best, best_d = s, dist
        return best

    def _things_in(self, area_id: str) -> tuple[list[str], list[str]]:
        """存放区里实际有的容器，和台账里没有的东西。"""
        present = [cid for cid in self.info.containers if cid != self.held and self.locate(cid)[0] == area_id]
        unknown = [
            name
            for name in self.info.unknown.get(area_id, [])
            if self._locate_point(self.data.xpos[self._body_id(name)])[0] == area_id
        ]
        return present, unknown

    def _occupant(self, area: Area, slot: int) -> str | None:
        for cid in self.info.containers:
            if cid != self.held and self.locate(cid) == (area.id, slot):
                return cid
        for name in self.info.unknown.get(area.id, []):
            if self._locate_point(self.data.xpos[self._body_id(name)]) == (area.id, slot):
                return name
        return None

    # ---------- 技能 ----------

    def navigate(self, world: World, waypoint: str) -> None:
        pose = self.waypoints[waypoint]
        self.caption = f"走到 {waypoint}"
        start = self.base.copy()
        goal = np.array([pose.position[0], pose.position[1], math.radians(pose.yaw_deg)])
        dyaw = (goal[2] - start[2] + math.pi) % (2 * math.pi) - math.pi
        dist = math.hypot(*(goal[:2] - start[:2]))
        duration = max(0.6, dist / 0.8)
        n = max(1, round(duration / STEP))
        slip_at = n // 2 if self.held in self.slippery else None
        hand_offset = self._hand() - self._rest_hand()
        for i in range(1, n + 1):
            u = _smooth(i / n)
            self.base = np.array([*(start[:2] + (goal[:2] - start[:2]) * u), start[2] + dyaw * u])
            self._place_base()
            self._place_hand(self._rest_hand() + hand_offset)
            if slip_at is not None and i == slip_at:
                return self._drop(world, waypoint)
            self._step(STEP)
        self.base[2] = goal[2]

    def _drop(self, world: World, waypoint: str) -> None:
        cid = self.held
        self._release()
        self._settle(1.0)
        c = world.home.containers[cid]
        location, slot = self.locate(cid)
        c.location, c.slot = location, slot
        world.robot.holding = None
        where = "掉在地上" if location == FLOOR else f"掉进了 {location}"
        left = math.hypot(*(np.array(self.waypoints[waypoint].position[:2]) - self.base[:2]))
        raise SkillFailed(f"去 {waypoint} 的路上{c.name}从手里滑落（仿真：{where}，离 {waypoint} 还有 {left:.1f} 米）")

    def _joint_q(self, fid: str) -> float:
        return float(self.data.qpos[self.model.joint(f"j_{fid}").qposadr[0]])

    def _set_fixture(self, fid: str, q: float) -> None:
        j = self.model.joint(f"j_{fid}")
        self.data.qpos[j.qposadr[0]] = q
        self.data.ctrl[self.model.actuator(f"a_{fid}").id] = q

    def _handle(self, fid: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(f"handle_{fid}").id].copy()

    def _operate(self, world: World, fid: str, want_open: bool) -> None:
        f = world.home.fixtures[fid]
        if fid not in self.info.door_open:
            raise SkillFailed(f"仿真里没有{f.name}（它围着的存放区没有写三维尺寸）")
        self.caption = f"{'打开' if want_open else '关上'}{f.name}"
        area = self.info.areas[self.info.door_area[fid]]
        full = self.info.door_open[fid]
        handle = self._handle(fid)
        self._check_reach(handle, f"{f.name}的把手")
        outward = self._to_world(area, (0, -1, 0)) - self._to_world(area, (0, 0, 0))
        self._move_hand(handle + 0.12 * outward, 0.5)
        self._move_hand(handle, 0.3)
        act = self.model.actuator(f"a_{fid}").id
        start, goal = float(self.data.ctrl[act]), (full if want_open else 0.0)
        n = round((1.2 if area.style == "cabinet" else 1.4) / STEP)  # 抽屉拉得慢一点，里面的东西不乱滑
        for i in range(1, n + 1):
            self.data.ctrl[act] = start + (goal - start) * _smooth(i / n)
            self._place_hand(self._handle(fid))
            self._step(STEP)
        for _ in range(round(0.3 / STEP)):
            self._place_hand(self._handle(fid))
            self._step(STEP)
        q = self._joint_q(fid)
        ok = q >= 0.8 * full if want_open else q <= 0.1 * full
        if not ok:
            self.data.ctrl[act] = q  # 不再硬拉
        self._move_hand(self._handle(fid) + 0.12 * outward, 0.3)
        self._move_hand(self._rest_hand(), 0.5)
        if not ok:
            pct = round(100 * q / full)
            if want_open:
                raise SkillFailed(f"{f.name}拉不开，可能卡住了（仿真：只开到 {pct}%）")
            raise SkillFailed(f"{f.name}关不上，可能卡住了（仿真：还开着 {pct}%）")

    def open(self, world: World, fixture: str) -> None:
        self._operate(world, fixture, True)

    def close(self, world: World, fixture: str) -> None:
        self._operate(world, fixture, False)

    def _area(self, location: str) -> Area:
        if location not in self.info.areas:
            raise SkillFailed(f"仿真里没有 {location}（规范文件没写它的三维尺寸）")
        return self.info.areas[location]

    def pick(self, world: World, container: str) -> None:
        c = world.home.containers[container]
        self.caption = f"取{c.name}"
        area = self._area(c.location)
        slot = area.slot_local(c.slot)
        top = slot[2] + c.size_m[2] + 0.012
        pre, at = self._approach_points(area, slot, top)
        self._check_reach(at, f"{area.name}槽位 {c.slot}")
        self._move_hand(pre, 0.6)
        self._move_hand(at + np.array([0, 0, 0.03]), 0.4)
        occupant = self._occupant(area, c.slot)
        if occupant != container:
            self._move_hand(pre, 0.4)
            self._move_hand(self._rest_hand(), 0.5)
            if occupant is None:
                raise SkillFailed(f"到了 {c.location} 槽位 {c.slot} 没找到{c.name}，台账可能过期（仿真：槽位是空的）")
            what = world.home.containers[occupant].name if occupant in world.home.containers else "一个没登记的东西"
            raise SkillFailed(f"{c.location} 槽位 {c.slot} 里是{what}，不是{c.name}，台账可能过期")
        actual = self._container_pos(container)
        self._move_hand(actual + np.array([0, 0, c.size_m[2] / 2 + 0.012]), 0.3)
        self._grab(container)
        self._move_hand(self._hand() + np.array([0, 0, CARRY_LIFT]), 0.3)
        self._move_hand(pre, 0.5)
        self._move_hand(self._rest_hand(), 0.6)

    def place(self, world: World, container: str, location: str, slot: int) -> tuple[str, int]:
        c = world.home.containers[container]
        self.caption = f"把{c.name}放进{self._area(location).name}"
        area = self._area(location)
        local = area.slot_local(slot)
        hand_z = local[2] + c.size_m[2] + 0.012 + 0.01  # 松手时容器离底面 1 厘米
        pre, at = self._approach_points(area, local, hand_z)
        self._check_reach(at, f"{area.name}槽位 {slot}")
        self._move_hand(pre, 0.6)
        self._move_hand(at, 0.6)
        self._release()
        self._settle(0.5)
        self._move_hand(self._hand() + np.array([0, 0, 0.02]), 0.2)
        self._move_hand(pre, 0.4)
        self._move_hand(self._rest_hand(), 0.6)
        self._settle(0.3)
        return self.locate(container)

    def inspect(self, world: World, location: str) -> Photo:
        area = self._area(location)
        self.caption = f"拍{area.name}"
        self._settle(0.3)
        present, unknown = self._things_in(location)
        photo = world.look(location, present=present, unknown=["一个没登记的袋子"] * len(unknown))
        photo.image = self.snapshot(f"inspect_{location}", look_at=self._to_world(area, (0, 0, area.geometry.size_m[2] / 2)))
        return photo

    def pour(self, world: World, container: str, into: str) -> None:
        c = world.home.containers[container]
        area = self._area(into)
        self.caption = f"把{c.name}里的东西倒进{area.name}"
        above = self._to_world(area, (0, 0, area.geometry.size_m[2] + c.size_m[2] + 0.1))
        self._check_reach(above, f"{area.name}上方")
        self._move_hand(above, 0.7)
        fwd = self._forward()
        axis = np.array([-fwd[1], fwd[0], 0.0])  # 绕身体左右方向转，罐口朝前倒
        tilt, q = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(tilt, axis, 2.0)
        mujoco.mju_mulQuat(q, tilt, _yaw_quat(self.base[2]))
        self._move_hand(above, 0.8, quat=q)
        food = f"food_{into}"
        if food in [self.model.geom(i).name for i in range(self.model.ngeom)]:
            self.model.geom_rgba[self.model.geom(food).id][3] = 1.0
        self._step(0.4)
        self._move_hand(above, 0.6)
        self._move_hand(self._rest_hand(), 0.6)

    def appliance(self, world: World, appliance: str, running: bool) -> None:
        name = f"led_{appliance}"
        self.caption = f"{world.home.appliances[appliance].name}{'开始运行' if running else '停下'}"
        try:
            g = self.model.geom(name).id
        except KeyError:
            return
        self.model.geom_rgba[g] = [0.95, 0.2, 0.15, 1] if running else [0.2, 0.8, 0.3, 1]
        self._step(0.4)

    # ---------- 画面 ----------

    def snapshot(self, name: str, look_at=None) -> str | None:
        """拍一张图存下来（需要录制目录和能用的渲染）；不录制时返回 None。"""
        if self.recorder is None:
            return None
        if name == "scene":  # 录像里停一下，让看的人知道这里出了问题
            self.caption = "出问题了：停下，拍现场照"
            self.recorder.hold(self, 1.5)
        return self.recorder.photo(self, name, look_at)

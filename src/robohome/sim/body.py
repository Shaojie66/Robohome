"""仿真里的机器人身体：技能库调用它，它在 MuJoCo 里真的做一遍，再按物理结果告诉技能发生了什么。

和模拟实现的区别：
- 门卡住、东西不在、半路滑落这些故障不再由技能"按设定报错"，而是物理上真的发生：
  门轴摩擦太大拉不开、到了槽位手下面是空的、松手后容器掉到地上
- 走路按地面地图找路，边走边用激光扫一圈；看到地上多了东西就绕开，绕不过去就停下、退回出发的点位
- 取东西不直接读容器的真实位置：手上的相机看容器身上的标签，按标签算出的位置去吸；标签看不到就不伸手
- 放下东西后看它实际停在哪个槽位，没放稳会如实报告
- 拍照看的是仿真里实际在那儿的东西，再和台账对比

简化：底盘和手按轨迹移动，不做腿和手臂的动力学（全身平衡、灵巧手都不做）；机器人身体和手里拿着的东西不参与碰撞；
开门时门板扫过的地方不检查；拍照清点还是按仿真里的真实位置判断东西在不在。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..skills.library import SkillFailed
from ..spec.models import Pose
from ..world import FLOOR, Photo, World
from .floor import G1_FOOTPRINT, SEEN_POINT, FloorMap, Footprint, path_length
from .layout import DEFAULT_REACH, HAND_ABOVE_TOP, RELEASE_GAP, Area, Reach
from .scene import Obstacle, build_mjcf

STEP = 0.002
CARRY_LIFT = 0.03  # 手里拿着东西在柜子里平移时，离槽位底面多抬高一点
TOP_CLEARANCE = 0.22  # 从上面取放时，先停在上方多高
REST_HAND = (0.28, 0.12, 0.85)  # 不干活时手在身体前方多远、偏右多少、离地多高

WALK_SPEED = 0.8  # 米/秒
TURN_SPEED = 2.0  # 弧度/秒
SCAN_EVERY = 0.2  # 每走这么远用激光扫一圈
LIDAR_HEIGHTS = (0.08, 0.35)  # 两圈水平激光：贴着地面一圈、膝盖高一圈
LIDAR_RAYS = 180
LIDAR_RANGE = 3.0
KNOWN_GAP = 0.04  # 激光点离规范里写的家具不到这么远，就当看到的是家具本身
MAX_REPLANS = 20
BLOCK_BEHIND = 0.45  # 注入"路被挡住"时，纸箱摆在点位后方这么远（正好挡住走进点位的那条线）

TAG_RANGE = 0.8  # 手上的相机最远这么远能认出标签
TAG_MAX_ANGLE = 60.0  # 标签朝向和视线最多差这么多度
TAG_NOISE = 0.002  # 按标签算出的容器位置误差（米）
GRIP_TOL = 0.015  # 磁吸板和容器顶面中心水平偏差在这以内才吸得住


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def _blocking(pose: Pose) -> Obstacle:
    """"路被挡住"的故障：在点位正后方摆一个纸箱，机器人走不进这个点位。"""
    yaw = math.radians(pose.yaw_deg)
    x, y = pose.position[0] - BLOCK_BEHIND * math.cos(yaw), pose.position[1] - BLOCK_BEHIND * math.sin(yaw)
    return Obstacle(x, y, pose.yaw_deg + 90)


class SimBody:
    """一个家对应一个仿真。用 SimBody.attach(world) 接到世界上，之后技能会通过它执行。"""

    def __init__(
        self,
        world: World,
        reach: Reach = DEFAULT_REACH,
        recorder=None,
        footprint: Footprint = G1_FOOTPRINT,
        obstacles: list[Obstacle] | tuple[Obstacle, ...] = (),
        jitter_seed: int | None = 0,
    ):
        """obstacles 是额外摆在地上的东西；jitter_seed 是容器摆放偏差的随机种子，None 就摆在槽位正中间。"""
        home = world.home
        self.home = home
        self.reach = reach
        self.recorder = recorder
        self.footprint = footprint
        self.waypoints = {wp: pose for r in home.rooms.values() for wp, pose in r.waypoints.items()}
        start = self.waypoints[world.robot.waypoint]
        self.base = np.array([start.position[0], start.position[1], math.radians(start.yaw_deg)])
        # 故障在这里变成物理条件，之后由物理去"发现"
        self.slippery = set(world.slippery)
        obstacles = [*obstacles, *(_blocking(self.waypoints[wp]) for wp in sorted(world.blocked_waypoints) if wp in self.waypoints)]
        xml, info = build_mjcf(
            home,
            missing=set(world.missing_containers),
            stuck=set(world.stuck_fixtures),
            unknown=dict(world.unknown_objects),
            start=(self.base[0], self.base[1], start.yaw_deg),
            covered=set(world.covered_tags),
            obstacles=obstacles,
            jitter_seed=jitter_seed,
        )
        self.rng = np.random.default_rng(jitter_seed or 0)  # 看标签的误差
        self.trail: list[tuple[float, float]] = []  # 最近一次走路经过的拐点（测试和录像用）
        self.floor: FloorMap | None = None  # 最近一次走路用的地面地图（含激光看到的点）
        angles = np.linspace(0, 2 * math.pi, LIDAR_RAYS, endpoint=False)
        self._ray_dirs = np.stack([np.cos(angles), np.sin(angles), np.zeros(LIDAR_RAYS)], axis=1)
        self._ray_vec = self._ray_dirs.reshape(-1).copy()
        self._ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # 只看会碰撞的东西（组 0），不看装饰和机器人自己
        self._ray_geom = np.zeros(LIDAR_RAYS, dtype=np.int32)
        self._ray_dist = np.zeros(LIDAR_RAYS)
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
                self._open_at_start(home, f.id)
        self._place_hand(self._rest_hand())
        mujoco.mj_forward(self.model, self.data)
        self._settle(0.6)

    @classmethod
    def attach(cls, world: World, reach: Reach = DEFAULT_REACH, recorder=None, **kw) -> SimBody:
        """接到世界上。注入的故障交给物理去体现，世界里的故障标记清掉，免得模拟实现替物理"报错"。"""
        body = cls(world, reach, recorder, **kw)
        world.missing_containers.clear()
        world.stuck_fixtures.clear()
        world.slippery.clear()
        world.unknown_objects.clear()
        world.blocked_waypoints.clear()
        world.covered_tags.clear()
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
        ahead, right, up = REST_HAND
        return np.array([self.base[0], self.base[1], 0.0]) + ahead * fwd - right * left + np.array([0, 0, up])

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
        if float(np.dot(start_q, end_q)) < 0:  # 同一个朝向有两个符号相反的四元数，取近的那个插值
            end_q = -end_q
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
        if not self.reach.reaches(dist, target[2]):
            raise SkillFailed(f"手够不着{what}：水平 {dist:.2f} 米、高 {target[2]:.2f} 米（仿真：{self.reach.limit_text(target[2])}）")

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

    def _read_tag(self, cid: str) -> np.ndarray | None:
        """手上的相机看容器身上的标签：看到一个就按它算出容器中心在哪（带几毫米误差）；一个都看不到返回 None。

        看得到要同时满足：离相机不太远、标签大致朝着相机、视线没被别的东西挡住（激光一样用射线检查）。
        """
        if cid not in self.info.tags:
            return None
        camera = self._hand()
        b = self._body_id(f"c_{cid}")
        pos, rot = self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()
        own = self.model.geom(f"g_{cid}").id
        exclude = self._body_id(f"c_{self.held}") if self.held else -1
        gid = np.zeros(1, dtype=np.int32)
        for local, normal in self.info.tags[cid]:
            tag = pos + rot @ np.asarray(local)
            view = camera - tag
            dist = float(np.linalg.norm(view))
            if dist > TAG_RANGE or float(np.dot(rot @ np.asarray(normal), view / dist)) < math.cos(math.radians(TAG_MAX_ANGLE)):
                continue
            mujoco.mj_ray(self.model, self.data, camera, -view / dist, self._ray_groups, 1, exclude, gid)
            if gid[0] != own:  # 先打到了别的东西（盖着的布、前排的盒子）
                continue
            noise = self.rng.normal(0.0, TAG_NOISE, 2)
            return pos + np.array([noise[0], noise[1], 0.0])
        return None

    def _occupant(self, area: Area, slot: int) -> str | None:
        for cid in self.info.containers:
            if cid != self.held and self.locate(cid) == (area.id, slot):
                return cid
        for name in self.info.unknown.get(area.id, []):
            if self._locate_point(self.data.xpos[self._body_id(name)]) == (area.id, slot):
                return name
        return None

    # ---------- 技能 ----------

    def _open_fixtures(self) -> set[str]:
        return {fid for fid, full in self.info.door_open.items() if self._joint_q(fid) > 0.15 * full}

    def _scan(self, floor: FloorMap) -> int:
        """激光扫一圈：打到的点离规范里的家具都远，就是地上多出来的东西，加进地图。返回新加了几个点。"""
        pts = []
        exclude = self._body_id(f"c_{self.held}") if self.held else -1
        for h in LIDAR_HEIGHTS:
            origin = np.array([self.base[0], self.base[1], h])
            mujoco.mj_multiRay(
                self.model, self.data, origin, self._ray_vec, self._ray_groups, 1, exclude,
                self._ray_geom, self._ray_dist, None, LIDAR_RAYS, LIDAR_RANGE,
            )
            hit = self._ray_geom >= 0
            for d, v in zip(self._ray_dist[hit], self._ray_dirs[hit]):
                x, y = origin[0] + d * v[0], origin[1] + d * v[1]
                if floor.static_distance(x, y) > KNOWN_GAP:
                    pts.append((float(x), float(y)))
        return floor.add_seen(pts)

    def _hand_offset(self) -> np.ndarray:
        """手离休息位置的偏差，换成身体坐标，走路转身时手跟着身体走。"""
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        d = self._hand() - self._rest_hand()
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])

    def _place_body(self, offset: np.ndarray) -> None:
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        self._place_base()
        self._place_hand(self._rest_hand() + np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2]]))

    def _turn_to(self, yaw: float, offset: np.ndarray) -> None:
        d = (yaw - self.base[2] + math.pi) % (2 * math.pi) - math.pi
        y0 = self.base[2]
        n = math.ceil(abs(d) / (TURN_SPEED * STEP))
        for i in range(1, n + 1):
            self.base[2] = y0 + d * _smooth(i / n)
            self._place_body(offset)
            self._step(STEP)

    def _segment(self, a, b, offset, tight: bool):
        """沿直线从 a 走到 b，一小步一小步地给出走过的距离。

        tight：起点离东西近、转不开身。这时要往身后走就不转身，直着倒退出来（从点位退出来就是这样）。
        """
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-4:
            return
        backwards = tight and float(np.dot(self._forward()[:2], seg / length)) < -0.3
        if not backwards:
            self._turn_to(math.atan2(seg[1], seg[0]), offset)
        n = max(1, math.ceil(length / (WALK_SPEED * STEP)))
        for i in range(1, n + 1):
            self.base[:2] = a + seg * (i / n)
            self._place_body(offset)
            self._step(STEP)
            yield length / n

    def navigate(self, world: World, waypoint: str) -> None:
        pose = self.waypoints[waypoint]
        self.caption = f"走到 {waypoint}"
        goal = (pose.position[0], pose.position[1], math.radians(pose.yaw_deg))
        start_wp = world.robot.waypoint
        floor = self.floor = FloorMap.of(world.home, open_fixtures=self._open_fixtures())
        self._scan(floor)
        offset = self._hand_offset()
        corners = [(float(self.base[0]), float(self.base[1]))]
        self.trail = corners
        walked, slip_after = 0.0, None
        for _ in range(MAX_REPLANS):
            pts, why = floor.route(tuple(self.base), goal, self.footprint)
            if pts is None:
                # 只按规范里的家具找得到路，说明是地上多出来的东西挡住了
                known = FloorMap.of(world.home, open_fixtures=self._open_fixtures())
                seen = bool(floor.seen) and known.route(tuple(self.base), goal, self.footprint)[0] is not None
                self._give_up(floor, waypoint, start_wp, why, corners, offset, seen)
            if slip_after is None:
                slip_after = path_length(pts) / 2 if self.held in self.slippery else math.inf
            since, replan = 0.0, False
            tight = floor.distance(*pts[0]) <= self.footprint.radius
            for k in range(len(pts) - 1):
                for step in self._segment(pts[k], pts[k + 1], offset, tight=tight and k == 0):
                    walked += step
                    since += step
                    if walked >= slip_after:
                        return self._drop(world, waypoint)
                    if since >= SCAN_EVERY:
                        since = 0.0
                        here = (float(self.base[0]), float(self.base[1]))
                        if self._scan(floor) and floor.seen_blocks([here, *pts[k + 1 :]], self.footprint.radius):
                            replan = True
                            break
                if replan:
                    break
                corners.append(pts[k + 1])
            if not replan:
                self._turn_to(goal[2], offset)
                self.base[2] = goal[2]
                return
            corners.append((float(self.base[0]), float(self.base[1])))
        self._give_up(floor, waypoint, start_wp, f"绕了 {MAX_REPLANS} 次还是过不去", corners, offset, True)

    def _give_up(
        self, floor: FloorMap, waypoint: str, start_wp: str, why: str, corners: list, offset: np.ndarray, seen: bool
    ) -> None:
        """路被挡住：停下，沿走过来的路退回出发的点位（台账里机器人还在那儿），再报告。"""
        here = (float(self.base[0]), float(self.base[1]))
        moved = len(corners) > 1 or math.hypot(here[0] - corners[0][0], here[1] - corners[0][1]) > 0.02
        if moved:
            back = [here, *reversed(corners)]
            tight = floor.distance(*here) <= self.footprint.radius
            for k in range(len(back) - 1):
                for _ in self._segment(back[k], back[k + 1], offset, tight=tight and k == 0):
                    pass
            start = self.waypoints[start_wp]
            self._turn_to(math.radians(start.yaw_deg), offset)
        where = f"退回 {start_wp}" if moved else f"停在 {start_wp}"
        if seen:
            head = why.split("：")[0]
            detail = {"走不到点位跟前": "点位跟前的地上有东西", "点位上站不下": "点位上有东西"}.get(head, f"地上有东西，{why}")
            raise SkillFailed(f"去 {waypoint} 的路被挡住了，{where}（仿真：激光看到{detail}）")
        raise SkillFailed(f"走不到 {waypoint}，{where}（仿真：{why}，规范里的点位或家具位置可能不对）")

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

    def _open_at_start(self, home, fid: str) -> None:
        """一开始就开着的门或抽屉。抽屉里的东西是按关着的位置摆的，要跟着抽屉一起挪出来。"""
        inside = [a for a, b in self.info.frame_body.items() if b == f"fx_{fid}"]
        mujoco.mj_forward(self.model, self.data)
        before = {a: self._frame(self.info.areas[a])[0] for a in inside}
        self._set_fixture(fid, self.info.door_open[fid])
        mujoco.mj_forward(self.model, self.data)
        for a in inside:
            shift = self._frame(self.info.areas[a])[0] - before[a]
            names = [f"c_{c}" for c in self.info.containers if home.containers[c].location == a]
            for name in names + self.info.unknown.get(a, []):
                adr = self.model.jnt_qposadr[self.model.body_jntadr[self._body_id(name)]]
                self.data.qpos[adr : adr + 3] += shift

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
        top = slot[2] + c.size_m[2] + HAND_ABOVE_TOP
        pre, at = self._approach_points(area, slot, top)
        self._check_reach(at, f"{area.name}槽位 {c.slot}")
        self._move_hand(pre, 0.6)
        occupant = self._occupant(area, c.slot)  # 有没有东西相机看得出来，是哪个容器要看标签
        seen = self._read_tag(occupant) if occupant is not None else None
        if occupant != container or seen is None:
            self._move_hand(self._rest_hand(), 0.5)
            if occupant is None:
                raise SkillFailed(f"到了 {c.location} 槽位 {c.slot} 没找到{c.name}，台账可能过期（仿真：槽位是空的）")
            if occupant not in world.home.containers:
                raise SkillFailed(f"{c.location} 槽位 {c.slot} 里是一个没登记的东西，不是{c.name}，台账可能过期")
            if seen is None:
                raise SkillFailed(
                    f"{area.name}槽位 {c.slot} 里有东西，但看不清上面的标签（可能被挡住了），没有伸手（仿真：手上的相机一个标签都没看到）"
                )
            what = world.home.containers[occupant].name
            raise SkillFailed(f"{c.location} 槽位 {c.slot} 里是{what}，不是{c.name}，台账可能过期")
        grip = seen + np.array([0, 0, c.size_m[2] / 2 + HAND_ABOVE_TOP])  # 按标签算出的位置去吸，不读真实位置
        self._move_hand(grip + np.array([0, 0, 0.03]), 0.4)
        self._move_hand(grip, 0.3)
        miss = math.hypot(*(self._container_pos(container)[:2] - grip[:2]))
        if miss > GRIP_TOL:
            self._move_hand(pre, 0.4)
            self._move_hand(self._rest_hand(), 0.5)
            raise SkillFailed(f"没对准{c.name}，没有吸住（仿真：手偏了 {miss * 100:.1f} 厘米）")
        self._grab(container)
        self._move_hand(self._hand() + np.array([0, 0, CARRY_LIFT]), 0.3)
        self._move_hand(pre, 0.5)
        self._move_hand(self._rest_hand(), 0.6)

    def place(self, world: World, container: str, location: str, slot: int) -> tuple[str, int]:
        c = world.home.containers[container]
        self.caption = f"把{c.name}放进{self._area(location).name}"
        area = self._area(location)
        local = area.slot_local(slot)
        hand_z = local[2] + c.size_m[2] + HAND_ABOVE_TOP + RELEASE_GAP
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

    def stow(self, world: World, parcel: str, container: str) -> None:
        """把手里的包装放进容器：对准容器顶上松手，看它是不是真的落在容器上。

        末端执行器还没定（见方案文档），开盖、把东西倒进盒子、合盖这几下先简化成"落在容器上就算装进去"，
        装好后包装从场景里收走。落偏了、掉下去了，就如实报告，台账不记。
        """
        p, c = world.home.containers[parcel], world.home.containers[container]
        area = self._area(c.location)
        self.caption = f"把{p.label or p.name}装进{c.name}"
        if container not in self.info.containers or self.locate(container)[0] != c.location:
            raise SkillFailed(f"{area.name}里没找到{c.name}，台账可能过期（仿真：容器不在那里），包装还拿在手里")
        above = c.size_m[2] + p.size_m[2] + HAND_ABOVE_TOP + RELEASE_GAP
        slot = area.slot_local(c.slot)
        look, at = self._approach_points(area, slot, slot[2] + above)  # 先按台账的槽位停到跟前，看标签
        self._check_reach(at, f"{area.name}里的{c.name}")
        self._move_hand(look, 0.6)
        seen = self._read_tag(container)
        if seen is None:
            self._move_hand(self._rest_hand(), 0.5)
            raise SkillFailed(f"看不清{c.name}上的标签（可能被挡住了），没有装，包装还拿在手里（仿真：手上的相机一个标签都没看到）")
        x, y, z = self._to_local(area, seen)  # 按标签算出的容器位置对准
        bottom = z - c.size_m[2] / 2
        pre, at = self._approach_points(area, (x, y, bottom), bottom + above)
        self._check_reach(at, f"{area.name}里的{c.name}")
        self._move_hand(pre, 0.3)
        self._move_hand(at, 0.6)
        self._release()
        self._settle(0.5)
        landed = self._container_pos(parcel)
        rel = self.data.xmat[self._body_id(f"c_{container}")].reshape(3, 3).T @ (landed - self._container_pos(container))
        on_top = (
            abs(rel[0]) <= c.size_m[0] / 2 + 0.02
            and abs(rel[1]) <= c.size_m[1] / 2 + 0.02
            and rel[2] >= c.size_m[2] / 2 - 0.01
        )
        self._move_hand(self._hand() + np.array([0, 0, 0.02]), 0.2)
        self._move_hand(pre, 0.4)
        self._move_hand(self._rest_hand(), 0.6)
        if not on_top:
            self._settle(0.5)
            location, slot = self.locate(parcel)
            p.location, p.slot = location, slot
            world.robot.holding = None
            where = "掉在地上" if location == FLOOR else f"落在 {location} 槽位 {slot}"
            raise SkillFailed(f"{p.name}没放进{c.name}（仿真：{where}）")
        self._absorb(parcel)
        self._settle(0.3)

    def _absorb(self, parcel: str) -> None:
        """装好了：包装从场景里收走（不再碰撞、看不见、不受重力），之后也不再算作场景里的容器。"""
        g = self.model.geom(f"g_{parcel}").id
        self._set_collision(parcel, False)
        self.model.geom_rgba[g][3] = 0.0
        self.model.body_gravcomp[self._body_id(f"c_{parcel}")] = 1.0
        j = self.model.joint(f"fj_{parcel}")
        self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0
        self.info.containers.remove(parcel)

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

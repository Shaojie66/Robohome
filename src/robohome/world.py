"""运行时世界状态：家（规范数据）+ 机器人自身状态 + 家电连接。

技能和调度器只读写这里。第一阶段所有改动都只发生在内存里；
换成仿真或真机时，技能实现会同时驱动真实执行器，再把结果写回这里。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from .appliances.base import ApplianceAdapter, MockAppliance
from .spec.models import Home

log = logging.getLogger(__name__)


@dataclass
class RobotState:
    room: str
    waypoint: str
    holding: str | None = None  # 手里拿着的容器 id


FLOOR = "floor"  # 搬运途中掉下去的容器的位置


@dataclass
class Photo:
    """模拟相机拍的一张照片：看到了什么，以及和台账对不上的地方。仿真和真机阶段换成真实图像。"""

    waypoint: str
    location: str | None  # 拍的是哪个存放区或家电；失败时随手拍的现场照为 None
    seen: list[str]
    notes: list[str] = field(default_factory=list)
    title: str = ""  # 给人看的名字，如"冷冻室"
    image: str | None = None  # 仿真或真机拍下的图片文件

    def describe(self) -> str:
        where = self.title or self.location or self.waypoint
        text = f"{where}：" + ("、".join(self.seen) if self.seen else "空的")
        if self.notes:
            text += "；" + "；".join(self.notes)
        return text


@dataclass
class World:
    home: Home
    robot: RobotState
    appliances: dict[str, ApplianceAdapter] = field(default_factory=dict)
    # 模拟"台账和现实不一致"：这些容器台账里有，实际去拿时找不到
    missing_containers: set[str] = field(default_factory=set)
    # 机器人自己打开、还没关上的门；离开前和任务结束时都要关好
    opened_by_robot: set[str] = field(default_factory=set)
    # 以下几种也是故障注入用：门卡住、搬运途中会滑落的容器、台账里没有的东西
    stuck_fixtures: set[str] = field(default_factory=set)
    slippery: set[str] = field(default_factory=set)
    unknown_objects: dict[str, list[str]] = field(default_factory=dict)
    photos: list[Photo] = field(default_factory=list)  # 这次任务拍的照片
    notices: list[str] = field(default_factory=list)  # 执行中发现、需要告诉主人的事
    started_by_robot: set[str] = field(default_factory=set)  # 机器人自己启动、还在运行的家电
    instruction: str = ""  # 正在执行的主人指令，安全规则用它判断主人有没有明确要求某件事
    # 执行动作的身体：None 时只改内存里的状态（模拟）；换成仿真或真机后，技能会先让它真的做一遍，
    # 再按它看到的结果更新这里。预演用的副本不带身体
    body: object | None = None

    @classmethod
    def from_home(cls, home: Home, start_room: str | None = None, start_waypoint: str | None = None) -> World:
        home = copy.deepcopy(home)
        room = start_room or next(iter(home.rooms))
        wp = start_waypoint or next(iter(home.rooms[room].waypoints))
        adapters: dict[str, ApplianceAdapter] = {
            a.id: MockAppliance(appliance_id=a.id, capabilities=list(a.capabilities))
            for a in home.appliances.values()
        }
        world = cls(home=home, robot=RobotState(room=room, waypoint=wp), appliances=adapters)
        world.sync_appliance_doors()
        return world

    def sync_appliance_doors(self) -> None:
        for a in self.home.appliances.values():
            adapter = self.appliances.get(a.id)
            if a.door and isinstance(adapter, MockAppliance):
                adapter.door_closed = self.home.fixtures[a.door].state == "closed"

    def appliance_running(self, appliance_id: str) -> bool | None:
        """True 在运行，False 没在运行，None 读不到状态（安全规则会按"可能在运行"处理）。"""
        try:
            return bool(self.appliances[appliance_id].status().get("running"))
        except Exception as e:
            log.warning("读不到 %s 的运行状态：%s", appliance_id, e)
            return None

    def container_at(self, location: str, slot: int) -> str | None:
        for c in self.home.containers.values():
            if c.location == location and c.slot == slot and c.id != self.robot.holding:
                return c.id
        return None

    def free_slot(self, location: str, prefer: int | None = None) -> int | None:
        """存放区或家电里第一个空槽位；prefer 空着就用 prefer。"""
        if location in self.home.storage:
            n = self.home.storage[location].slots
        elif location in self.home.appliances:
            n = self.home.appliances[location].slots
        else:
            return None
        if prefer is not None and 0 <= prefer < n and self.container_at(location, prefer) is None:
            return prefer
        return next((i for i in range(n) if self.container_at(location, i) is None), None)

    def _contents(self, container_id: str) -> list[str]:
        return [i.name for i in self.home.items.values() if i.container == container_id]

    def _label(self, container_id: str) -> str:
        c = self.home.containers[container_id]
        contents = self._contents(container_id)
        return f"{c.name}（{'、'.join(contents)}）" if contents else c.name

    def look(self, location: str, present: list[str] | None = None, unknown: list[str] | None = None) -> Photo:
        """拍一张存放区或家电内部的照片，并和台账对一遍。

        present 是实际看到的容器（仿真或真机给出）；不给就按台账"看到"，再把注入的故障体现出来。
        unknown 是看到的、台账里没有的东西。
        """
        home = self.home
        seen, notes = [], []
        ledger = [c.id for c in sorted(home.containers.values(), key=lambda c: c.slot) if c.location == location]
        if present is None:
            present = [c for c in ledger if c not in self.missing_containers]
        for cid in ledger:
            if cid not in present:
                notes.append(f"台账里有{home.containers[cid].name}，照片里没看到")
        for cid in present:
            c = home.containers[cid]
            seen.append(self._label(cid))
            if cid not in ledger:
                notes.append(f"看到了{c.name}，台账记的却是 {c.location}")
            area = home.storage.get(location)
            if area and area.kind == "surface" and c.home is None:
                notes.append(f"{c.name}没有登记原位，不知道该放回哪里")
        seen += [i.name for i in home.items.values() if i.container == location]  # 倒进家电里的食物
        unknown = self.unknown_objects.get(location, []) if unknown is None else unknown
        if unknown:
            notes.append(f"有台账里没有的东西：{'、'.join(unknown)}")
        area = home.storage.get(location) or home.appliances.get(location)
        return Photo(self.robot.waypoint, location, seen, notes, title=area.name if area else location)

    def scene_photo(self) -> Photo:
        """出问题时在当前位置拍一张现场照：手里拿着什么、这里的门开着没、地上有没有掉落的东西。"""
        home = self.home
        here = self.robot.waypoint
        seen = []
        if self.robot.holding:
            seen.append(f"手里拿着{self._label(self.robot.holding)}")
        for f in home.fixtures.values():
            if f.waypoint == here:
                seen.append(f"{f.name}{'开着' if f.state == 'open' else '关着'}")
        seen += [f"地上有{self._label(c.id)}" for c in home.containers.values() if c.location == FLOOR]
        image = self.body.snapshot("scene") if self.body is not None else None
        return Photo(here, None, seen, title=f"现场（{here}）", image=image)

    def sandbox(self) -> World:
        """复制一份用来预演计划的世界：家和机器人状态照搬，家电换成同状态的模拟家电。

        预演只知道台账里写的东西，所以不带 missing_containers；家电在线与否照搬已知状态。
        """
        adapters: dict[str, ApplianceAdapter] = {}
        for a in self.home.appliances.values():
            status, online = {}, True
            try:
                status = self.appliances[a.id].status()
                online = status.get("online", True) is not False
            except Exception as e:
                # 读不到状态就按"离线"预演，和真实执行时会遇到的情况一致
                log.warning("预演时读不到 %s 的状态：%s", a.id, e)
                online = False
            adapters[a.id] = MockAppliance(
                appliance_id=a.id,
                capabilities=list(a.capabilities),
                online=online,
                running=bool(status.get("running")),
                mode=status.get("mode"),
            )
        # 门卡住、东西会滑落这类故障预演时也不知道，和 missing_containers 一样不带过去
        world = World(
            home=copy.deepcopy(self.home),
            robot=copy.deepcopy(self.robot),
            appliances=adapters,
            opened_by_robot=set(self.opened_by_robot),
            started_by_robot=set(self.started_by_robot),
            instruction=self.instruction,
        )
        world.sync_appliance_doors()
        return world

    def snapshot(self) -> dict:
        """给日志和回报用的简要状态。"""
        return {
            "robot": f"{self.robot.room}/{self.robot.waypoint}",
            "holding": self.robot.holding,
            "open_fixtures": [f.id for f in self.home.fixtures.values() if f.state == "open"],
            "opened_by_robot": sorted(self.opened_by_robot),
            "appliances": {k: self._status_or_error(v) for k, v in self.appliances.items()},
        }

    @staticmethod
    def _status_or_error(adapter: ApplianceAdapter) -> dict:
        try:
            return adapter.status()
        except Exception as e:
            return {"error": f"读不到状态：{e}"}

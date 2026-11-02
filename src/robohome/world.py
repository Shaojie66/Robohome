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


@dataclass
class World:
    home: Home
    robot: RobotState
    appliances: dict[str, ApplianceAdapter] = field(default_factory=dict)
    # 模拟"台账和现实不一致"：这些容器台账里有，实际去拿时找不到
    missing_containers: set[str] = field(default_factory=set)
    # 机器人自己打开、还没关上的门；离开前和任务结束时都要关好
    opened_by_robot: set[str] = field(default_factory=set)

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
        world = World(
            home=copy.deepcopy(self.home),
            robot=copy.deepcopy(self.robot),
            appliances=adapters,
            opened_by_robot=set(self.opened_by_robot),
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

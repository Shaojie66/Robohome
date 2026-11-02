"""运行时世界状态：家（规范数据）+ 机器人自身状态 + 家电连接。

技能和调度器只读写这里。第一阶段所有改动都只发生在内存里；
换成仿真或真机时，技能实现会同时驱动真实执行器，再把结果写回这里。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .appliances.base import ApplianceAdapter, MockAppliance
from .spec.models import Home


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

    def appliance_running(self, appliance_id: str) -> bool:
        try:
            return bool(self.appliances[appliance_id].status().get("running"))
        except Exception:
            return False

    def container_at(self, location: str, slot: int) -> str | None:
        for c in self.home.containers.values():
            if c.location == location and c.slot == slot and c.id != self.robot.holding:
                return c.id
        return None

    def snapshot(self) -> dict:
        """给日志和回报用的简要状态。"""
        return {
            "robot": f"{self.robot.room}/{self.robot.waypoint}",
            "holding": self.robot.holding,
            "open_fixtures": [f.id for f in self.home.fixtures.values() if f.state == "open"],
            "appliances": {k: v.status() for k, v in self.appliances.items()},
        }

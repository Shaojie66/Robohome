"""家用规范数据层：把一个家"封装"成机器人能直接读的结构。

一个家由五类东西组成：
- Room       房间，以及机器人可以停靠的点位（waypoint）
- Fixture    可操作部件：柜门、抽屉、冰箱门。机器人只会开和关
- Container  标准容器：尺寸已知、位置已知，物品都装在容器里
- Item       物品台账：每件东西在哪个容器里
- Appliance  联网家电：机器人只调用它的接口，内部工作由家电自己完成

坐标统一用米，原点和朝向在建家时确定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Vec3 = tuple[float, float, float]


@dataclass
class Pose:
    """位置（米）和绕竖直轴的朝向（度）。"""

    position: Vec3
    yaw_deg: float = 0.0


@dataclass
class Room:
    id: str
    name: str
    waypoints: dict[str, Pose] = field(default_factory=dict)
    forbidden_waypoints: list[str] = field(default_factory=list)  # 禁区点位：机器人不能去


@dataclass
class Fixture:
    id: str
    name: str
    kind: Literal["door", "drawer", "lid"]
    room: str
    waypoint: str  # 操作这个部件时机器人要站的点位
    handle_pose: Pose
    state: Literal["open", "closed"] = "closed"
    # 部件关上后围起来的存放区，例如冷冻室，容器放在这里
    encloses: str | None = None


@dataclass
class Container:
    id: str
    name: str
    size_m: Vec3  # 长宽高
    capacity_l: float
    location: str  # 存放区 id 或家电 id
    slot: int = 0
    pose: Pose | None = None
    microwave_safe: bool = False


@dataclass
class Item:
    id: str
    name: str
    category: str
    container: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class Appliance:
    id: str
    name: str
    kind: str  # microwave / oven / fridge / dishwasher ...
    room: str
    waypoint: str
    door: str | None = None  # 家电门对应的 Fixture id
    capabilities: list[str] = field(default_factory=list)
    hazards: list[Literal["heat", "blade"]] = field(default_factory=list)
    endpoint: str = "mock://"  # 真实接入时换成 Home Assistant / 米家地址


@dataclass
class StorageArea:
    """存放区：冷冻室、冷藏室、某层橱柜。"""

    id: str
    name: str
    room: str
    slots: int
    temperature: Literal["ambient", "chilled", "frozen"] = "ambient"


@dataclass
class Home:
    name: str
    rooms: dict[str, Room]
    fixtures: dict[str, Fixture]
    storage: dict[str, StorageArea]
    containers: dict[str, Container]
    items: dict[str, Item]
    appliances: dict[str, Appliance]

    def find_item(self, text: str) -> Item | None:
        """按名字或别名在台账里找物品，用于把一句话里的物品对上号。"""
        for item in self.items.values():
            for name in [item.name, *item.aliases]:
                if name and name in text:
                    return item
        return None

    def storage_of(self, container_id: str) -> StorageArea | None:
        loc = self.containers[container_id].location
        return self.storage.get(loc)

    def fixture_enclosing(self, location_id: str) -> Fixture | None:
        for f in self.fixtures.values():
            if f.encloses == location_id:
                return f
        return None

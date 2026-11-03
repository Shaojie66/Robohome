"""家用规范数据层：把一个家"封装"成机器人能直接读的结构。

一个家由五类东西组成：
- Room       房间，以及机器人可以停靠的点位（waypoint）
- Fixture    可操作部件：柜门、抽屉、冰箱门。机器人只会开和关
- Container  标准容器：尺寸已知、位置已知，物品都装在容器里；碗、杯子这类要归位的东西也按容器登记；
             刚买回来、摆在入库台上还没装盒的包装也先按容器登记（kind="parcel"），标签上写着是什么
- Item       物品台账：每件东西在哪个容器里（倒进家电后就记在家电里）
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
class Geometry:
    """存放区或家电内部空间的三维尺寸，仿真用它搭模型，也用它检查槽位放不放得下、手够不够得着。

    position 是内部底面的中心；size_m 是内部的宽（x）、深（y）、高（z）；开口朝向由 yaw_deg 决定，
    0 度表示开口朝 -y（机器人站在 y 更小的一侧）。layout 是槽位排布：几列（x）、几行（y，0 行在最外面）、几层（z）。
    """

    position: Vec3
    size_m: Vec3
    yaw_deg: float = 0.0
    layout: tuple[int, int, int] | None = None


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
    # 门轴在哪一边（从外面看）：left / right / bottom（洗碗机那种往下翻的门）。不写就放在离把手远的一边；抽屉不用写
    hinge: Literal["left", "right", "bottom"] | None = None


FOOD_KINDS = ("box", "jar")  # 能装食物、入库时可以分配的容器种类
PARCEL = "parcel"  # 刚买回来、还没装盒的包装
HELD, FLOOR = "robot", "floor"  # 容器拿在机器人手里、掉在地上时的位置：只在运行时出现，规范文件里不该有
RUNTIME_LOCATIONS = (HELD, FLOOR)


@dataclass
class Container:
    id: str
    name: str
    size_m: Vec3  # 长宽高
    capacity_l: float = 0.0  # 能装多少升；碗杯和包装可以不写
    location: str = ""  # 存放区 id 或家电 id
    slot: int = 0
    pose: Pose | None = None
    kind: str = "box"  # box 保鲜盒 / jar 密封罐 / bowl 碗 / cup 杯子 / object 其他常用物品 / parcel 待入库的包装
    safe_for: list[str] = field(default_factory=list)  # 可以放进哪些加热家电（家电 kind），如 ["microwave"]
    home: str | None = None  # 原位：用完要放回的存放区
    home_slot: int | None = None
    label: str = ""  # 包装上的标签（扫描读到的文字）；空着表示看不清

    @property
    def microwave_safe(self) -> bool:
        return "microwave" in self.safe_for


@dataclass
class Item:
    id: str
    name: str
    category: str
    container: str  # 容器 id；倒进家电（如空气炸锅）后是家电 id
    aliases: list[str] = field(default_factory=list)
    volume_l: float = 0.0  # 占用容器的体积，入库分配容器时用
    # 怎么加热：家电动作 → 参数，例如 {"air_fry": {"minutes": 12, "temp_c": 180}}，相当于包装上的做法说明
    cook: dict[str, dict] = field(default_factory=dict)


@dataclass
class Appliance:
    id: str
    name: str
    kind: str  # microwave / oven / fridge / dishwasher ...
    room: str
    waypoint: str
    door: str | None = None  # 家电门对应的 Fixture id
    capabilities: list[str] = field(default_factory=list)
    hazards: list[Literal["heat", "blade", "hot_water"]] = field(default_factory=list)
    endpoint: str = "mock://"  # 真实接入时换成 Home Assistant / 米家地址
    max_minutes: int = 30  # 危险动作单次最长运行分钟数，超过就拒绝
    max_temp_c: int | None = None  # 能设温度的家电（空气炸锅、烤箱）允许的最高温度
    slots: int = 1  # 里面能放几个容器，例如洗碗机的碗位
    accepts_pour: bool = False  # 能不能把食物直接倒进去（空气炸锅的炸篮）
    geometry: Geometry | None = None  # 内部空间的三维尺寸（仿真用）


@dataclass
class StorageArea:
    """存放区：冷冻室、冷藏室、某层橱柜；也可以是餐桌这样的台面（kind="surface"），东西只是临时放在那里；
    或者是入库台（kind="intake"）：主人把买回来的东西摆在这里，机器人逐件扫描、装盒、放进对应温区。"""

    id: str
    name: str
    room: str
    slots: int
    temperature: Literal["ambient", "chilled", "frozen"] = "ambient"
    kind: Literal["storage", "surface", "intake"] = "storage"
    waypoint: str | None = None  # 没有门的存放区（台面、开放式层架）要写明在哪个点位取放
    geometry: Geometry | None = None  # 三维尺寸和槽位排布（仿真用）


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
        """按名字或别名在台账里找物品，用于把一句话里的物品对上号。

        取最长的匹配，避免"鸡米花"被别名"米"认成大米。
        """
        best, best_len = None, 0
        for item in self.items.values():
            for name in [item.name, *item.aliases]:
                if name and name in text and len(name) > best_len:
                    best, best_len = item, len(name)
        return best

    def find_container(self, text: str) -> Container | None:
        """按名字找容器（碗、杯子这类本身就是要搬的东西）。"""
        for c in sorted(self.containers.values(), key=lambda c: -len(c.name)):
            if c.name in text:
                return c
        return None

    def storage_of(self, container_id: str) -> StorageArea | None:
        loc = self.containers[container_id].location
        return self.storage.get(loc)

    def fixture_enclosing(self, location_id: str) -> Fixture | None:
        for f in self.fixtures.values():
            if f.encloses == location_id:
                return f
        return None

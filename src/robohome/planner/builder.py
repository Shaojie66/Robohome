"""计划构建器：把"拿什么、放哪、倒进哪、启动什么"补成机器人能直接执行的完整步骤。

家已经封装好了：每个存放区由哪扇门围着、门在哪个点位、哪些槽位空着，全都是已知的。
所以走动和开关门不需要大模型操心，由这里按固定规矩补上：

- 取放、拍照、倒东西之前，先走到对应点位，有门就打开
- 离开一个点位前，关上在这里打开的门
- 放东西只放空槽位：指定的槽位被占了就换同一处的空槽位
- 启动带危险的家电前，先关好它的门；手里还拿着东西就先放回取出的地方
- 第一次从台面（餐桌）取东西前先拍照确认（方案文档："查台账，到点后视觉确认"）
- 计划结束时，打开过的门全部关好

规则规划器用它写计划，大模型规划器用 normalize() 把模型给的计划补全。
补全后的计划照样要经过预演、安全规则和前置条件检查，这里不替任何检查放行。
"""

from __future__ import annotations

from ..skills.library import STOP_ACTIONS
from ..world import World
from .base import Step


class Builder:
    """边写计划边记住机器人会在哪、开着哪些门、东西放在哪。"""

    def __init__(self, world: World):
        self.home = world.home
        self.waypoint = world.robot.waypoint
        self.holding = world.robot.holding
        self.door_open = {f.id: f.state == "open" for f in world.home.fixtures.values()}
        self.opened = set(world.opened_by_robot)  # 机器人自己开的门，最后要关
        self.where = {c.id: (c.location, c.slot) for c in world.home.containers.values()}
        self.origin: dict[str, tuple[str, int]] = {}  # 这次计划里每个容器是从哪取出来的
        self.inspected: set[str] = set()
        self.steps: list[Step] = []

    # ---- 位置 ----

    def waypoint_of(self, location: str) -> str | None:
        fixture = self.home.fixture_enclosing(location)
        if fixture:
            return fixture.waypoint
        if location in self.home.storage:
            return self.home.storage[location].waypoint
        if location in self.home.appliances:
            return self.home.appliances[location].waypoint
        return None

    def goto(self, waypoint: str | None) -> None:
        if waypoint is None or waypoint == self.waypoint:
            return
        for fid in sorted(self.opened):
            if self.home.fixtures[fid].waypoint == self.waypoint:
                self.close(fid)
        self.steps.append(Step("navigate", {"waypoint": waypoint}))
        self.waypoint = waypoint

    def access(self, location: str) -> None:
        """走到存放区或家电跟前，有门就打开。"""
        fixture = self.home.fixture_enclosing(location)
        self.goto(self.waypoint_of(location))
        if fixture and not self.door_open[fixture.id]:
            self.steps.append(Step("open", {"fixture": fixture.id}))
            self.door_open[fixture.id] = True
            self.opened.add(fixture.id)

    def close(self, fid: str) -> None:
        if not self.door_open.get(fid):
            return
        self.goto(self.home.fixtures[fid].waypoint)
        self.steps.append(Step("close", {"fixture": fid}))
        self.door_open[fid] = False
        self.opened.discard(fid)

    def close_area(self, location: str) -> None:
        fixture = self.home.fixture_enclosing(location)
        if fixture and fixture.id in self.opened:
            self.close(fixture.id)

    def finish(self) -> list[Step]:
        here = [f for f in sorted(self.opened) if self.home.fixtures[f].waypoint == self.waypoint]
        for fid in here + [f for f in sorted(self.opened) if f not in here]:
            self.close(fid)
        return self.steps

    # ---- 动作 ----

    def free_slot(self, location: str, prefer: int | None = None) -> int | None:
        if location in self.home.storage:
            n = self.home.storage[location].slots
        elif location in self.home.appliances:
            n = self.home.appliances[location].slots
        else:
            return None
        taken = {slot for cid, (loc, slot) in self.where.items() if loc == location and cid != self.holding}
        if isinstance(prefer, int) and 0 <= prefer < n and prefer not in taken:
            return prefer
        return next((i for i in range(n) if i not in taken), None)

    def pick(self, container: str) -> None:
        location, slot = self.where[container]
        self.access(location)
        area = self.home.storage.get(location)
        if area and area.kind == "surface" and location not in self.inspected:
            self.inspect(location)  # 台面上的东西是人随手放的，先拍照确认
        self.steps.append(Step("pick", {"container": container}))
        self.origin.setdefault(container, (location, slot))
        self.where[container] = ("robot", 0)
        self.holding = container

    def place(self, container: str, location: str, slot: int) -> None:
        self.access(location)
        self.steps.append(Step("place", {"container": container, "location": location, "slot": slot}))
        self.where[container] = (location, slot)
        self.holding = None

    def put_back(self, container: str) -> bool:
        """把手里的容器放回这次取出它的地方（原槽位被占就换个空槽位）。"""
        if container not in self.origin:
            return False
        location, slot = self.origin[container]
        free = self.free_slot(location, slot)
        if free is None:
            return False
        self.place(container, location, free)
        self.close_area(location)
        return True

    def inspect(self, location: str) -> None:
        self.access(location)
        self.steps.append(Step("inspect", {"location": location}))
        self.inspected.add(location)

    def pour(self, container: str, into: str) -> None:
        self.access(into)
        self.steps.append(Step("pour", {"container": container, "into": into}))

    def call(self, appliance: str, action: str, params: dict | None = None) -> None:
        params = {"minutes": None, "temp_c": None, **(params or {})}
        a = self.home.appliances.get(appliance)
        if a and a.hazards and action not in STOP_ACTIONS:
            if self.holding:
                self.put_back(self.holding)  # 手里拿着东西不能启动危险家电
            if a.door:
                self.close(a.door)  # 家电门开着不能启动
        self.steps.append(Step("appliance", {"appliance": appliance, "action": action, "params": params}))


def normalize(steps: list[Step], world: World) -> list[Step]:
    """把大模型给的计划补全：补上走动和开关门，修正被占的槽位。

    模型写的 navigate / open / close 会被吸收进来（重复的去掉，缺的补上）；
    认不出来的步骤原样保留，让预演去报错，而不是在这里猜。
    """
    b = Builder(world)
    home = world.home
    for step in steps:
        a = step.args if isinstance(step.args, dict) else {}
        skill = step.skill
        if skill == "navigate" and a.get("waypoint") in home.rooms[world.robot.room].waypoints:
            b.goto(a["waypoint"])
        elif skill == "open" and a.get("fixture") in home.fixtures:
            f = home.fixtures[a["fixture"]]
            if f.encloses:
                b.access(f.encloses)
            else:
                b.steps.append(step)
        elif skill == "close" and a.get("fixture") in home.fixtures:
            b.close(a["fixture"])
        elif skill == "pick" and a.get("container") in b.where and b.where[a["container"]][0] != "robot":
            b.pick(a["container"])
        elif skill == "place" and a.get("container") == b.holding and b.free_slot(a.get("location", ""), 0) is not None:
            location = a["location"]
            b.place(a["container"], location, b.free_slot(location, a.get("slot")))
        elif skill == "inspect" and (a.get("location") in home.storage or a.get("location") in home.appliances):
            if a["location"] not in b.inspected:
                b.inspect(a["location"])
        elif skill == "pour" and a.get("container") == b.holding and a.get("into") in home.appliances:
            b.pour(a["container"], a["into"])
        elif skill == "appliance" and a.get("appliance") in home.appliances:
            b.call(a["appliance"], a.get("action", ""), a.get("params") if isinstance(a.get("params"), dict) else None)
        else:
            b.steps.append(step)
    return b.finish()

"""规则规划器：只认识几种固定句式（MVP 的五个任务），用于在接入大模型之前跑通全链路。

它的价值在于说明：因为家已经被"封装"好了，把一句话落成动作序列只需要查表，
牛肉在哪个盒子、盒子在哪个存放区、存放区由哪扇门围着、要站在哪个点位、碗的原位在哪，全部是已知的。

认识的说法：
- 看看冰箱里还有什么 / 看看冷冻室           → 开门、拍照、关门
- 把冷冻牛肉转到冷藏层解冻                  → 冷冻室取盒、放进冷藏室
- 把牛肉拿出来解冻（不提冷藏）              → 放进微波炉解冻
- 把鸡米花放进空气炸锅并启动                → 取密封罐、拉出炸篮、倒入、推回、罐子放回、接口启动
- 把洗好的碗从洗碗机收到柜子                → 开洗碗机、逐个取碗、放回原位
- 把桌上散落的常用物品收回原位              → 拍照识别、逐个放回原位
- 把牛奶拿出来                              → 取出容器
"""

from __future__ import annotations

import re

from ..world import World
from .base import PlanningError, Step

LOOK_WORDS = ("看看", "查看", "看一下", "看一眼", "有什么", "拍")
TIDY_WORDS = ("归位", "原位", "收回", "收拾", "收好")
DEFAULT_FRY = {"minutes": 12, "temp_c": 180}
DEFAULT_DEFROST_MINUTES = 8


class _Builder:
    """边写计划边记住机器人会在哪、开着哪些门、东西放在哪。

    保证三件事：离开一个点位前关上在这里打开的门；放东西只放空槽位；计划结束时打开过的门都关好。
    """

    def __init__(self, world: World):
        self.home = world.home
        self.waypoint = world.robot.waypoint
        self.holding = world.robot.holding
        self.door_open = {f.id: f.state == "open" for f in world.home.fixtures.values()}
        self.opened = set(world.opened_by_robot)  # 机器人自己开的门，最后要关
        self.where = {c.id: (c.location, c.slot) for c in world.home.containers.values()}
        self.steps: list[Step] = []

    # ---- 位置 ----

    def _waypoint_of(self, location: str) -> str | None:
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
                self._close(fid)
        self.steps.append(Step("navigate", {"waypoint": waypoint}))
        self.waypoint = waypoint

    def access(self, location: str) -> None:
        """走到存放区或家电跟前，有门就打开。"""
        fixture = self.home.fixture_enclosing(location)
        self.goto(self._waypoint_of(location))
        if fixture and not self.door_open[fixture.id]:
            self.steps.append(Step("open", {"fixture": fixture.id}))
            self.door_open[fixture.id] = True
            self.opened.add(fixture.id)

    def _close(self, fid: str) -> None:
        self.goto(self.home.fixtures[fid].waypoint)
        self.steps.append(Step("close", {"fixture": fid}))
        self.door_open[fid] = False
        self.opened.discard(fid)

    def close_area(self, location: str) -> None:
        fixture = self.home.fixture_enclosing(location)
        if fixture and fixture.id in self.opened:
            self._close(fixture.id)

    def finish(self) -> list[Step]:
        here = [f for f in sorted(self.opened) if self.home.fixtures[f].waypoint == self.waypoint]
        for fid in here + [f for f in sorted(self.opened) if f not in here]:
            self._close(fid)
        return self.steps

    # ---- 动作 ----

    def free_slot(self, location: str, prefer: int | None = None) -> int | None:
        if location in self.home.storage:
            n = self.home.storage[location].slots
        else:
            n = self.home.appliances[location].slots
        taken = {slot for cid, (loc, slot) in self.where.items() if loc == location and cid != self.holding}
        if prefer is not None and 0 <= prefer < n and prefer not in taken:
            return prefer
        return next((i for i in range(n) if i not in taken), None)

    def pick(self, container: str) -> None:
        location, _ = self.where[container]
        self.access(location)
        self.steps.append(Step("pick", {"container": container}))
        self.where[container] = ("robot", 0)
        self.holding = container

    def place(self, container: str, location: str, slot: int) -> None:
        self.access(location)
        self.steps.append(Step("place", {"container": container, "location": location, "slot": slot}))
        self.where[container] = (location, slot)
        self.holding = None

    def inspect(self, location: str) -> None:
        self.access(location)
        self.steps.append(Step("inspect", {"location": location}))

    def pour(self, container: str, into: str) -> None:
        self.access(into)
        self.steps.append(Step("pour", {"container": container, "into": into}))

    def call(self, appliance: str, action: str, minutes: int | None = None, temp_c: int | None = None) -> None:
        params = {"minutes": minutes, "temp_c": temp_c}
        self.steps.append(Step("appliance", {"appliance": appliance, "action": action, "params": params}))


class RulePlanner:
    def plan(self, instruction: str, world: World) -> list[Step]:
        if "洗碗机" in instruction:
            return self._unload_dishwasher(instruction, world)
        if "桌" in instruction and any(w in instruction for w in TIDY_WORDS):
            return self._tidy_surface(instruction, world)
        if "空气炸锅" in instruction or "炸" in instruction:
            return self._air_fry(instruction, world)
        if "解冻" in instruction:
            if "冷藏" in instruction:
                return self._thaw_in_chiller(instruction, world)
            return self._defrost(instruction, world)
        if any(w in instruction for w in LOOK_WORDS):
            return self._look(instruction, world)
        if "拿出来" in instruction or "取出" in instruction:
            return self._fetch(instruction, world)
        raise PlanningError(f"规则规划器还不认识这句话：{instruction}")

    # ---- 共用 ----

    @staticmethod
    def _item_container(instruction: str, world: World) -> str:
        item = world.home.find_item(instruction)
        if item is None:
            raise PlanningError("台账里找不到这句话提到的物品")
        if item.container not in world.home.containers:
            raise PlanningError(f"{item.name}现在不在容器里（在 {item.container}）")
        return item.container

    @staticmethod
    def _home_of(world: World, b: _Builder, container: str) -> tuple[str, int]:
        c = world.home.containers[container]
        if c.home is None:
            raise PlanningError(f"{c.name}没有登记原位，不知道该放回哪里")
        slot = b.free_slot(c.home, c.home_slot)
        if slot is None:
            raise PlanningError(f"{world.home.storage[c.home].name}放满了，{c.name}放不回去")
        return c.home, slot

    # ---- 1. 远程查看 ----

    def _look(self, instruction: str, world: World) -> list[Step]:
        home = world.home
        if "冰箱" in instruction:
            targets = [s.id for s in home.storage.values() if s.temperature in ("chilled", "frozen")]
        else:
            targets = [
                s.id for s in home.storage.values()
                if s.name in instruction or s.name.removesuffix("室") in instruction
            ] + [a.id for a in home.appliances.values() if a.name in instruction]
        if not targets:
            raise PlanningError("不知道要看哪里")
        b = _Builder(world)
        for location in targets:
            b.inspect(location)
            b.close_area(location)
        return b.finish()

    # ---- 2. 解冻 ----

    def _thaw_in_chiller(self, instruction: str, world: World) -> list[Step]:
        container = self._item_container(instruction, world)
        chiller = next((s for s in world.home.storage.values() if s.temperature == "chilled"), None)
        if chiller is None:
            raise PlanningError("家里没有冷藏区")
        b = _Builder(world)
        source, _ = b.where[container]
        b.pick(container)
        b.close_area(source)
        slot = b.free_slot(chiller.id)
        if slot is None:
            raise PlanningError(f"{chiller.name}放满了")
        b.place(container, chiller.id, slot)
        b.close_area(chiller.id)
        return b.finish()

    def _defrost(self, instruction: str, world: World) -> list[Step]:
        microwave = next((a for a in world.home.appliances.values() if "defrost" in a.capabilities), None)
        if microwave is None:
            raise PlanningError("家里没有能解冻的家电")
        container = self._item_container(instruction, world)
        b = _Builder(world)
        source, _ = b.where[container]
        b.pick(container)
        b.close_area(source)
        b.place(container, microwave.id, 0)
        b.close_area(microwave.id)
        b.call(microwave.id, "defrost", minutes=DEFAULT_DEFROST_MINUTES)
        return b.finish()

    # ---- 3. 空气炸锅 ----

    def _air_fry(self, instruction: str, world: World) -> list[Step]:
        fryer = next((a for a in world.home.appliances.values() if a.accepts_pour and "air_fry" in a.capabilities), None)
        if fryer is None:
            raise PlanningError("家里没有能直接倒进去的空气炸锅")
        container = self._item_container(instruction, world)
        minutes = _number(instruction, r"(\d+)\s*分钟") or DEFAULT_FRY["minutes"]
        temp = _number(instruction, r"(\d+)\s*度") or DEFAULT_FRY["temp_c"]
        b = _Builder(world)
        source, slot = b.where[container]
        b.pick(container)
        b.close_area(source)
        b.pour(container, fryer.id)  # 拉出炸篮、倒入
        b.close_area(fryer.id)  # 推回炸篮
        b.place(container, source, slot)  # 罐子放回原处，空着手才能启动
        b.close_area(source)
        b.call(fryer.id, "air_fry", minutes=minutes, temp_c=temp)
        return b.finish()

    # ---- 4. 洗碗机收碗 ----

    def _unload_dishwasher(self, instruction: str, world: World) -> list[Step]:
        dishwasher = next((a for a in world.home.appliances.values() if a.kind == "dishwasher"), None)
        if dishwasher is None:
            raise PlanningError("家里没有洗碗机")
        dishes = sorted(
            (c for c in world.home.containers.values() if c.location == dishwasher.id),
            key=lambda c: c.slot,
        )
        if not dishes:
            raise PlanningError(f"{dishwasher.name}里没有东西")
        b = _Builder(world)
        for c in dishes:
            b.pick(c.id)
            location, slot = self._home_of(world, b, c.id)
            b.place(c.id, location, slot)
        return b.finish()

    # ---- 5. 桌面归位 ----

    def _tidy_surface(self, instruction: str, world: World) -> list[Step]:
        surfaces = [s for s in world.home.storage.values() if s.kind == "surface" and "桌" in s.name]
        named = [s for s in surfaces if s.name in instruction]
        surfaces = named or surfaces
        if not surfaces:
            raise PlanningError("家里没有登记桌面")
        b = _Builder(world)
        for surface in surfaces:
            b.inspect(surface.id)  # 先拍照识别桌上有什么
            items = sorted(
                (c for c in world.home.containers.values() if c.location == surface.id and c.home is not None),
                key=lambda c: c.slot,
            )
            for c in items:  # 没有原位的东西留在桌上，照片里会提醒主人
                b.pick(c.id)
                location, slot = self._home_of(world, b, c.id)
                b.place(c.id, location, slot)
        return b.finish()

    # ---- 取出 ----

    def _fetch(self, instruction: str, world: World) -> list[Step]:
        container = self._item_container(instruction, world)
        b = _Builder(world)
        source, _ = b.where[container]
        b.pick(container)
        b.close_area(source)
        return b.finish()


def _number(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None

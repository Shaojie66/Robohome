"""规则规划器：只认识几种固定句式，用于在接入大模型之前跑通全链路。

它的价值在于说明：因为家已经被"封装"好了，把一句话落成动作序列只需要查表，
牛肉在哪个盒子、盒子在哪个存放区、存放区由哪扇门围着、要站在哪个点位，全部是已知的。
"""

from __future__ import annotations

from ..world import World
from .base import PlanningError, Step


class RulePlanner:
    def plan(self, instruction: str, world: World) -> list[Step]:
        if "解冻" in instruction:
            return self._defrost(instruction, world)
        if "拿出来" in instruction or "取出" in instruction:
            return self._fetch(instruction, world)
        raise PlanningError(f"规则规划器还不认识这句话：{instruction}")

    # 取出一个物品所在的容器，并把存放区的门关好
    def _take_out(self, instruction: str, world: World) -> tuple[list[Step], str]:
        home = world.home
        item = home.find_item(instruction)
        if item is None:
            raise PlanningError("台账里找不到这句话提到的物品")
        container = home.containers[item.container]
        fixture = home.fixture_enclosing(container.location)
        steps: list[Step] = []
        if fixture:
            steps += [
                Step("navigate", {"waypoint": fixture.waypoint}),
                Step("open", {"fixture": fixture.id}),
                Step("pick", {"container": container.id}),
                Step("close", {"fixture": fixture.id}),
            ]
        else:
            steps.append(Step("pick", {"container": container.id}))
        return steps, container.id

    def _fetch(self, instruction: str, world: World) -> list[Step]:
        steps, _ = self._take_out(instruction, world)
        return steps

    def _defrost(self, instruction: str, world: World) -> list[Step]:
        home = world.home
        microwave = next((a for a in home.appliances.values() if "defrost" in a.capabilities), None)
        if microwave is None:
            raise PlanningError("家里没有能解冻的家电")
        steps, container = self._take_out(instruction, world)
        steps.append(Step("navigate", {"waypoint": microwave.waypoint}))
        if microwave.door:
            steps.append(Step("open", {"fixture": microwave.door}))
        steps.append(Step("place", {"container": container, "location": microwave.id}))
        if microwave.door:
            steps.append(Step("close", {"fixture": microwave.door}))
        steps.append(Step("appliance", {"appliance": microwave.id, "action": "defrost", "params": {"minutes": 8}}))
        return steps

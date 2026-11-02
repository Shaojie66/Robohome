"""技能库：机器人能做的每个动作都是一个函数。

每个技能有三部分：
- params   参数的 JSON Schema，规划层直接拿去约束大模型的输出
- check    前置条件：在当前世界状态下能不能做，不能做就返回原因
- run      执行：第一阶段只改世界状态（模拟实现）；仿真和真机阶段在这里驱动执行器

技能只管"能不能做、怎么做"，"该不该做"的安全规则放在 scheduler/safety.py，统一检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..appliances.base import ApplianceError
from ..world import World

HELD = "robot"  # 容器被拿在手里时的位置


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    params: dict[str, dict]  # 参数名 → JSON Schema
    check: Callable[..., str | None]
    run: Callable[..., str]


# ---------- 辅助 ----------


def _enclosure_open(world: World, location: str) -> str | None:
    """存放区或家电如果有门，门必须开着。"""
    fixture = world.home.fixture_enclosing(location)
    if fixture and fixture.state != "open":
        return f"{fixture.name}没开"
    return None


def _waypoint_for_location(world: World, location: str) -> str | None:
    fixture = world.home.fixture_enclosing(location)
    if fixture:
        return fixture.waypoint
    if location in world.home.appliances:
        return world.home.appliances[location].waypoint
    return None


def _at(world: World, waypoint: str | None) -> str | None:
    if waypoint and world.robot.waypoint != waypoint:
        return f"机器人在 {world.robot.waypoint}，需要先到 {waypoint}"
    return None


# ---------- navigate ----------


def _check_navigate(world: World, waypoint: str) -> str | None:
    room = world.home.rooms[world.robot.room]
    if waypoint not in room.waypoints:
        return f"点位 {waypoint} 不存在"
    return None


def _run_navigate(world: World, waypoint: str) -> str:
    world.robot.waypoint = waypoint
    return f"到达 {waypoint}"


# ---------- open / close ----------


def _check_fixture(world: World, fixture: str, want: str) -> str | None:
    f = world.home.fixtures.get(fixture)
    if f is None:
        return f"部件 {fixture} 不存在"
    if err := _at(world, f.waypoint):
        return err
    if f.state == want:
        return f"{f.name}已经是{'开' if want == 'open' else '关'}的"
    return None


def _run_open(world: World, fixture: str) -> str:
    f = world.home.fixtures[fixture]
    f.state = "open"
    world.sync_appliance_doors()
    return f"打开{f.name}"


def _run_close(world: World, fixture: str) -> str:
    f = world.home.fixtures[fixture]
    f.state = "closed"
    world.sync_appliance_doors()
    return f"关上{f.name}"


# ---------- pick / place ----------


def _check_pick(world: World, container: str) -> str | None:
    c = world.home.containers.get(container)
    if c is None:
        return f"容器 {container} 不存在"
    if world.robot.holding:
        return f"手里已经拿着 {world.robot.holding}"
    if c.location == HELD:
        return f"{c.name}已经在手里"
    if err := _at(world, _waypoint_for_location(world, c.location)):
        return err
    if err := _enclosure_open(world, c.location):
        return err
    return None


def _run_pick(world: World, container: str) -> str:
    c = world.home.containers[container]
    if container in world.missing_containers:
        raise SkillFailed(f"到了 {c.location} 槽位 {c.slot} 没找到{c.name}，台账可能过期")
    src = c.location
    c.location, c.slot = HELD, 0
    world.robot.holding = container
    return f"从 {src} 取出{c.name}"


def _check_place(world: World, container: str, location: str, slot: int = 0) -> str | None:
    if world.robot.holding != container:
        return f"手里没有拿着 {container}"
    if location in world.home.storage:
        if not 0 <= slot < world.home.storage[location].slots:
            return f"{location} 没有槽位 {slot}"
    elif location not in world.home.appliances:
        return f"位置 {location} 不存在"
    if err := _at(world, _waypoint_for_location(world, location)):
        return err
    if err := _enclosure_open(world, location):
        return err
    if other := world.container_at(location, slot):
        return f"{location} 槽位 {slot} 已被 {other} 占用"
    return None


def _run_place(world: World, container: str, location: str, slot: int = 0) -> str:
    c = world.home.containers[container]
    c.location, c.slot = location, slot
    world.robot.holding = None
    return f"把{c.name}放进 {location}"


# ---------- appliance ----------


STOP_ACTIONS = ("stop", "off")


def _check_appliance(world: World, appliance: str, action: str, params: dict | None = None) -> str | None:
    a = world.home.appliances.get(appliance)
    if a is None:
        return f"家电 {appliance} 不存在"
    if action not in a.capabilities:
        return f"{a.name}不支持 {action}"
    # 加热、切割这类有危险的动作必须给出运行时间，不能让家电无限期运行
    if a.hazards and action not in STOP_ACTIONS and not (params or {}).get("minutes"):
        return f"启动{a.name} {action} 必须指定运行分钟数（params.minutes）"
    return None


def _run_appliance(world: World, appliance: str, action: str, params: dict | None = None) -> str:
    params = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        status = world.appliances[appliance].call(action, params)
    except ApplianceError as e:
        # 让调度器当作失败处理
        raise SkillFailed(str(e)) from e
    name = world.home.appliances[appliance].name
    return f"调用{name} {action} {params if params else ''}".rstrip() + f" → running={status.get('running')}"


class SkillFailed(RuntimeError):
    """执行中途失败（家电拒绝、抓取失手等）。"""


def _s(type_, description: str) -> dict:
    return {"type": type_, "description": description}


SKILLS: dict[str, Skill] = {
    s.name: s
    for s in [
        Skill(
            "navigate",
            "移动到当前房间里的一个点位",
            {"waypoint": _s("string", "点位 id")},
            _check_navigate,
            _run_navigate,
        ),
        Skill(
            "open",
            "打开柜门、抽屉或家电门（只握把手）",
            {"fixture": _s("string", "部件 id")},
            lambda w, fixture: _check_fixture(w, fixture, "open"),
            _run_open,
        ),
        Skill(
            "close",
            "关上柜门、抽屉或家电门",
            {"fixture": _s("string", "部件 id")},
            lambda w, fixture: _check_fixture(w, fixture, "closed"),
            _run_close,
        ),
        Skill(
            "pick",
            "从存放区或家电里取出一个标准容器",
            {"container": _s("string", "容器 id")},
            _check_pick,
            _run_pick,
        ),
        Skill(
            "place",
            "把手里的容器放进存放区某个槽位，或放进家电",
            {
                "container": _s("string", "容器 id"),
                "location": _s("string", "存放区或家电 id"),
                "slot": _s("integer", "槽位序号；放进家电时填 0"),
            },
            _check_place,
            _run_place,
        ),
        Skill(
            "appliance",
            "调用联网家电的接口，例如微波炉解冻",
            {
                "appliance": _s("string", "家电 id"),
                "action": _s("string", "家电支持的动作，如 defrost"),
                "params": {
                    "type": "object",
                    "description": "动作参数",
                    "properties": {"minutes": _s(["integer", "null"], "运行分钟数；启动加热等危险动作时必填，停止时填 null")},
                    "required": ["minutes"],
                    "additionalProperties": False,
                },
            },
            _check_appliance,
            _run_appliance,
        ),
    ]
}


def catalog() -> list[dict]:
    """技能清单：名字、说明、参数 Schema。规划层用它约束大模型只能输出合法的技能调用。"""
    return [
        {
            "name": s.name,
            "description": s.description,
            "input_schema": {
                "type": "object",
                "properties": s.params,
                "required": list(s.params),
                "additionalProperties": False,
            },
        }
        for s in SKILLS.values()
    ]

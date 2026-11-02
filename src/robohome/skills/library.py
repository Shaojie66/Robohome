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
from ..world import FLOOR, World

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
    if location in world.home.storage:
        return world.home.storage[location].waypoint
    if location in world.home.appliances:
        return world.home.appliances[location].waypoint
    return None


def _slots(world: World, location: str) -> int | None:
    if location in world.home.storage:
        return world.home.storage[location].slots
    if location in world.home.appliances:
        return world.home.appliances[location].slots
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
    if world.body is not None:
        world.body.navigate(world, waypoint)  # 半路滑落会抛 SkillFailed，并把容器记到地上
    held = world.robot.holding
    if held and held in world.slippery:
        # 搬运途中滑落：东西掉在原地，机器人停在原地，不再继续
        c = world.home.containers[held]
        c.location, c.slot = FLOOR, 0
        world.robot.holding = None
        raise SkillFailed(f"去 {waypoint} 的路上{c.name}从手里滑落")
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
    if world.body is not None:
        world.body.open(world, fixture)
    f = world.home.fixtures[fixture]
    if fixture in world.stuck_fixtures:
        raise SkillFailed(f"{f.name}拉不开，可能卡住了")
    f.state = "open"
    world.opened_by_robot.add(fixture)
    world.sync_appliance_doors()
    return f"打开{f.name}"


def _run_close(world: World, fixture: str) -> str:
    if world.body is not None:
        world.body.close(world, fixture)
    f = world.home.fixtures[fixture]
    if fixture in world.stuck_fixtures:
        raise SkillFailed(f"{f.name}关不上，可能卡住了")
    f.state = "closed"
    world.opened_by_robot.discard(fixture)
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
    if world.body is not None:
        world.body.pick(world, container)  # 到了发现东西不在、或者是别的容器，会抛 SkillFailed
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
    n = _slots(world, location)
    if n is None:
        return f"位置 {location} 不存在"
    if not isinstance(slot, int) or not 0 <= slot < n:
        return f"{location} 没有槽位 {slot}（共 {n} 个，从 0 开始）"
    if err := _at(world, _waypoint_for_location(world, location)):
        return err
    if err := _enclosure_open(world, location):
        return err
    if other := world.container_at(location, slot):
        return f"{location} 槽位 {slot} 已被 {other} 占用"
    return None


def _run_place(world: World, container: str, location: str, slot: int = 0) -> str:
    c = world.home.containers[container]
    actual = (location, slot)
    if world.body is not None:
        actual = world.body.place(world, container, location, slot)  # 松手后它实际落在哪
    c.location, c.slot = actual
    world.robot.holding = None
    if actual != (location, slot):
        raise SkillFailed(f"{c.name}没放稳，最后落在 {actual[0]} 槽位 {actual[1]}，不是 {location} 槽位 {slot}")
    return f"把{c.name}放进 {location} 槽位 {slot}"


# ---------- inspect / pour ----------


def _check_inspect(world: World, location: str) -> str | None:
    if _slots(world, location) is None:
        return f"位置 {location} 不存在"
    if err := _at(world, _waypoint_for_location(world, location)):
        return err
    if err := _enclosure_open(world, location):
        return err
    return None


def _run_inspect(world: World, location: str) -> str:
    photo = world.body.inspect(world, location) if world.body is not None else world.look(location)
    world.photos.append(photo)
    world.notices += photo.notes
    return f"拍照：{photo.describe()}"


def _check_pour(world: World, container: str, into: str) -> str | None:
    if world.robot.holding != container:
        return f"手里没有拿着 {container}"
    a = world.home.appliances.get(into)
    if a is None:
        return f"家电 {into} 不存在"
    if not a.accepts_pour:
        return f"{a.name}不能直接倒东西进去"
    if not any(i.container == container for i in world.home.items.values()):
        return f"{world.home.containers[container].name}是空的"
    if err := _at(world, a.waypoint):
        return err
    if err := _enclosure_open(world, into):
        return err
    return None


def _run_pour(world: World, container: str, into: str) -> str:
    if world.body is not None:
        world.body.pour(world, container, into)
    moved = [i for i in world.home.items.values() if i.container == container]
    for i in moved:
        i.container = into
    names = "、".join(i.name for i in moved)
    return f"把{names}倒进{world.home.appliances[into].name}"


# ---------- appliance ----------


STOP_ACTIONS = ("stop", "off")


def _check_appliance(world: World, appliance: str, action: str, params: dict | None = None) -> str | None:
    a = world.home.appliances.get(appliance)
    if a is None:
        return f"家电 {appliance} 不存在"
    if action not in a.capabilities:
        return f"{a.name}不支持 {action}"
    # 加热、切割这类有危险的动作必须给出合理的运行时间，不能让家电无限期运行
    if a.hazards and action not in STOP_ACTIONS:
        minutes = (params or {}).get("minutes")
        # bool 也是 int 的子类，要单独排除
        if not isinstance(minutes, int) or isinstance(minutes, bool):
            return f"启动{a.name} {action} 必须指定运行分钟数（params.minutes，整数）"
        if not 1 <= minutes <= a.max_minutes:
            return f"{a.name} {action} 的运行时间必须在 1 到 {a.max_minutes} 分钟之间，收到 {minutes}"
    temp = (params or {}).get("temp_c")
    if temp is not None:
        if a.max_temp_c is None:
            return f"{a.name}不能设温度"
        if not isinstance(temp, int) or isinstance(temp, bool) or not 1 <= temp <= a.max_temp_c:
            return f"{a.name}的温度必须是 1 到 {a.max_temp_c} 之间的整数，收到 {temp}"
    return None


def _run_appliance(world: World, appliance: str, action: str, params: dict | None = None) -> str:
    params = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        status = world.appliances[appliance].call(action, params)
    except ApplianceError as e:
        # 让调度器当作失败处理
        raise SkillFailed(str(e)) from e
    if action in STOP_ACTIONS:
        world.started_by_robot.discard(appliance)
    else:
        world.started_by_robot.add(appliance)
    if world.body is not None:
        world.body.appliance(world, appliance, bool(status.get("running")))
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
                "slot": _s("integer", "槽位序号，从 0 开始，必须是空的；家电只有一个位置时填 0"),
            },
            _check_place,
            _run_place,
        ),
        Skill(
            "inspect",
            "给存放区或家电内部拍照，看看里面有什么、和台账对不对得上（有门的要先开门）",
            {"location": _s("string", "存放区或家电 id")},
            _check_inspect,
            _run_inspect,
        ),
        Skill(
            "pour",
            "把手里容器里的食物倒进家电（只限 accepts_pour 的家电，例如空气炸锅的炸篮），容器仍在手里",
            {"container": _s("string", "容器 id"), "into": _s("string", "家电 id")},
            _check_pour,
            _run_pour,
        ),
        Skill(
            "appliance",
            "调用联网家电的接口，例如微波炉解冻、空气炸锅启动；走网络接口，不需要站在家电旁边",
            {
                "appliance": _s("string", "家电 id"),
                "action": _s("string", "家电支持的动作，如 defrost"),
                "params": {
                    "type": "object",
                    "description": "动作参数",
                    "properties": {
                        "minutes": _s(["integer", "null"], "运行分钟数（整数）；启动加热等危险动作时必填且不超过家电的 max_minutes，停止时填 null"),
                        "temp_c": _s(["integer", "null"], "温度（摄氏度）；只有 max_temp_c 不为空的家电能设，其他填 null"),
                    },
                    "required": ["minutes", "temp_c"],
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

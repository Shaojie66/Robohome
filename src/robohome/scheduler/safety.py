"""安全规则：在每一步执行前统一检查，任何一条不过就停。

原则（来自方案文档）：
- 机器人不进禁区
- 机器人不直接接触热源和刀具：家电运行时不开它的门、不从里面取放东西
- 放进加热家电（微波炉、空气炸锅……）的容器必须标明能进这种家电
- 手里拿着东西时不调用带危险的家电，避免边拿边启动
- 不擅自停掉别人启动、正在运行的家电（例如洗碗机没洗完），主人明确要求的除外

规则和技能分开写，这样换仿真或真机时规则一行不用改。
"""

from __future__ import annotations

from typing import Callable

from ..world import World

Rule = Callable[[World, str, dict], str | None]


def no_forbidden_zone(world: World, skill: str, args: dict) -> str | None:
    if skill != "navigate":
        return None
    room = world.home.rooms[world.robot.room]
    if args.get("waypoint") in room.forbidden_waypoints:
        return f"{args['waypoint']} 是禁区"
    return None


def no_touching_running_hazard(world: World, skill: str, args: dict) -> str | None:
    target = None
    if skill in ("open", "close"):
        f = world.home.fixtures.get(args.get("fixture", ""))
        target = f.encloses if f else None
    elif skill == "pick":
        c = world.home.containers.get(args.get("container", ""))
        target = c.location if c else None
    elif skill in ("place", "inspect"):
        target = args.get("location")
    elif skill == "pour":
        target = args.get("into")
    a = world.home.appliances.get(target or "")
    if not (a and a.hazards):
        return None
    running = world.appliance_running(a.id)
    if running is None:
        return f"读不到{a.name}的状态，可能正在运行（{'/'.join(a.hazards)}），为安全起见不接触"
    if running:
        return f"{a.name}正在运行（{'/'.join(a.hazards)}），不能接触"
    return None


def heat_safe_containers_only(world: World, skill: str, args: dict) -> str | None:
    if skill != "place":
        return None
    a = world.home.appliances.get(args.get("location", ""))
    c = world.home.containers.get(args.get("container", ""))
    if a and "heat" in a.hazards and c and a.kind not in c.safe_for:
        return f"{c.name}不能进{a.name}"
    return None



def hands_free_before_hazard(world: World, skill: str, args: dict) -> str | None:
    if skill != "appliance" or args.get("action") in ("stop", "off"):
        return None
    a = world.home.appliances.get(args.get("appliance", ""))
    if a and a.hazards and world.robot.holding:
        return f"手里拿着 {world.robot.holding} 时不启动{a.name}"
    return None


STOP_WORDS = ("停", "关掉", "关闭", "关了", "暂停")


def no_interrupting_appliances(world: World, skill: str, args: dict) -> str | None:
    """家电正在跑主人的程序时，机器人不去碰它：不停它，也不给它下新程序。

    Mac 上 qwen3:8b 实测：洗碗机还在洗就被停掉；空气炸锅正在运行，又被重新下了一次 air_fry。
    """
    if skill != "appliance":
        return None
    a = world.home.appliances.get(args.get("appliance", ""))
    if a is None or a.id in world.started_by_robot or not world.appliance_running(a.id):
        return None
    if args.get("action") not in ("stop", "off"):
        return f"{a.name}正在运行，不是机器人启动的，不能再给它下新命令，会打断主人的程序"
    if a.name in world.instruction and any(w in world.instruction for w in STOP_WORDS):
        return None  # 主人明确让停
    return f"{a.name}正在运行，不是机器人启动的，主人没有要求就不能停"


def close_doors_before_leaving(world: World, skill: str, args: dict) -> str | None:
    if skill != "navigate" or args.get("waypoint") == world.robot.waypoint:
        return None
    left_open = [
        world.home.fixtures[f].name
        for f in sorted(world.opened_by_robot)
        if world.home.fixtures[f].waypoint == world.robot.waypoint and world.home.fixtures[f].state == "open"
    ]
    if left_open:
        return f"离开 {world.robot.waypoint} 前要先关上{'、'.join(left_open)}"
    return None


DEFAULT_RULES: list[Rule] = [
    no_forbidden_zone,
    no_touching_running_hazard,
    heat_safe_containers_only,
    hands_free_before_hazard,
    no_interrupting_appliances,
    close_doors_before_leaving,
]

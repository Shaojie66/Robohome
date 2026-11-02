"""安全规则：在每一步执行前统一检查，任何一条不过就停。

原则（来自方案文档）：
- 机器人不进禁区
- 机器人不直接接触热源和刀具：家电运行时不开它的门、不从里面取放东西
- 进微波炉的容器必须是微波炉可用的
- 手里拿着东西时不调用带危险的家电，避免边拿边启动

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
    elif skill == "place":
        target = args.get("location")
    a = world.home.appliances.get(target or "")
    if a and a.hazards and world.appliance_running(a.id):
        return f"{a.name}正在运行（{'/'.join(a.hazards)}），不能接触"
    return None


def microwave_safe_only(world: World, skill: str, args: dict) -> str | None:
    if skill != "place":
        return None
    a = world.home.appliances.get(args.get("location", ""))
    c = world.home.containers.get(args.get("container", ""))
    if a and a.kind == "microwave" and c and not c.microwave_safe:
        return f"{c.name}不能进微波炉"
    return None


def hands_free_before_hazard(world: World, skill: str, args: dict) -> str | None:
    if skill != "appliance" or args.get("action") in ("stop", "off"):
        return None
    a = world.home.appliances.get(args.get("appliance", ""))
    if a and a.hazards and world.robot.holding:
        return f"手里拿着 {world.robot.holding} 时不启动{a.name}"
    return None


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
    microwave_safe_only,
    hands_free_before_hazard,
    close_doors_before_leaving,
]

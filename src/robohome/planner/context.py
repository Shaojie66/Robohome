"""把当前世界状态压成一份给大模型看的简要描述。

只放规划需要的信息：点位、部件开关状态、存放区还有哪些空槽位、容器在哪装着什么、原位在哪、
家电能做什么、机器人在哪拿着什么。
坐标、尺寸这类执行层才用的数据不放，省 token 也减少干扰。
"""

from __future__ import annotations

import json

from ..world import World


def describe_world(world: World) -> dict:
    home = world.home
    contents: dict[str, list[str]] = {}
    for item in home.items.values():
        contents.setdefault(item.container, []).append(item.name)

    return {
        "robot": {
            "room": world.robot.room,
            "waypoint": world.robot.waypoint,
            "holding": world.robot.holding,
            "opened_by_robot": sorted(world.opened_by_robot),  # 上一个任务留下没关的门，规划时要先关上
        },
        "rooms": [
            {
                "id": r.id,
                "name": r.name,
                "waypoints": sorted(r.waypoints),
                "forbidden_waypoints": r.forbidden_waypoints,
            }
            for r in home.rooms.values()
        ],
        "fixtures": [
            {"id": f.id, "name": f.name, "state": f.state, "waypoint": f.waypoint, "encloses": f.encloses}
            for f in home.fixtures.values()
        ],
        "storage": [
            {
                "id": s.id,
                "name": s.name,
                "kind": s.kind,
                "temperature": s.temperature,
                "waypoint": s.waypoint,  # 没有门的存放区（如餐桌）在这个点位取放
                "free_slots": _free_slots(world, s.id, s.slots),
            }
            for s in home.storage.values()
        ],
        "containers": [
            {
                "id": c.id,
                "name": c.name,
                "kind": c.kind,
                "location": c.location,
                "slot": c.slot,
                "home": c.home,
                "home_slot": c.home_slot,
                "safe_for": c.safe_for,
                "contents": contents.get(c.id, []),
            }
            for c in home.containers.values()
        ],
        "items": [
            {"id": i.id, "name": i.name, "aliases": i.aliases, "category": i.category, "container": i.container}
            for i in home.items.values()
        ],
        "appliances": [
            {
                "id": a.id,
                "name": a.name,
                "kind": a.kind,
                "waypoint": a.waypoint,
                "door": a.door,
                "capabilities": a.capabilities,
                "hazards": a.hazards,
                "max_minutes": a.max_minutes,
                "max_temp_c": a.max_temp_c,
                "accepts_pour": a.accepts_pour,
                "free_slots": _free_slots(world, a.id, a.slots),
                "contents": contents.get(a.id, []),  # 倒进去的食物
                "running": world.appliance_running(a.id),  # None 表示读不到状态
                "online": _online(world, a.id),
            }
            for a in home.appliances.values()
        ],
    }


def _free_slots(world: World, location: str, n: int) -> list[int]:
    return [i for i in range(n) if world.container_at(location, i) is None]


def _online(world: World, appliance_id: str) -> bool | None:
    try:
        return world.appliances[appliance_id].status().get("online", True) is not False
    except Exception:
        return None  # appliance_running 已经记过日志


def world_json(world: World) -> str:
    return json.dumps(describe_world(world), ensure_ascii=False, indent=1)

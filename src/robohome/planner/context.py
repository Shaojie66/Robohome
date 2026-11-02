"""把当前世界状态压成一份给大模型看的简要描述。

只放规划需要的信息：点位、部件开关状态、容器在哪装着什么、家电能做什么、机器人在哪拿着什么。
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
        "robot": {"room": world.robot.room, "waypoint": world.robot.waypoint, "holding": world.robot.holding},
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
            {"id": s.id, "name": s.name, "slots": s.slots, "temperature": s.temperature}
            for s in home.storage.values()
        ],
        "containers": [
            {
                "id": c.id,
                "name": c.name,
                "location": c.location,
                "slot": c.slot,
                "microwave_safe": c.microwave_safe,
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
                "running": world.appliance_running(a.id),
            }
            for a in home.appliances.values()
        ],
    }


def world_json(world: World) -> str:
    return json.dumps(describe_world(world), ensure_ascii=False, indent=1)

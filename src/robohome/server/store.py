"""家庭服务器的数据目录：台账、机器人状态、任务和问题、照片、配对码。

    <数据目录>/
      home.json    台账：家里每样东西在哪（格式和规范文件一样，机器人手里拿着的东西也记在这里）
      state.json   机器人在哪、手里拿着什么、自己开了哪些门、家电的状态
      tasks.json   任务、进度、回报、等主人回答的问题
      photos/      机器人拍的照片
      key          手机连上来用的配对码

每做完一步就存一次，服务重启（或者电脑断电）之后接着用。写文件先写临时文件再改名，不会写坏一半。
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from ..appliances.base import MockAppliance
from ..spec.dump import dumps_compact, home_to_dict
from ..spec.loader import parse_home
from ..spec.models import Home
from ..world import RobotState, World

KEY_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"  # 去掉了容易看错的 l、o、0、1
KEY_LENGTH = 8


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.photos = self.root / "photos"
        self.photos.mkdir(parents=True, exist_ok=True)

    # ---------- 读写 ----------

    def read(self, name: str, default=None):
        path = self.root / name
        if not path.exists():
            return default
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def write(self, name: str, value, compact: bool = False) -> None:
        path = self.root / name
        tmp = path.with_name(path.name + ".tmp")
        text = dumps_compact(value) if compact else json.dumps(value, ensure_ascii=False, indent=1)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def key(self) -> str:
        """配对码：第一次启动时随机生成，之后一直用这个；想换就删掉 key 文件重启。"""
        path = self.root / "key"
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
        key = "".join(secrets.choice(KEY_ALPHABET) for _ in range(KEY_LENGTH))
        path.write_text(key + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return key

    # ---------- 世界 ----------

    def has_world(self) -> bool:
        return (self.root / "home.json").exists()

    def save_world(self, world: World) -> None:
        self.write("home.json", home_to_dict(world.home), compact=True)
        self.write("state.json", world_state(world))

    def load_world(self) -> World:
        home = parse_home(self.read("home.json"), runtime=True)
        return restore_world(home, self.read("state.json", {}))


def world_state(world: World) -> dict:
    """台账以外、重启后还要接着用的状态。注入的故障是测试用的，不存。"""
    appliances = {}
    for aid, adapter in world.appliances.items():
        if isinstance(adapter, MockAppliance):
            appliances[aid] = {
                "online": adapter.online, "running": adapter.running, "mode": adapter.mode,
                "last_params": adapter.last_params,
            }
    r = world.robot
    return {
        "robot": {"room": r.room, "waypoint": r.waypoint, "holding": r.holding},
        "opened_by_robot": sorted(world.opened_by_robot),
        "started_by_robot": sorted(world.started_by_robot),
        "appliances": appliances,
    }


def restore_world(home: Home, state: dict) -> World:
    robot = state.get("robot") or {}
    room = robot.get("room") if robot.get("room") in home.rooms else None
    waypoint = robot.get("waypoint") if room and robot.get("waypoint") in home.rooms[room].waypoints else None
    if room is None:
        first = next(iter(home.rooms.values()))
        room, waypoint = first.id, "entrance" if "entrance" in first.waypoints else None
    world = World.from_home(home, start_room=room, start_waypoint=waypoint)
    holding = robot.get("holding")
    world.robot = RobotState(room=world.robot.room, waypoint=world.robot.waypoint,
                             holding=holding if holding in home.containers else None)
    world.opened_by_robot = {f for f in state.get("opened_by_robot", []) if f in home.fixtures}
    world.started_by_robot = {a for a in state.get("started_by_robot", []) if a in home.appliances}
    for aid, saved in (state.get("appliances") or {}).items():
        adapter = world.appliances.get(aid)
        if isinstance(adapter, MockAppliance):
            adapter.online = saved.get("online", True)
            adapter.running = bool(saved.get("running"))
            adapter.mode = saved.get("mode")
            adapter.last_params = dict(saved.get("last_params") or {})
    world.sync_appliance_doors()
    return world

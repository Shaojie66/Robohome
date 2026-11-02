"""家电适配层。

机器人从不碰家电内部：它只把东西放进去、关上门，然后像调用函数一样调用接口。
加热、切割这些危险工作由家电自己完成，家电自己负责门没关好就不启动这类安全逻辑。

真实接入（Home Assistant、米家）时只需要再写一个实现 ApplianceAdapter 的类，
调度器和技能库都不用改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class ApplianceError(RuntimeError):
    """家电拒绝执行或无法连接。"""


class ApplianceAdapter(Protocol):
    appliance_id: str

    def call(self, action: str, params: dict | None = None) -> dict: ...

    def status(self) -> dict: ...


@dataclass
class MockAppliance:
    """模拟家电：接口形状和真实家电一致，只在内存里改状态。

    door_closed 由调度器在关门、开门时同步过来（真实家电会自己上报门状态）。
    """

    appliance_id: str
    capabilities: list[str]
    online: bool = True
    door_closed: bool = True
    running: bool = False
    mode: str | None = None
    log: list[str] = field(default_factory=list)
    last_params: dict = field(default_factory=dict)  # 最近一次启动时的参数，测试里核对时间和温度

    def call(self, action: str, params: dict | None = None) -> dict:
        params = params or {}
        if not self.online:
            raise ApplianceError(f"{self.appliance_id} 离线")
        if action not in self.capabilities:
            raise ApplianceError(f"{self.appliance_id} 不支持 {action}")
        if action == "stop" or action == "off":
            self.running, self.mode = False, None
        else:
            if not self.door_closed:
                raise ApplianceError(f"{self.appliance_id} 门没关，拒绝启动")
            self.running, self.mode = True, action
        self.log.append(f"{action} {params}".strip())
        if action not in ("stop", "off"):
            self.last_params = dict(params)
        return self.status()

    def status(self) -> dict:
        return {
            "id": self.appliance_id,
            "online": self.online,
            "running": self.running,
            "mode": self.mode,
            "door_closed": self.door_closed,
        }

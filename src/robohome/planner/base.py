"""规划层：把一句话变成技能调用序列。

第 1 步只放一个规则规划器，用来把调度链路跑通；
第 2 步接入大模型规划器（LLMPlanner），它实现同一个 Planner 接口，调度器不用改。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from ..world import World


@dataclass
class Step:
    skill: str
    args: dict = field(default_factory=dict)

    def describe(self) -> str:
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in self.args.items())
        return f"{self.skill}({args})"


class Planner(Protocol):
    def plan(self, instruction: str, world: World) -> list[Step]: ...


class PlanningError(ValueError):
    pass

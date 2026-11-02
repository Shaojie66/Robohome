"""大模型规划器：让大模型把一句话拆成技能调用序列。

后端可以是 Claude（云端），也可以是本地模型（Ollama 等），见 robohome.llm。

流程：
1. 把技能清单（带参数 Schema）和当前世界状态交给大模型，用结构化输出约束它只能返回合法的技能调用
2. 拿到计划后先在世界副本上预演（安全规则 + 前置条件都会检查）
3. 预演失败就把失败原因告诉大模型，让它重新规划；超过次数就放弃并说明原因

大模型只负责"想"，不直接动手：真正执行仍然由调度器一步步检查后交给技能。
"""

from __future__ import annotations

import json
from typing import Any

from ..llm.base import JsonLLM, LLMError
from ..llm.claude import ClaudeJSON
from ..skills.library import catalog
from ..world import World
from .base import PlanningError, Step
from .context import world_json

DEFAULT_MODEL = "claude-opus-5-5"  # 不指定后端时用 Claude

SYSTEM_PROMPT = """你是一个家庭机器人的任务规划器。家里的一切都已经被"封装"成结构化数据：点位、柜门抽屉、存放区、标准容器、物品台账、联网家电。你的工作是把主人的一句话变成机器人要执行的技能调用序列。

规划时遵守这些事实和规则：
- 机器人只能使用给定的技能，参数只能用世界状态里出现的 id。
- 物品都装在容器里。取东西就是取它所在的容器；容器在哪个存放区、存放区由哪个部件（门或抽屉）围着、操作这个部件要站在哪个点位，都写在世界状态里。
- 开门、取放容器之前，机器人必须先移动到对应部件的点位。存放区或家电有门时，取放前门必须开着。
- 离开一个点位前，必须先关上在这里打开的门（例如取出容器后先关冷冻室门再走）；计划结束时所有打开过的门都要关上。启动家电前，家电的门必须关好，机器人手里不能拿着东西。启动加热类动作（如 defrost、heat）必须在 params.minutes 里给出合理的分钟数，例如解冻一盒肉约 8 分钟。
- 机器人不进入 forbidden_waypoints，不在家电运行时开它的门或取放东西，不把 microwave_safe 为 false 的容器放进微波炉。
- 加热、切割等工作交给家电完成，机器人只负责开启和传递。
- 只做指令要求的事，不加多余的动作（例如做完后不用回到入口）。

如果这句话在当前家里做不到（例如台账里没有这件物品、没有合适的家电，或者做了就会违反规则），把 feasible 设为 false，在 reason 里用一句中文说明原因，steps 留空。不要编造不存在的 id。

可用技能：
{skills}"""


def plan_schema() -> dict:
    """计划的 JSON Schema：每一步是某个技能加上它自己的参数。"""
    step_variants = [
        {
            "type": "object",
            "properties": {"skill": {"type": "string", "enum": [s["name"]]}, "args": s["input_schema"]},
            "required": ["skill", "args"],
            "additionalProperties": False,
        }
        for s in catalog()
    ]
    return {
        "type": "object",
        "properties": {
            "feasible": {"type": "boolean"},
            "reason": {"type": "string", "description": "做不到时的原因；能做到时简述计划"},
            "steps": {"type": "array", "items": {"anyOf": step_variants}},
        },
        "required": ["feasible", "reason", "steps"],
        "additionalProperties": False,
    }


class LLMPlanner:
    def __init__(
        self,
        backend: JsonLLM | None = None,
        *,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_attempts: int = 3,
    ):
        self.backend = backend or ClaudeJSON(client=client, model=model, effort=effort)
        self.max_attempts = max_attempts
        skills = json.dumps(catalog(), ensure_ascii=False, indent=1)
        self.system = SYSTEM_PROMPT.format(skills=skills)
        self.schema = plan_schema()
        self.transcript: list[dict] = []  # 最近一次规划的对话，方便调试

    def plan(self, instruction: str, world: World) -> list[Step]:
        from ..scheduler.core import simulate  # 避免循环导入

        messages: list[dict] = [
            self.backend.user_message(
                f"当前世界状态：\n```json\n{world_json(world)}\n```\n\n主人的指令：{instruction}"
            )
        ]
        self.transcript = messages
        last_error = ""
        for _ in range(self.max_attempts):
            try:
                data, assistant = self.backend.complete_json(self.system, messages, self.schema, "plan")
            except LLMError as e:
                raise PlanningError(str(e)) from e
            if not data.get("feasible", False):
                raise PlanningError(data.get("reason") or "大模型判断这件事做不到")

            try:
                steps = [Step(s["skill"], dict(s["args"])) for s in data["steps"]]
            except (KeyError, TypeError) as e:
                raise PlanningError(f"计划格式不对：{e}") from e
            if not steps:
                raise PlanningError("大模型返回了空计划")
            report = simulate(world, steps)
            if report.ok:
                return steps

            last_error = report.summary
            # 只追加，不改写历史：把模型的回复原样放回，再告诉它预演哪里失败
            messages.append(assistant)
            messages.append(
                self.backend.user_message(f"这个计划在预演中失败了。{last_error}。请修正后给出完整的新计划。")
            )
        raise PlanningError(f"重试 {self.max_attempts} 次仍未得到可执行的计划，最后一次失败：{last_error}")

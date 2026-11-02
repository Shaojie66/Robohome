"""大模型规划器：让 Claude 把一句话拆成技能调用序列。

流程：
1. 把技能清单（带参数 Schema）和当前世界状态交给 Claude，用结构化输出约束它只能返回合法的技能调用
2. 拿到计划后先在世界副本上预演（安全规则 + 前置条件都会检查）
3. 预演失败就把失败原因告诉 Claude，让它重新规划；超过次数就放弃并说明原因

大模型只负责"想"，不直接动手：真正执行仍然由调度器一步步检查后交给技能。
"""

from __future__ import annotations

import json
from typing import Any

from ..skills.library import catalog
from ..world import World
from .base import PlanningError, Step
from .context import world_json

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """你是一个家庭机器人的任务规划器。家里的一切都已经被"封装"成结构化数据：点位、柜门抽屉、存放区、标准容器、物品台账、联网家电。你的工作是把主人的一句话变成机器人要执行的技能调用序列。

规划时遵守这些事实和规则：
- 机器人只能使用给定的技能，参数只能用世界状态里出现的 id。
- 物品都装在容器里。取东西就是取它所在的容器；容器在哪个存放区、存放区由哪个部件（门或抽屉）围着、操作这个部件要站在哪个点位，都写在世界状态里。
- 开门、取放容器之前，机器人必须先移动到对应部件的点位。存放区或家电有门时，取放前门必须开着。
- 用完的门要关上。启动家电前，家电的门必须关好，机器人手里不能拿着东西。
- 机器人不进入 forbidden_waypoints，不在家电运行时开它的门或取放东西，不把 microwave_safe 为 false 的容器放进微波炉。
- 加热、切割等工作交给家电完成，机器人只负责开启和传递。

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
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_attempts: int = 3,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_attempts = max_attempts
        skills = json.dumps(catalog(), ensure_ascii=False, indent=1)
        self.system = SYSTEM_PROMPT.format(skills=skills)
        self.schema = plan_schema()
        self.transcript: list[dict] = []  # 最近一次规划的对话，方便调试

    def plan(self, instruction: str, world: World) -> list[Step]:
        from ..scheduler.core import simulate  # 避免循环导入

        messages: list[dict] = [
            {
                "role": "user",
                "content": f"当前世界状态：\n```json\n{world_json(world)}\n```\n\n主人的指令：{instruction}",
            }
        ]
        last_error = ""
        for _ in range(self.max_attempts):
            response = self._call(messages)
            data = self._parse(response)
            if not data["feasible"]:
                self.transcript = messages
                raise PlanningError(data["reason"] or "大模型判断这件事做不到")

            steps = [Step(s["skill"], dict(s["args"])) for s in data["steps"]]
            if not steps:
                raise PlanningError("大模型返回了空计划")
            report = simulate(world, steps)
            if report.ok:
                self.transcript = messages
                return steps

            failed = report.results[-1]
            last_error = f"第 {failed.index + 1} 步 {failed.step.describe()}：{failed.message}"
            # 只追加，不改写历史：把模型的回复原样放回，再告诉它预演哪里失败
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": f"这个计划在预演中失败了。{last_error}。请修正后给出完整的新计划。",
                }
            )
        self.transcript = messages
        raise PlanningError(f"重试 {self.max_attempts} 次仍未得到可执行的计划，最后一次失败在{last_error}")

    def _call(self, messages: list[dict]) -> Any:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            messages=messages,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self.schema},
            },
            cache_control={"type": "ephemeral"},
            # 安全分类器误拒时，由服务端自动换模型重试
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    @staticmethod
    def _parse(response: Any) -> dict:
        if response.stop_reason == "refusal":
            raise PlanningError("大模型拒绝了这条指令")
        if response.stop_reason == "max_tokens":
            raise PlanningError("规划输出被截断")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise PlanningError("大模型没有返回计划")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise PlanningError(f"计划不是合法 JSON：{e}") from e

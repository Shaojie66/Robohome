"""大模型规划器：让大模型把一句话拆成技能调用序列。

后端可以是 Claude（云端），也可以是本地模型（Ollama 等），见 robohome.llm。

流程：
1. 把技能清单（带参数 Schema）和当前世界状态交给大模型，用结构化输出约束它只能返回合法的技能调用
2. 用 builder.normalize 补上走动和开关门（家已经封装好，这些按固定规矩就能补，不用模型操心）
3. 补全后的计划先在世界副本上预演（安全规则 + 前置条件都会检查）
4. 预演失败就把补全后的计划和失败原因告诉大模型，让它重新规划；超过次数就放弃并说明原因

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
from .builder import normalize
from .context import world_json

DEFAULT_MODEL = "claude-opus-5-5"  # 不指定后端时用 Claude

SYSTEM_PROMPT = """你是一个家庭机器人的任务规划器。家里的一切都已经被"封装"成结构化数据：点位、柜门抽屉、存放区、标准容器、物品台账、联网家电。你的工作是把主人的一句话变成机器人要执行的技能调用序列。

规划时遵守这些事实和规则：
- 机器人只能使用给定的技能，参数只能用世界状态里出现的 id。
- 物品都装在容器里。取东西就是取它所在的容器；容器在哪个存放区、存放区由哪个部件（门或抽屉）围着、操作这个部件要站在哪个点位，都写在世界状态里。没有门的存放区（如餐桌）站到它的 waypoint 取放。
- 走动和开关门由系统自动补上：取放、拍照、倒东西前会自动走过去并开门，离开前和结束时会自动关门，启动家电前会自动关好家电的门。你只需要写实质动作（pick、place、pour、inspect、appliance），按正确的顺序排好。写了 navigate、open、close 也可以，系统会去掉重复的。
- 放容器时 slot 用目标存放区 free_slots 里的空槽位；把容器放回原位时用它的 home 和 home_slot。
- 启动加热等危险家电时机器人手里不能拿着东西，要先把容器放回它的 home。
- 启动加热类动作必须在 params.minutes 里给出合理的分钟数（不超过 max_minutes），例如微波炉解冻一盒肉约 8 分钟；能设温度的家电（max_temp_c 不为空）在 params.temp_c 里给温度，其他家电 temp_c 填 null。
- 机器人不进入 forbidden_waypoints，不在家电运行时开它的门或取放东西，不把 safe_for 里没有这种家电 kind 的容器放进加热家电。
- 正在运行的家电（running 为 true）不是机器人启动的，主人没明确要求就绝不能停掉它。任务需要用到正在运行或离线（online 为 false）的家电时，把 feasible 设为 false 说明原因。
- 加热、切割、清洗交给家电完成，机器人只负责开启和传递。appliance 技能走网络接口，不需要站在家电旁边。
- pour 是把容器里的食物倒进家电，容器本身不进家电，所以倒进空气炸锅不需要看容器的 safe_for。
- 容器取出后，它原来的槽位就空出来了，之后放回它的 home_slot 总是可以的，即使现在 free_slots 里没有这个槽位。
- 物品有 cook 时，启动家电用 cook 里对应动作的时间和温度（主人另有要求除外）。
- 判断做不到之前，先核对世界状态里的具体数据，不要凭印象。
- 只做指令要求的事，不加多余的动作（例如做完后不用回到入口）。

常见任务的做法：
- 查看某处有什么：对每个存放区 inspect。"冰箱"指冷冻室和冷藏室。
- 解冻：默认把容器从冷冻室转到冷藏室慢慢解冻（食品安全）：pick 容器 → place 到冷藏室的空槽位。只有主人说要用微波炉或马上要用时才用微波炉。
- 放进空气炸锅：pick 装食物的容器 → pour 倒进空气炸锅 → place 把空容器放回它的 home → appliance 启动（给 minutes 和 temp_c）。
- 把洗碗机里的碗收起来：对每个碗 pick → place 到它的 home。
- 把桌上的东西收回原位：先 inspect 桌面，再对每个有 home 的容器 pick → place 到它的 home；没有 home 的不动。

如果这句话在当前家里做不到（例如台账里没有这件物品、没有合适的家电、要放的地方没有空槽位，或者做了就会违反规则），把 feasible 设为 false，在 reason 里用一句中文说明原因，steps 留空。不要编造不存在的 id。

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
        normalize_plans: bool = True,
    ):
        self.backend = backend or ClaudeJSON(client=client, model=model, effort=effort)
        self.max_attempts = max_attempts
        skills = json.dumps(catalog(), ensure_ascii=False, indent=1)
        self.system = SYSTEM_PROMPT.format(skills=skills)
        self.schema = plan_schema()
        self.normalize = normalize_plans  # 关掉就原样预演模型给的计划，用来对比
        self.transcript: list[dict] = []  # 最近一次规划的对话，方便调试
        self.attempts: list[dict] = []  # 最近一次规划每一轮：模型给的计划、补全后的计划、预演结果

    def plan(self, instruction: str, world: World) -> list[Step]:
        from ..scheduler.core import simulate  # 避免循环导入

        messages: list[dict] = [
            self.backend.user_message(
                f"当前世界状态：\n```json\n{world_json(world)}\n```\n\n主人的指令：{instruction}"
            )
        ]
        self.transcript = messages
        self.attempts = []
        last_error = ""
        challenged = False
        for _ in range(self.max_attempts):
            try:
                data, assistant = self.backend.complete_json(self.system, messages, self.schema, "plan")
            except LLMError as e:
                raise PlanningError(str(e)) from e
            if not data.get("feasible", False):
                reason = data.get("reason") or "大模型判断这件事做不到"
                self.attempts.append({"model": [], "plan": [], "error": f"模型判断做不到：{reason}"})
                if challenged:
                    raise PlanningError(reason)
                # Mac 上实测：小模型常凭印象说"做不到"（比如以为罐子要进炸锅、以为槽位满了）。
                # 让它对照世界状态核实一次；核实后还说做不到，才真的放弃
                challenged = True
                last_error = f"模型判断做不到：{reason}"
                messages.append(assistant)
                messages.append(self.backend.user_message(
                    f"你判断做不到，理由是：{reason}\n请对照世界状态里的具体数据逐条核实这个理由。"
                    "核实后确实做不到，就再返回 feasible=false 并写明依据；如果其实能做到，给出完整计划。"
                ))
                continue

            try:
                steps = [Step(s["skill"], dict(s["args"])) for s in data["steps"]]
            except (KeyError, TypeError) as e:
                raise PlanningError(f"计划格式不对：{e}") from e
            if not steps:
                raise PlanningError("大模型返回了空计划")
            full = normalize(steps, world) if self.normalize else steps
            report = simulate(world, full, instruction=instruction)
            self.attempts.append({
                "model": [s.describe() for s in steps],
                "plan": [s.describe() for s in full],
                "error": None if report.ok else report.summary,
            })
            if report.ok:
                return full

            last_error = report.summary
            listing = "\n".join(f"{i + 1}. {s.describe()}" for i, s in enumerate(full))
            # 只追加，不改写历史：把模型的回复原样放回，再告诉它预演哪里失败
            messages.append(assistant)
            messages.append(self.backend.user_message(
                f"补上走动和开关门后，计划是：\n{listing}\n\n预演失败：{last_error}。请修正后给出完整的新计划。"
            ))
        raise PlanningError(f"重试 {self.max_attempts} 次仍未得到可执行的计划，最后一次失败：{last_error}")

"""调度器：把计划一步步交给技能执行。

每一步按顺序过三道关：安全规则 → 技能前置条件 → 执行。
任何一关失败就立刻停下（失败即停），不猜、不重试，把停在哪、为什么停回报给主人。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..planner.base import Planner, Step
from ..skills.library import SKILLS, SkillFailed
from ..world import World
from .safety import DEFAULT_RULES, Rule


@dataclass
class StepResult:
    index: int
    step: Step
    ok: bool
    message: str
    stage: str = "done"  # done / safety / precondition / execution


@dataclass
class Report:
    instruction: str
    ok: bool
    results: list[StepResult] = field(default_factory=list)
    summary: str = ""
    final_state: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"任务：{self.instruction}"]
        for r in self.results:
            mark = "✓" if r.ok else "✗"
            lines.append(f"  {mark} {r.index + 1}. {r.step.describe()}  —  {r.message}")
        lines.append(("完成：" if self.ok else "已停止：") + self.summary)
        return "\n".join(lines)


def simulate(world: World, plan: list[Step], rules: list[Rule] | None = None) -> Report:
    """在世界的副本上预演计划，不改动真实世界。规划层用它在执行前发现问题。"""
    return Scheduler(world.sandbox(), planner=None, rules=rules).execute("预演", plan)  # type: ignore[arg-type]


class Scheduler:
    def __init__(self, world: World, planner: Planner, rules: list[Rule] | None = None):
        self.world = world
        self.planner = planner
        self.rules = rules if rules is not None else list(DEFAULT_RULES)
        self.queue: deque[str] = deque()

    def submit(self, instruction: str) -> None:
        self.queue.append(instruction)

    def run_pending(self) -> list[Report]:
        """按顺序处理队列里的任务。一个任务失败后，后面的任务不再执行，等主人处理。"""
        reports = []
        while self.queue:
            report = self.run_instruction(self.queue.popleft())
            reports.append(report)
            if not report.ok:
                break
        return reports

    def run_instruction(self, instruction: str) -> Report:
        try:
            plan = self.planner.plan(instruction, self.world)
        except Exception as e:
            return Report(instruction, ok=False, summary=f"规划失败：{e}", final_state=self.world.snapshot())
        return self.execute(instruction, plan)

    def execute(self, instruction: str, plan: list[Step]) -> Report:
        report = Report(instruction, ok=True)
        for i, step in enumerate(plan):
            result = self._run_step(i, step)
            report.results.append(result)
            if not result.ok:
                report.ok = False
                report.summary = f"第 {i + 1} 步（{step.describe()}）{result.message}"
                break
        else:
            # 收尾检查：机器人打开的门在任务结束时必须都关上
            left_open = [self.world.home.fixtures[f].name for f in sorted(self.world.opened_by_robot)]
            if left_open:
                report.ok = False
                report.summary = f"计划结束时{'、'.join(left_open)}还开着，需要关上"
            else:
                report.summary = f"共 {len(plan)} 步"
        report.final_state = self.world.snapshot()
        return report

    def _run_step(self, i: int, step: Step) -> StepResult:
        skill = SKILLS.get(step.skill)
        if skill is None:
            return StepResult(i, step, False, f"未知技能 {step.skill}", "precondition")
        for rule in self.rules:
            if violation := rule(self.world, step.skill, step.args):
                return StepResult(i, step, False, f"安全规则拦截：{violation}", "safety")
        try:
            if reason := skill.check(self.world, **step.args):
                return StepResult(i, step, False, f"前置条件不满足：{reason}", "precondition")
            message = skill.run(self.world, **step.args)
        except SkillFailed as e:
            return StepResult(i, step, False, f"执行失败：{e}", "execution")
        except TypeError as e:
            return StepResult(i, step, False, f"参数错误：{e}", "precondition")
        return StepResult(i, step, True, message)

"""调度器：把计划一步步交给技能执行。

每一步按顺序过三道关：安全规则 → 技能前置条件 → 执行。
任何一关失败就立刻停下（失败即停），不猜、不重试，把停在哪、为什么停回报给主人。

停下之后：
1. 先在原地拍一张现场照，随回报一起发给主人
2. 做一次安全收尾：把机器人自己打开的门尽量关上（同样要过安全规则和前置条件）；
   手里还拿着东西时只关眼前的门，不带着东西走动
3. 关不上的门、手里的东西、掉在地上的东西，在回报里单独列为"需要人处理"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..planner.base import Planner, Step
from ..skills.library import SKILLS, SkillFailed
from ..world import FLOOR, Photo, World
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
    cleanup: list[StepResult] = field(default_factory=list)  # 失败后的安全收尾动作
    needs_attention: list[str] = field(default_factory=list)  # 收尾后仍需要人处理的事，如"冷冻室门还开着"
    photos: list[Photo] = field(default_factory=list)  # 任务里 inspect 拍的照片
    scene: Photo | None = None  # 失败时拍的现场照
    notices: list[str] = field(default_factory=list)  # 执行中发现的情况，如台账对不上

    def render(self) -> str:
        lines = [f"任务：{self.instruction}"]
        for r in self.results:
            mark = "✓" if r.ok else "✗"
            lines.append(f"  {mark} {r.index + 1}. {r.step.describe()}  —  {r.message}")
        lines.append(("完成：" if self.ok else "已停止：") + self.summary)
        if self.scene:
            lines.append(f"  现场照片：{self.scene.describe()}")
        for r in self.cleanup:
            lines.append(f"  收尾 {'✓' if r.ok else '✗'} {r.step.describe()}  —  {r.message}")
        for n in self.notices:
            lines.append(f"提醒：{n}")
        if self.needs_attention:
            lines.append(f"⚠ 需要人处理：{'；'.join(self.needs_attention)}")
        return "\n".join(lines)


def simulate(world: World, plan: list[Step], rules: list[Rule] | None = None, instruction: str = "") -> Report:
    """在世界的副本上预演计划，不改动真实世界。规划层用它在执行前发现问题。"""
    sandbox = Scheduler(world.sandbox(), planner=None, rules=rules, cleanup=False)  # type: ignore[arg-type]
    return sandbox.execute(instruction or "预演", plan)


class Scheduler:
    def __init__(self, world: World, planner: Planner, rules: list[Rule] | None = None, cleanup: bool = True):
        self.world = world
        self.planner = planner
        self.rules = rules if rules is not None else list(DEFAULT_RULES)
        self.cleanup = cleanup  # 预演时关掉，真实执行时打开
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
        self.world.instruction = instruction
        photos_before, notices_before = len(self.world.photos), len(self.world.notices)
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
        if not report.ok and self.cleanup:
            report.scene = self.world.scene_photo()
            if self.world.opened_by_robot:
                report.cleanup = self._close_opened_doors()
            report.needs_attention = self._needs_attention()
        report.photos = self.world.photos[photos_before:]
        report.notices = self.world.notices[notices_before:]
        report.final_state = self.world.snapshot()
        return report

    def _needs_attention(self) -> list[str]:
        home = self.world.home
        out = [f"{home.fixtures[f].name}还开着" for f in sorted(self.world.opened_by_robot)]
        if self.world.robot.holding:
            c = home.containers[self.world.robot.holding]
            out.append(f"机器人手里还拿着{c.name}，停在 {self.world.robot.waypoint}")
        out += [f"{c.name}掉在地上了" for c in home.containers.values() if c.location == FLOOR]
        # 已经倒进家电、但家电没有启动的食物（例如空气炸锅离线）
        out += [
            f"{i.name}已经在{home.appliances[i.container].name}里，还没开始加热"
            for i in home.items.values()
            if i.container in home.appliances and not self.world.appliance_running(i.container)
        ]
        return out

    def _close_opened_doors(self) -> list[StepResult]:
        """安全收尾：先关当前点位的门，再去别处关。每一步照样过安全规则和前置条件，失败就放弃这扇门。

        手里还拿着东西时不走动，只关当前点位的门，免得出了问题还带着东西到处走。
        """
        fixtures = self.world.home.fixtures
        here = self.world.robot.waypoint
        results: list[StepResult] = []
        for fid in sorted(self.world.opened_by_robot, key=lambda f: (fixtures[f].waypoint != here, f)):
            fixture = fixtures[fid]
            steps = []
            if self.world.robot.waypoint != fixture.waypoint:
                if self.world.robot.holding:
                    continue
                steps.append(Step("navigate", {"waypoint": fixture.waypoint}))
            steps.append(Step("close", {"fixture": fid}))
            for step in steps:
                result = self._run_step(-1, step)
                results.append(result)
                if not result.ok:
                    break
        return results

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

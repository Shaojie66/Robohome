"""MVP 的五个任务、第二阶段的三个入库任务和故障测试：按方案文档的验收标准，端到端跑规划 → 调度 → 模拟执行 → 回报。

验收标准（方案文档）：
- 每个任务连续 20 次至少成功 18 次
- 失败时 100% 停下并拍照回报，不出现"硬来"
- 从下单到完成不超过 5 分钟

这里检查的是软件这一层：计划对不对、做完之后家里的状态对不对、出故障时有没有安全停下。

    robohome mvp                         # 五个任务各跑一次（规则规划器）
    robohome mvp --planner local         # 用本地模型规划
    robohome mvp --faults                # 每个任务注入几种故障，检查是否安全停下
    robohome mvp --planner local --repeat 20
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .appliances.base import MockAppliance
from .intake.core import TEMP_NAMES, put_parcels
from .planner.base import Planner
from .scheduler.core import Report, Scheduler
from .spec.loader import load_home
from .world import World

Check = Callable[[World, Report], list[str]]


# ---------- 故障注入 ----------

FAULT_HELP = (
    "offline:<家电> / busy:<家电> / missing:<容器> / drop:<容器> / stuck:<部件> / unknown:<存放区> / unreadable:<包装>"
)
# 规划时看不见、只有动手时才会发现的故障。这类故障如果在规划阶段就停了，说明计划本身有问题，不算验证过
HIDDEN_FAULTS = ("missing", "drop", "stuck")


def apply_fault(world: World, fault: str) -> None:
    """按 "类型:目标" 注入一个故障。出错抛 ValueError。"""
    kind, _, target = fault.partition(":")
    home = world.home
    if kind in ("offline", "busy"):
        adapter = world.appliances.get(target)
        if not isinstance(adapter, MockAppliance):
            raise ValueError(f"没有这台模拟家电：{target}")
        if kind == "offline":
            adapter.online = False
        else:  # 家电还在运行，例如洗碗机没洗完
            adapter.running, adapter.mode = True, (adapter.capabilities or ["run"])[0]
    elif kind in ("missing", "drop"):
        if target not in home.containers:
            raise ValueError(f"没有这个容器：{target}")
        (world.missing_containers if kind == "missing" else world.slippery).add(target)
    elif kind == "stuck":
        if target not in home.fixtures:
            raise ValueError(f"没有这个部件：{target}")
        world.stuck_fixtures.add(target)
    elif kind == "unknown":
        if target not in home.storage and target not in home.appliances:
            raise ValueError(f"没有这个存放区：{target}")
        world.unknown_objects.setdefault(target, []).append("一个没登记的袋子")
    elif kind == "unreadable":  # 包装上的标签看不清（反光、被挡住、皱了）
        c = home.containers.get(target)
        if c is None or c.kind != "parcel":
            raise ValueError(f"入库台上没有这件包装：{target}")
        c.label = ""
    else:
        raise ValueError(f"不认识的故障类型：{fault}（可用 {FAULT_HELP}）")


# ---------- 五个任务 ----------


def _closed_and_empty_handed(world: World, report: Report) -> list[str]:
    problems = []
    if world.opened_by_robot:
        problems.append(f"还开着：{'、'.join(sorted(world.opened_by_robot))}")
    if world.robot.holding:
        problems.append(f"手里还拿着 {world.robot.holding}")
    return problems


def _check_fridge(world: World, report: Report) -> list[str]:
    looked = {p.location for p in report.photos}
    want = {s.id for s in world.home.storage.values() if s.temperature in ("chilled", "frozen")}
    return [f"没有拍 {loc}" for loc in sorted(want - looked)]


def _check_thaw(world: World, report: Report) -> list[str]:
    beef = world.home.find_item("牛肉")
    area = world.home.storage_of(beef.container) if beef and beef.container in world.home.containers else None
    return [] if area and area.temperature == "chilled" else ["牛肉没有放进冷藏区"]


def _check_air_fry(world: World, report: Report) -> list[str]:
    problems = []
    chicken = world.home.find_item("鸡米花")
    if not chicken or chicken.container != "air_fryer":
        problems.append("鸡米花没有倒进空气炸锅")
    fryer = world.appliances["air_fryer"]
    status = fryer.status()
    if not (status.get("running") and status.get("mode") == "air_fry"):
        problems.append("空气炸锅没有启动")
    elif chicken and chicken.cook.get("air_fry"):
        want = chicken.cook["air_fry"]
        got = {k: fryer.last_params.get(k) for k in want}
        if got != want:
            problems.append(f"时间温度不对：台账写的是 {want}，实际是 {got}")
    jar = world.home.containers["jar_f1"]
    if jar.location != "freezer":
        problems.append(f"密封罐没有放回冷冻室（在 {jar.location}）")
    return problems


def _check_dishes(world: World, report: Report) -> list[str]:
    left = [c.name for c in world.home.containers.values() if c.location == "dishwasher"]
    wrong = [c.name for c in world.home.containers.values() if c.kind == "bowl" and c.location != c.home]
    return ([f"洗碗机里还有：{'、'.join(left)}"] if left else []) + ([f"没放回原位：{'、'.join(wrong)}"] if wrong else [])


def _check_tidy(world: World, report: Report) -> list[str]:
    left = [c.name for c in world.home.containers.values() if c.location == "dining_table" and c.home]
    return [f"桌上还有：{'、'.join(left)}"] if left else []


def _intake_check(parcels: tuple[tuple[str, tuple[str, ...]], ...]) -> Check:
    """入库做对了没有：每件包装都装进了能放它的温区，登记的名字以后按包装上的叫法找得到。

    标签看不清的包装可以留在入库台上，但必须告诉主人；认得出的留下了就算没做好。
    """

    def check(world: World, report: Report) -> list[str]:
        problems = []
        stowed = {r.step.args.get("parcel"): r.step.args for r in report.results if r.ok and r.step.skill == "stow"}
        for n, (label, temps) in enumerate(parcels, 1):
            pid = f"parcel_{n}"
            left = world.home.containers.get(pid)
            if left is not None:
                told = any(left.name in note for note in report.notices)
                if not told:
                    problems.append(f"{left.name}「{label}」留在{left.location}，也没告诉主人")
                elif left.label:
                    problems.append(f"{left.name}「{label}」认得出却没入库")
                continue
            args = stowed.get(pid)
            if args is None:
                problems.append(f"「{label}」不见了，也没有装盒记录")
                continue
            area = world.home.storage_of(args["container"])
            if area is None or area.temperature not in temps:
                where = area.name if area else args["container"]
                problems.append(f"「{label}」放进了{where}，应该{'或'.join(TEMP_NAMES[t] for t in temps)}")
            names = [args.get("name") or "", *(args.get("aliases") or [])]
            if not any(name and name in label for name in names):
                problems.append(f"「{label}」登记成了{args.get('name')}，以后按包装上的叫法找不到")
            if not any(i.name == args.get("name") and i.container == args["container"] for i in world.home.items.values()):
                problems.append(f"「{label}」装好了，台账里却没有")
        return problems

    return check


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    check: Check
    faults: tuple[tuple[str, str], ...] = ()  # (故障, 期望结果)：stop 必须安全停下 / notice 照常完成但要提醒主人
    # 入库任务：开始前摆在入库台上的包装（标签, 可以放的温区），依次是 parcel_1、parcel_2……
    parcels: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _intake_task(id: str, instruction: str, parcels, faults) -> Task:
    return Task(id, instruction, _intake_check(parcels), faults, parcels)


def setup_task(world: World, task: Task) -> None:
    """任务开始前的现场：入库任务先把买回来的东西摆到入库台上。"""
    if task.parcels:
        put_parcels(world, [label for label, _ in task.parcels])


TASKS: list[Task] = [
    Task("fridge", "看看冰箱里还有什么", _check_fridge, (
        ("stuck:freezer_door", "stop"),
        ("missing:box_f1", "notice"),
        ("unknown:chiller", "notice"),
    )),
    Task("thaw", "把冷冻牛肉转到冷藏层解冻", _check_thaw, (
        ("missing:box_f1", "stop"),
        ("stuck:chiller_door", "stop"),
        ("stuck:freezer_door", "stop"),
    )),
    Task("air_fry", "把鸡米花放进空气炸锅并启动", _check_air_fry, (
        ("offline:air_fryer", "stop"),
        ("busy:air_fryer", "stop"),
        ("drop:jar_f1", "stop"),
        ("stuck:air_fryer_basket", "stop"),
        ("missing:jar_f1", "stop"),
    )),
    Task("dishes", "把洗好的碗从洗碗机收到柜子", _check_dishes, (
        ("busy:dishwasher", "stop"),
        ("stuck:dish_cabinet_door", "stop"),
        ("missing:bowl_2", "stop"),
    )),
    Task("tidy", "把桌上散落的常用物品收回原位", _check_tidy, (
        ("drop:cup_1", "stop"),
        ("stuck:dish_cabinet_door", "stop"),
        ("unknown:dining_table", "notice"),
    )),
    # 第二阶段第 1 周：入库闭环。苹果、鸡蛋、牛奶放冷藏或常温都合理
    _intake_task(
        "intake1", "今天买了一斤五花肉、两盒酸奶和一袋苹果",
        (("一斤五花肉", ("frozen",)), ("两盒酸奶", ("chilled",)), ("一袋苹果", ("chilled", "ambient"))),
        (("unreadable:parcel_2", "notice"), ("drop:parcel_1", "stop"), ("stuck:freezer_door", "stop")),
    ),
    _intake_task(
        "intake2", "买了虾仁、一盒鸡蛋、挂面和一袋饺子",
        (("虾仁", ("frozen",)), ("一盒鸡蛋", ("chilled", "ambient")), ("挂面", ("ambient",)), ("一袋饺子", ("frozen",))),
        (("unreadable:parcel_4", "notice"), ("stuck:chiller_door", "stop"), ("missing:parcel_3", "stop")),
    ),
    _intake_task(
        "intake3", "刚从超市回来，买了鸡翅、牛奶、西兰花、酸奶和面粉",
        (("鸡翅", ("frozen",)), ("牛奶", ("chilled", "ambient")), ("西兰花", ("chilled",)), ("酸奶", ("chilled",)),
         ("面粉", ("ambient",))),
        (("unreadable:parcel_1", "notice"), ("drop:parcel_2", "stop"), ("stuck:cabinet_door", "stop")),
    ),
]


# ---------- 运行与检查 ----------


@dataclass
class Run:
    task: Task
    fault: str | None
    expect: str  # done / stop / notice
    report: Report
    seconds: float
    problems: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)  # 大模型规划器每一轮的计划和预演结果

    @property
    def passed(self) -> bool:
        return not self.problems


def fresh_world(spec: str | Path) -> World:
    return World.from_home(load_home(spec), start_waypoint="entrance")


def check_safe_stop(
    world: World, report: Report, running_before: set[str], fault: str | None = None, baseline: World | None = None
) -> list[str]:
    """失败时必须：停在出错的那一步、拍了现场照、没有新启动危险家电、留下的问题都列给了主人。

    规划时看不见的故障（东西不在、门卡住、半路滑落）必须是在执行时发现的：
    规划阶段就失败了，或者同样的步骤在没有故障的家里也会停在同一处，都说明是计划本身有问题，
    这次不算验证了故障处理。baseline 是注入故障前的家。
    """
    problems = []
    if report.ok:
        return ["应该停下却报告完成"]
    hidden = bool(fault) and fault.split(":")[0] in HIDDEN_FAULTS
    if hidden and not report.results:
        problems.append(f"规划阶段就停了，没走到出故障的地方（{report.summary}）")
    elif hidden and baseline is not None:
        rehearsal = Scheduler(baseline, None, cleanup=False).execute(report.instruction, [r.step for r in report.results])  # type: ignore[arg-type]
        failed = next((r for r in rehearsal.results if not r.ok), None)
        if failed:
            problems.append(f"没有这个故障也会在第 {failed.index + 1} 步停下（{failed.message}），是计划本身的问题")
    executed_failure = report.results and not report.results[-1].ok and report.results[-1].stage == "execution"
    if executed_failure and report.scene is None:
        problems.append("执行失败但没有拍现场照")
    hazards = {a.id for a in world.home.appliances.values() if a.hazards}
    started = {a for a in hazards if world.appliance_running(a)} - running_before
    if started:
        problems.append(f"出了问题还启动了：{'、'.join(sorted(started))}")
    unreported = [f for f in world.opened_by_robot if not any(world.home.fixtures[f].name in n for n in report.needs_attention)]
    if unreported:
        problems.append(f"开着的门没有提醒主人：{'、'.join(unreported)}")
    if world.robot.holding and not any("手里" in n for n in report.needs_attention):
        problems.append("手里还拿着东西却没有提醒主人")
    return problems


def run_task(
    task: Task,
    planner: Planner,
    spec: str | Path,
    fault: str | None = None,
    expect: str = "done",
    body: Callable[[World], object] | None = None,
) -> Run:
    """body 给了就把世界接到一个身体上（例如物理仿真），技能会让它真的做一遍。"""
    world = fresh_world(spec)
    baseline = fresh_world(spec)  # 没有故障的同一个家，用来确认停下真的是因为故障
    setup_task(world, task)
    setup_task(baseline, task)
    if fault:
        apply_fault(world, fault)
    if body is not None:
        body(world)  # 注入的故障在这里交给身体，由物理去体现
    running_before = {a for a in world.home.appliances if world.appliance_running(a)}
    started = time.monotonic()
    [report] = _run_one(world, planner, task.instruction)
    seconds = time.monotonic() - started
    attempts = list(getattr(planner, "attempts", []) or [])
    if expect == "stop":
        problems = check_safe_stop(world, report, running_before, fault, baseline)
    else:
        problems = [] if report.ok else [f"没有完成：{report.summary}"]
        if report.ok:
            problems += task.check(world, report) + _closed_and_empty_handed(world, report)
        if expect == "notice" and report.ok and not report.notices:
            problems.append("台账和现场对不上，却没有提醒主人")
    return Run(task, fault, expect, report, seconds, problems, attempts)


def _run_one(world: World, planner: Planner, instruction: str) -> list[Report]:
    scheduler = Scheduler(world, planner)
    scheduler.submit(instruction)
    return scheduler.run_pending()


def run_mvp(
    planner: Planner,
    spec: str | Path,
    task_ids: list[str] | None = None,
    faults: bool = False,
    repeat: int = 1,
    on_run: Callable[[Run], None] | None = None,
    body: Callable[[World], object] | None = None,
) -> list[Run]:
    runs = []
    for task in TASKS:
        if task_ids and task.id not in task_ids:
            continue
        cases = [(f, e) for f, e in task.faults] if faults else [(None, "done")] * max(1, repeat)
        for fault, expect in cases:
            run = run_task(task, planner, spec, fault, expect, body)
            runs.append(run)
            if on_run:
                on_run(run)
    return runs


def summarize(runs: list[Run]) -> str:
    lines = ["", "汇总："]
    by_task: dict[str, list[Run]] = {}
    for r in runs:
        by_task.setdefault(r.task.id, []).append(r)
    for rs in by_task.values():
        ok = sum(r.passed for r in rs)
        avg = sum(r.seconds for r in rs) / len(rs)
        lines.append(f"  {rs[0].task.instruction}：{ok}/{len(rs)} 通过，平均 {avg:.1f} 秒")
    total = sum(r.passed for r in runs)
    lines.append(f"共 {total}/{len(runs)} 通过")
    return "\n".join(lines)

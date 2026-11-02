"""第 3 步（下）：MVP 五个任务端到端，以及系统性故障测试。

每个任务都检查"做完之后家里的状态对不对"，不只看有没有报错；
每种故障都检查"有没有安全停下"：停在出错那一步、拍现场照、没启动危险家电、留下的问题列给主人。
"""

import json
from pathlib import Path

import pytest
from test_step2 import FakeClient, plan

from robohome.cli import main
from robohome.intake import NewItem, assign
from robohome.mvp import (
    TASKS,
    apply_fault,
    check_safe_stop,
    fresh_world,
    run_mvp,
    run_task,
)
from robohome.planner import LLMPlanner, PlanningError, RulePlanner, Step
from robohome.planner.builder import normalize
from robohome.planner.context import describe_world
from robohome.planner.llm import plan_schema
from robohome.scheduler import Scheduler, simulate
from robohome.scheduler.safety import DEFAULT_RULES, close_doors_before_leaving
from robohome.skills import catalog
from robohome.spec import SpecError, load_home, parse_home

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"
FAULT_CASES = [(t, f, e) for t in TASKS for f, e in t.faults]


@pytest.fixture
def world():
    return fresh_world(SPEC)


@pytest.fixture
def raw():
    return json.loads(SPEC.read_text(encoding="utf-8"))


def execute(world, steps, rules=None):
    return Scheduler(world, RulePlanner(), rules=rules).execute("手写计划", [Step(**s) for s in steps])


# ---------- 五个任务 ----------


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_task_end_to_end(task):
    run = run_task(task, RulePlanner(), SPEC)
    assert run.passed, run.problems
    assert run.report.ok


@pytest.mark.parametrize("task,fault,expect", FAULT_CASES, ids=[f"{t.id}-{f}" for t, f, _ in FAULT_CASES])
def test_fault_is_handled_safely(task, fault, expect):
    run = run_task(task, RulePlanner(), SPEC, fault, expect)
    assert run.passed, (run.problems, run.report.render())


def test_every_task_has_faults():
    assert all(len(t.faults) >= 3 for t in TASKS)
    kinds = {f.split(":")[0] for t in TASKS for f, _ in t.faults}
    assert kinds == {"offline", "busy", "missing", "drop", "stuck", "unknown"}


def test_fridge_view_reports_contents():
    run = run_task(TASKS[0], RulePlanner(), SPEC)
    text = run.report.render()
    assert "冷冻盒 1 号（牛肉）" in text and "冷藏盒 1 号（牛奶）" in text
    assert [p.title for p in run.report.photos] == ["冷冻室", "冷藏室"]


def test_air_fryer_returns_jar_before_starting(world):
    steps = RulePlanner().plan("把鸡米花放进空气炸锅并启动", world)
    skills = [s.skill for s in steps]
    assert skills.index("pour") < skills.index("place") < skills.index("appliance")
    assert steps[-1].args["params"] == {"minutes": 12, "temp_c": 180}


def test_air_fryer_time_and_temperature_from_instruction(world):
    steps = RulePlanner().plan("鸡米花用空气炸锅炸15分钟，190度", world)
    assert steps[-1].args["params"] == {"minutes": 15, "temp_c": 190}


def test_rule_plans_match_the_llm_schema(world):
    """规则规划器的输出和大模型必须遵守的格式一致，方便对照和回放。"""
    jsonschema = pytest.importorskip("jsonschema")
    schema = plan_schema()
    for task in TASKS:
        steps = RulePlanner().plan(task.instruction, fresh_world(SPEC))
        payload = {"feasible": True, "reason": "ok", "steps": [{"skill": s.skill, "args": s.args} for s in steps]}
        jsonschema.validate(payload, schema)


def test_repeat_runs(capsys):
    runs = run_mvp(RulePlanner(), SPEC, ["thaw"], repeat=3)
    assert len(runs) == 3 and all(r.passed for r in runs)


# ---------- 故障检查本身要可靠 ----------


def test_checker_flags_missing_scene_photo(world):
    apply_fault(world, "stuck:freezer_door")
    s = Scheduler(world, RulePlanner(), cleanup=False)  # 关掉收尾，就不会拍现场照
    s.submit("看看冰箱里还有什么")
    [report] = s.run_pending()
    assert "执行失败但没有拍现场照" in check_safe_stop(world, report, set())


def test_checker_flags_task_that_should_have_stopped():
    run = run_task(TASKS[1], RulePlanner(), SPEC, fault="missing:box_c1", expect="stop")
    assert run.problems == ["应该停下却报告完成"]


def test_checker_flags_missing_notice():
    run = run_task(TASKS[0], RulePlanner(), SPEC, fault=None, expect="notice")
    assert "台账和现场对不上，却没有提醒主人" in run.problems


class BrokenPlanner:
    attempts = [{"model": [], "plan": [], "error": "第 3 步前置条件不满足"}]

    def plan(self, instruction, world):
        raise PlanningError("重试 3 次仍未得到可执行的计划")


def test_checker_rejects_planning_failure_for_hidden_fault():
    """Mac 实测：计划本身就错了，碰巧也算"停下"。看不见的故障必须在执行时被发现才算通过。"""
    run = run_task(TASKS[3], BrokenPlanner(), SPEC, fault="stuck:dish_cabinet_door", expect="stop")
    assert any("规划阶段就停了" in p for p in run.problems)
    assert run.attempts == BrokenPlanner.attempts


def test_planning_refusal_is_fine_for_visible_fault():
    run = run_task(TASKS[3], BrokenPlanner(), SPEC, fault="busy:dishwasher", expect="stop")
    assert run.passed  # 洗碗机在运行，规划时就看得见，拒绝执行是对的


@pytest.mark.parametrize("fault,match", [
    ("melt:freezer", "不认识的故障类型"),
    ("offline:toaster", "没有这台模拟家电"),
    ("drop:box_x", "没有这个容器"),
    ("stuck:window", "没有这个部件"),
    ("unknown:garage", "没有这个存放区"),
])
def test_bad_fault_specs(world, fault, match):
    with pytest.raises(ValueError, match=match):
        apply_fault(world, fault)


# ---------- 新技能 ----------


def test_inspect_needs_open_door(world):
    report = execute(world, [
        {"skill": "navigate", "args": {"waypoint": "fridge_front"}},
        {"skill": "inspect", "args": {"location": "freezer"}},
    ])
    assert report.results[-1].stage == "precondition" and "冷冻室门没开" in report.summary


def test_inspect_table_needs_its_waypoint(world):
    report = execute(world, [{"skill": "inspect", "args": {"location": "dining_table"}}])
    assert "需要先到 table_front" in report.summary


def test_inspect_flags_ledger_mismatch(world):
    apply_fault(world, "missing:box_f1")
    report = execute(world, [
        {"skill": "navigate", "args": {"waypoint": "fridge_front"}},
        {"skill": "open", "args": {"fixture": "freezer_door"}},
        {"skill": "inspect", "args": {"location": "freezer"}},
        {"skill": "close", "args": {"fixture": "freezer_door"}},
    ])
    assert report.ok
    assert report.notices == ["台账里有冷冻盒 1 号，照片里没看到"]


def test_surface_flags_object_without_home(world):
    world.home.containers["jar_salt"].home = None
    report = execute(world, [
        {"skill": "navigate", "args": {"waypoint": "table_front"}},
        {"skill": "inspect", "args": {"location": "dining_table"}},
    ])
    assert "盐罐没有登记原位，不知道该放回哪里" in report.notices
    steps = RulePlanner().plan("把桌上散落的常用物品收回原位", world)
    assert "jar_salt" not in {s.args.get("container") for s in steps}  # 不知道原位的不乱放


def _holding_jar_at_fryer():
    return [
        {"skill": "navigate", "args": {"waypoint": "fridge_front"}},
        {"skill": "open", "args": {"fixture": "freezer_door"}},
        {"skill": "pick", "args": {"container": "jar_f1"}},
        {"skill": "close", "args": {"fixture": "freezer_door"}},
        {"skill": "navigate", "args": {"waypoint": "counter_front"}},
        {"skill": "open", "args": {"fixture": "air_fryer_basket"}},
    ]


def test_pour_only_into_pour_appliances(world):
    report = execute(world, [*_holding_jar_at_fryer()[:5],
                             {"skill": "open", "args": {"fixture": "microwave_door"}},
                             {"skill": "pour", "args": {"container": "jar_f1", "into": "microwave"}}])
    assert "微波炉不能直接倒东西进去" in report.summary


def test_pour_empty_container_rejected(world):
    world.home.items["popcorn_chicken"].container = "box_f3"
    report = execute(world, [*_holding_jar_at_fryer(), {"skill": "pour", "args": {"container": "jar_f1", "into": "air_fryer"}}])
    assert "密封罐 1 号是空的" in report.summary


def test_pour_into_running_fryer_blocked(world):
    plan = [*_holding_jar_at_fryer()[:5], {"skill": "pour", "args": {"container": "jar_f1", "into": "air_fryer"}}]
    world.appliances["air_fryer"].running = True
    report = execute(world, plan)
    assert report.results[-1].stage == "safety" and "正在运行" in report.summary


def test_plastic_box_cannot_go_into_air_fryer(world):
    report = execute(world, [*_holding_jar_at_fryer(),
                             {"skill": "place", "args": {"container": "jar_f1", "location": "air_fryer", "slot": 0}}])
    assert report.results[-1].stage == "safety" and "密封罐 1 号不能进空气炸锅" in report.summary


@pytest.mark.parametrize("params,match", [
    ({"minutes": 12, "temp_c": 260}, "1 到 200"),
    ({"minutes": 12, "temp_c": True}, "1 到 200"),
    ({"minutes": 45, "temp_c": 180}, "1 到 30 分钟"),
])
def test_air_fryer_limits(world, params, match):
    report = simulate(world, [Step("appliance", {"appliance": "air_fryer", "action": "air_fry", "params": params})])
    assert not report.ok and match in report.results[-1].message


def test_temperature_only_for_appliances_that_have_it(world):
    report = simulate(world, [Step("appliance", {"appliance": "microwave", "action": "heat", "params": {"minutes": 2, "temp_c": 100}})])
    assert "微波炉不能设温度" in report.results[-1].message


def test_appliance_slots_are_checked(world):
    report = execute(world, [
        {"skill": "navigate", "args": {"waypoint": "sink_front"}},
        {"skill": "open", "args": {"fixture": "dishwasher_door"}},
        {"skill": "pick", "args": {"container": "bowl_1"}},
        {"skill": "place", "args": {"container": "bowl_1", "location": "dishwasher", "slot": 6}},
    ])
    assert "没有槽位 6" in report.summary
    # 第一次失败后收尾把洗碗机门关上了，重新打开再试
    report = execute(world, [
        {"skill": "open", "args": {"fixture": "dishwasher_door"}},
        {"skill": "place", "args": {"container": "bowl_1", "location": "dishwasher", "slot": 1}},
    ])
    assert "已被 bowl_2 占用" in report.summary


def test_cleanup_does_not_walk_while_holding(world):
    rules = [r for r in DEFAULT_RULES if r is not close_doors_before_leaving]
    apply_fault(world, "stuck:microwave_door")
    s = Scheduler(world, RulePlanner(), rules=rules)
    report = s.execute("故意不关门就走", [Step(**x) for x in [
        {"skill": "navigate", "args": {"waypoint": "fridge_front"}},
        {"skill": "open", "args": {"fixture": "freezer_door"}},
        {"skill": "pick", "args": {"container": "box_f1"}},
        {"skill": "navigate", "args": {"waypoint": "counter_front"}},
        {"skill": "open", "args": {"fixture": "microwave_door"}},
    ]])
    assert report.cleanup == []  # 冷冻室门在别处，手里有东西就不走过去
    assert report.needs_attention == ["冷冻室门还开着", "机器人手里还拿着冷冻盒 1 号，停在 counter_front"]


def test_stuck_door_cannot_be_closed_either(world):
    world.home.fixtures["chiller_door"].state = "open"
    world.opened_by_robot.add("chiller_door")
    world.robot.waypoint = "fridge_front"
    apply_fault(world, "stuck:chiller_door")
    report = execute(world, [{"skill": "close", "args": {"fixture": "chiller_door"}}])
    assert "关不上" in report.summary
    assert report.needs_attention == ["冷藏室门还开着"]


# ---------- 数据层 ----------


def test_find_item_prefers_longest_name():
    home = load_home(SPEC)
    assert home.find_item("把鸡米花放进空气炸锅").id == "popcorn_chicken"
    assert home.find_item("米快没了").id == "rice"


def test_legacy_microwave_safe_flag(raw):
    raw["containers"][0].pop("safe_for", None)
    raw["containers"][0]["microwave_safe"] = True
    home = parse_home(raw)
    assert home.containers["box_f1"].safe_for == ["microwave"]
    assert home.containers["box_f1"].microwave_safe


def test_storage_without_door_needs_waypoint(raw):
    table = next(s for s in raw["storage"] if s["id"] == "dining_table")
    del table["waypoint"]
    with pytest.raises(SpecError, match="机器人不知道站在哪里取放"):
        parse_home(raw)


def test_bad_home_is_rejected(raw):
    cup = next(c for c in raw["containers"] if c["id"] == "cup_1")
    cup["home"] = "air_fryer"
    with pytest.raises(SpecError, match="原位 air_fryer 不是存放区"):
        parse_home(raw)
    cup["home"], cup["home_slot"] = "dish_cabinet", 9
    with pytest.raises(SpecError, match="原位槽位 9 超出"):
        parse_home(raw)


def test_food_can_be_recorded_inside_appliance(raw):
    next(i for i in raw["items"] if i["id"] == "popcorn_chicken")["container"] = "air_fryer"
    assert parse_home(raw).items["popcorn_chicken"].container == "air_fryer"


def test_intake_skips_bowls_and_table(world):
    result = assign(world, [NewItem("花椒", "调料", "ambient", 0.2), NewItem("面粉", "干货", "ambient", 1.0)])
    containers = [p.container for p in result.placements]
    assert "jar_salt" not in containers and "cup_1" not in containers
    assert all(world.home.containers[c].kind in ("box", "jar") for c in containers)


def test_context_has_slots_homes_and_kinds(world):
    ctx = describe_world(world)
    chiller = next(s for s in ctx["storage"] if s["id"] == "chiller")
    assert chiller["free_slots"] == [3, 4, 5]
    cup = next(c for c in ctx["containers"] if c["id"] == "cup_1")
    assert (cup["kind"], cup["home"], cup["home_slot"]) == ("cup", "dish_cabinet", 3)
    fryer = next(a for a in ctx["appliances"] if a["id"] == "air_fryer")
    assert fryer["accepts_pour"] and fryer["max_temp_c"] == 200


def test_catalog_has_new_skills():
    names = [s["name"] for s in catalog()]
    assert "inspect" in names and "pour" in names
    appliance = next(s for s in catalog() if s["name"] == "appliance")
    assert appliance["input_schema"]["properties"]["params"]["required"] == ["minutes", "temp_c"]


# ---------- 大模型规划器碰到新规则 ----------


def test_fryer_plan_without_jar_return_is_completed(world):
    """Mac 上 qwen3:8b 实测：倒完直接启动，罐子还在手里。补全后先把罐子放回原位再启动。"""
    model_steps = [
        {"skill": "pick", "args": {"container": "jar_f1"}},
        {"skill": "pour", "args": {"container": "jar_f1", "into": "air_fryer"}},
        {"skill": "appliance", "args": {"appliance": "air_fryer", "action": "air_fry", "params": {"minutes": 12, "temp_c": 180}}},
    ]
    client = FakeClient(plan(model_steps))
    planner = LLMPlanner(client=client)
    steps = planner.plan("把鸡米花放进空气炸锅并启动", world)
    assert len(client.calls) == 1
    s = Scheduler(world, FixedPlan(steps))
    s.submit("把鸡米花放进空气炸锅并启动")
    [report] = s.run_pending()
    assert report.ok, report.render()
    assert TASKS[2].check(world, report) == []
    assert planner.attempts[0]["model"][0] == 'pick(container="jar_f1")'


def test_model_plan_with_only_essential_steps(world):
    """模型只写实质动作，走动和开关门全部由系统补上。"""
    model_steps = [
        {"skill": "inspect", "args": {"location": "dining_table"}},
        {"skill": "pick", "args": {"container": "cup_1"}},
        {"skill": "place", "args": {"container": "cup_1", "location": "dish_cabinet", "slot": 3}},
        {"skill": "pick", "args": {"container": "jar_salt"}},
        {"skill": "place", "args": {"container": "jar_salt", "location": "upper_cabinet", "slot": 0}},  # 槽位被占
    ]
    steps = LLMPlanner(client=FakeClient(plan(model_steps))).plan("把桌上散落的常用物品收回原位", world)
    s = Scheduler(world, FixedPlan(steps))
    s.submit("把桌上散落的常用物品收回原位")
    [report] = s.run_pending()
    assert report.ok, report.render()
    assert world.home.containers["jar_salt"].location == "upper_cabinet"
    assert world.home.containers["jar_salt"].slot != 0


def test_surface_is_photographed_before_first_pick(world):
    model_steps = [{"skill": "pick", "args": {"container": "cup_1"}},
                   {"skill": "place", "args": {"container": "cup_1", "location": "dish_cabinet", "slot": 3}}]
    steps = normalize([Step(**x) for x in model_steps], world)
    skills = [s.skill for s in steps]
    assert skills.index("inspect") < skills.index("pick")


def test_unknown_steps_are_left_for_preview_to_reject(world):
    steps = normalize([Step("open", {"fixture": "air_fry,er_basket"})], world)
    assert steps == [Step("open", {"fixture": "air_fry,er_basket"})]
    report = simulate(world, steps)
    assert "部件 air_fry,er_basket 不存在" in report.summary


def test_robot_does_not_stop_appliance_it_did_not_start(world):
    """Mac 上 qwen3:8b 实测：洗碗机还在洗，模型直接把它停了再收碗。"""
    apply_fault(world, "busy:dishwasher")
    report = Scheduler(world, RulePlanner()).execute("把洗好的碗从洗碗机收到柜子", [
        Step("appliance", {"appliance": "dishwasher", "action": "stop", "params": {"minutes": None, "temp_c": None}}),
    ])
    assert report.results[-1].stage == "safety"
    assert "不是机器人启动的" in report.summary
    assert world.appliance_running("dishwasher")


def test_owner_can_ask_to_stop(world):
    apply_fault(world, "busy:dishwasher")
    report = Scheduler(world, RulePlanner()).execute("把洗碗机停了", [
        Step("appliance", {"appliance": "dishwasher", "action": "stop", "params": {"minutes": None, "temp_c": None}}),
    ])
    assert report.ok
    assert not world.appliance_running("dishwasher")


def test_robot_can_stop_what_it_started(world):
    s = Scheduler(world, RulePlanner())
    s.submit("把牛肉拿出来解冻")
    s.run_pending()
    report = s.execute("算了别解冻了", [
        Step("appliance", {"appliance": "microwave", "action": "stop", "params": {"minutes": None, "temp_c": None}}),
    ])
    assert report.ok


def test_preview_knows_the_instruction(world):
    apply_fault(world, "busy:dishwasher")
    stop = [Step("appliance", {"appliance": "dishwasher", "action": "stop", "params": {"minutes": None, "temp_c": None}})]
    assert not simulate(world, stop).ok
    assert simulate(world, stop, instruction="洗碗机先暂停一下").ok


def test_default_home_is_initial_place():
    home = load_home(SPEC)
    assert (home.containers["box_f1"].home, home.containers["box_f1"].home_slot) == ("freezer", 0)
    assert home.containers["cup_1"].home == "dish_cabinet"  # 写了原位的照写的来
    assert home.containers["bowl_1"].home == "dish_cabinet"


class FixedPlan:
    def __init__(self, steps):
        self.steps = steps

    def plan(self, instruction, world):
        return self.steps


# ---------- 命令行 ----------


def test_cli_mvp(capsys):
    assert main(["mvp"]) == 0
    out = capsys.readouterr().out
    assert "共 5/5 通过" in out


def test_cli_mvp_faults_for_one_task(capsys):
    assert main(["mvp", "--faults", "--task", "air_fry"]) == 0
    out = capsys.readouterr().out
    assert "共 5/5 通过" in out and "现场照片" in out


def test_cli_run_with_new_faults(capsys):
    assert main(["run", "把桌上散落的常用物品收回原位", "--fault", "drop:cup_1"]) == 1
    out = capsys.readouterr().out
    assert "滑落" in out and "水杯掉在地上了" in out
    assert main(["run", "x", "--fault", "melt:freezer"]) == 2


def test_cli_json_has_photos(capsys):
    assert main(["run", "看看冰箱里还有什么", "--json"]) == 0
    [report] = json.loads(capsys.readouterr().out)
    assert [p["title"] for p in report["photos"]] == ["冷冻室", "冷藏室"]
    assert report["scene"] is None

"""第 2 步：大模型规划层和入库模块。大模型调用全部用假客户端，不联网、不花钱。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robohome.cli import main
from robohome.intake import KeywordRecognizer, LLMRecognizer, NewItem, assign
from robohome.planner import LLMPlanner, PlanningError, Step
from robohome.planner.context import describe_world
from robohome.planner.llm import plan_schema
from robohome.scheduler import Scheduler, simulate
from robohome.skills import catalog
from robohome.spec import load_home, parse_home, home_to_dict, save_home
from robohome.world import World

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"

GOOD_PLAN = [
    {"skill": "navigate", "args": {"waypoint": "freezer_front"}},
    {"skill": "open", "args": {"fixture": "freezer_door"}},
    {"skill": "pick", "args": {"container": "box_f1"}},
    {"skill": "close", "args": {"fixture": "freezer_door"}},
    {"skill": "navigate", "args": {"waypoint": "counter_front"}},
    {"skill": "open", "args": {"fixture": "microwave_door"}},
    {"skill": "place", "args": {"container": "box_f1", "location": "microwave", "slot": 0}},
    {"skill": "close", "args": {"fixture": "microwave_door"}},
    {"skill": "appliance", "args": {"appliance": "microwave", "action": "defrost", "params": {"minutes": 8}}},
]
# 忘了开门就去拿
BAD_PLAN = [
    {"skill": "navigate", "args": {"waypoint": "freezer_front"}},
    {"skill": "pick", "args": {"container": "box_f1"}},
]


def response(payload, stop_reason="end_turn"):
    text = json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    """按顺序返回预设回复，并记录每次请求的参数。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # 记录当时的消息快照（之后列表会被追加）
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def plan(steps, feasible=True, reason="ok"):
    return response({"feasible": feasible, "reason": reason, "steps": steps})


@pytest.fixture
def world():
    return World.from_home(load_home(SPEC), start_waypoint="entrance")


# ---------- 规划层 ----------


def test_llm_plan_first_try(world):
    client = FakeClient(plan(GOOD_PLAN))
    steps = LLMPlanner(client=client).plan("把牛肉拿出来解冻", world)
    assert [s.skill for s in steps][:3] == ["navigate", "open", "pick"]
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "box_f1" in call["messages"][0]["content"]  # 世界状态交给了模型
    # 预演不能改动真实世界
    assert world.home.fixtures["freezer_door"].state == "closed"
    assert world.robot.waypoint == "entrance"


def test_llm_replans_after_failed_simulation(world):
    # 关掉自动补全，直接测重规划：模型忘了开门，预演失败后把原因告诉它
    client = FakeClient(plan(BAD_PLAN), plan(GOOD_PLAN))
    steps = LLMPlanner(client=client, normalize_plans=False).plan("把牛肉拿出来解冻", world)
    assert len(steps) == len(GOOD_PLAN)
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "冷冻室门没开" in second[2]["content"]


def test_llm_infeasible_raises_with_reason(world):
    # 第一次说做不到会被要求核实一次，核实后还是做不到才放弃
    client = FakeClient(plan([], feasible=False, reason="台账里没有榴莲"), plan([], feasible=False, reason="台账里没有榴莲"))
    with pytest.raises(PlanningError, match="榴莲"):
        LLMPlanner(client=client).plan("把榴莲拿出来", world)
    assert len(client.calls) == 2
    assert "核实" in str(client.calls[1]["messages"][-1]["content"])


def test_llm_wrong_refusal_is_challenged_once(world):
    good = [
        {"skill": "pick", "args": {"container": "box_f1"}},
        {"skill": "place", "args": {"container": "box_f1", "location": "microwave", "slot": 0}},
        {"skill": "appliance", "args": {"appliance": "microwave", "action": "defrost", "params": {"minutes": 8, "temp_c": None}}},
    ]
    client = FakeClient(plan([], feasible=False, reason="微波炉不能放盒子"), plan(good))
    planner = LLMPlanner(client=client)
    steps = planner.plan("把牛肉拿出来解冻", world)
    assert any(s.skill == "appliance" and s.args["appliance"] == "microwave" for s in steps)
    assert len(client.calls) == 2
    assert "做不到" in planner.attempts[0]["error"]


def test_llm_gives_up_after_max_attempts(world):
    other = [{"skill": "navigate", "args": {"waypoint": "freezer_front"}}, {"skill": "pick", "args": {"container": "box_f2"}}]
    client = FakeClient(plan(BAD_PLAN), plan(other))
    with pytest.raises(PlanningError, match="重试 2 次"):
        LLMPlanner(client=client, max_attempts=2, normalize_plans=False).plan("把牛肉拿出来解冻", world)


def test_llm_stops_when_it_repeats_the_same_plan(world):
    """同一个计划预演只会同样失败，模型交回同一个计划就不再浪费时间。"""
    client = FakeClient(plan(BAD_PLAN), plan(BAD_PLAN), plan(BAD_PLAN))
    with pytest.raises(PlanningError, match="冷冻室门没开（模型改不出别的计划）"):
        LLMPlanner(client=client, normalize_plans=False).plan("把牛肉拿出来解冻", world)
    assert len(client.calls) == 2


def test_llm_refusal_and_truncation(world):
    with pytest.raises(PlanningError, match="拒绝"):
        LLMPlanner(client=FakeClient(response({}, "refusal"))).plan("x", world)
    with pytest.raises(PlanningError, match="截断"):
        LLMPlanner(client=FakeClient(response({}, "max_tokens"))).plan("x", world)


def test_scheduler_runs_llm_plan_end_to_end(world):
    s = Scheduler(world, LLMPlanner(client=FakeClient(plan(GOOD_PLAN))))
    s.submit("把牛肉拿出来解冻")
    [report] = s.run_pending()
    assert report.ok, report.render()
    assert world.appliances["microwave"].status()["mode"] == "defrost"


def test_llm_plan_still_checked_at_execution(world):
    """预演通过后现实变了（容器不见了），调度器照样失败即停。"""
    world.missing_containers.add("box_f1")
    s = Scheduler(world, LLMPlanner(client=FakeClient(plan(GOOD_PLAN))))
    s.submit("把牛肉拿出来解冻")
    [report] = s.run_pending()
    assert not report.ok and report.results[-1].stage == "execution"


def test_plan_schema_covers_every_skill():
    variants = plan_schema()["properties"]["steps"]["items"]["anyOf"]
    assert {v["properties"]["skill"]["enum"][0] for v in variants} == {s["name"] for s in catalog()}

    def strict(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            for v in node.values():
                strict(v)
        elif isinstance(node, list):
            for v in node:
                strict(v)

    strict(plan_schema())


def test_describe_world_lists_contents(world):
    d = describe_world(world)
    box = next(c for c in d["containers"] if c["id"] == "box_f1")
    assert box["contents"] == ["牛肉"]
    assert d["robot"]["waypoint"] == "entrance"


def test_simulate_ignores_reality_gaps(world):
    world.missing_containers.add("box_f1")
    report = simulate(world, [Step(**s) for s in GOOD_PLAN])
    assert report.ok  # 预演只看台账
    assert world.home.containers["box_f1"].location == "freezer"


# ---------- 入库 ----------


def test_keyword_recognizer():
    items, unknown = KeywordRecognizer().recognize("今天买了一斤五花肉、两盒酸奶和一包薯片")
    assert [(i.name, i.temperature) for i in items] == [("猪肉", "frozen"), ("酸奶", "chilled")]
    assert unknown == ["一包薯片"]


def test_assign_prefers_same_category(world):
    result = assign(world, [NewItem("猪肉", "肉类", "frozen", 1.0)])
    p = result.placements[0]
    assert p.container == "box_f1"
    assert world.home.items[p.item_id].container == "box_f1"


def test_assign_opens_empty_container_and_never_mixes(world):
    result = assign(world, [NewItem("虾", "水产", "frozen", 1.0)])
    assert result.placements[0].container == "box_f3"


def test_assign_respects_capacity_across_batch(world):
    # 冷冻盒 1 号 4.5 升，已有牛肉 1 升；再放 3 升还行，再来 1 升就得换盒子
    items = [NewItem("猪肉", "肉类", "frozen", 3.0), NewItem("羊肉", "肉类", "frozen", 1.0)]
    result = assign(world, items)
    assert [p.container for p in result.placements] == ["box_f1", "box_f3"]


def test_assign_reports_when_nothing_fits(world):
    result = assign(world, [NewItem("西瓜", "水果", "chilled", 20.0)])
    assert not result.ok
    assert "放得下" in result.placements[0].note


def test_assign_without_commit_leaves_ledger(world):
    before = dict(world.home.items)
    assign(world, [NewItem("虾", "水产", "frozen", 1.0)], commit=False)
    assert world.home.items == before


def test_intake_result_can_be_planned_against(world):
    """入库后的新物品立刻能被规则规划器找到。"""
    from robohome.planner import RulePlanner

    assign(world, [NewItem("虾", "水产", "frozen", 1.0, aliases=["大虾"])])
    steps = RulePlanner().plan("把大虾拿出来", world)
    assert Step("pick", {"container": "box_f3"}) in steps


def test_llm_recognizer_with_fake_client():
    payload = {
        "items": [{"name": "五花肉", "category": "肉类", "temperature": "frozen", "volume_l": 1.0, "aliases": ["猪肉"]}],
        "unrecognized": [],
    }
    client = FakeClient(response(payload))
    items, unknown = LLMRecognizer(client=client).recognize("买了五花肉", categories=["肉类"])
    assert items[0].aliases == ["猪肉"] and unknown == []
    assert "肉类" in client.calls[0]["system"]


def test_save_and_reload_round_trip(world, tmp_path):
    assign(world, [NewItem("虾", "水产", "frozen", 1.0)])
    out = tmp_path / "home.json"
    save_home(world.home, out)
    reloaded = load_home(out)
    assert reloaded == world.home
    assert parse_home(home_to_dict(reloaded)) == reloaded


def test_cli_intake(capsys, tmp_path):
    out = tmp_path / "home.json"
    # 不加 --yes 只预览，不写文件
    assert main(["intake", "买了两盒酸奶", "--save", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "冷藏盒 1 号" in printed and "只是预览" in printed
    assert not out.exists()
    assert main(["intake", "买了两盒酸奶", "--save", str(out), "--yes"]) == 0
    assert "台账已写入" in capsys.readouterr().out
    assert any(i.name == "酸奶" for i in load_home(out).items.values())
    assert main(["intake", "买了一包薯片"]) == 1


def test_hazard_appliance_requires_run_time(world):
    """Mac 上用 qwen3:8b 实测时，模型把解冻时间留空了；这种计划必须在预演里被拦下。"""
    plan_no_time = [dict(s) for s in GOOD_PLAN]
    plan_no_time[-1] = {"skill": "appliance",
                        "args": {"appliance": "microwave", "action": "defrost", "params": {"minutes": None}}}
    report = simulate(world, [Step(**s) for s in plan_no_time])
    assert not report.ok
    assert report.results[-1].stage == "precondition"
    assert "运行分钟数" in report.results[-1].message


def test_llm_replans_when_run_time_missing(world):
    plan_no_time = [dict(s) for s in GOOD_PLAN]
    plan_no_time[-1] = {"skill": "appliance",
                        "args": {"appliance": "microwave", "action": "defrost", "params": {"minutes": None}}}
    client = FakeClient(plan(plan_no_time), plan(GOOD_PLAN))
    steps = LLMPlanner(client=client).plan("把牛肉拿出来解冻", world)
    assert steps[-1].args["params"] == {"minutes": 8, "temp_c": None}
    assert "运行分钟数" in client.calls[1]["messages"][2]["content"]


# Mac 上用 qwen3:8b 第二次实测：模型取完牛肉没关冷冻室门就走了
PLAN_DOOR_LEFT_OPEN = [s for s in GOOD_PLAN if s != {"skill": "close", "args": {"fixture": "freezer_door"}}]


def test_leaving_with_door_open_is_blocked(world):
    report = simulate(world, [Step(**s) for s in PLAN_DOOR_LEFT_OPEN])
    assert not report.ok
    assert report.results[-1].stage == "safety"
    assert "离开 freezer_front 前要先关上冷冻室门" in report.summary


def test_plan_must_end_with_doors_closed(world):
    plan_ = [Step("navigate", {"waypoint": "fridge_front"}), Step("open", {"fixture": "chiller_door"}),
             Step("pick", {"container": "box_c1"})]
    report = simulate(world, plan_)
    assert not report.ok
    assert "冷藏室门还开着" in report.summary


def test_door_opened_by_someone_else_does_not_block(world):
    world.home.fixtures["chiller_door"].state = "open"  # 主人自己开着的门，不算机器人的
    report = simulate(world, [Step("navigate", {"waypoint": "fridge_front"}),
                              Step("navigate", {"waypoint": "counter_front"})])
    assert report.ok


def test_llm_replans_when_door_left_open(world):
    client = FakeClient(plan(PLAN_DOOR_LEFT_OPEN), plan(GOOD_PLAN))
    steps = LLMPlanner(client=client, normalize_plans=False).plan("把牛肉拿出来解冻", world)
    assert Step("close", {"fixture": "freezer_door"}) in steps
    assert "关上冷冻室门" in client.calls[1]["messages"][2]["content"]


def test_door_left_open_is_fixed_without_replanning(world):
    """第 3 周 Mac 实测后加的自动补全：忘了关门不用再麻烦模型重想一遍。"""
    client = FakeClient(plan(PLAN_DOOR_LEFT_OPEN))
    steps = LLMPlanner(client=client).plan("把牛肉拿出来解冻", world)
    assert len(client.calls) == 1
    assert steps.index(Step("close", {"fixture": "freezer_door"})) < steps.index(Step("navigate", {"waypoint": "counter_front"}))

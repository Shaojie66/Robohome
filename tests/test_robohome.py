import copy
import json
from pathlib import Path

import pytest

from robohome.cli import main
from robohome.planner import RulePlanner, Step
from robohome.scheduler import Scheduler
from robohome.skills import catalog
from robohome.spec import SpecError, load_home, parse_home
from robohome.world import World

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"


@pytest.fixture
def raw():
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture
def world():
    return World.from_home(load_home(SPEC), start_waypoint="entrance")


def run(world, *instructions):
    s = Scheduler(world, RulePlanner())
    for i in instructions:
        s.submit(i)
    return s.run_pending()


# ---------- 数据层 ----------


def test_sample_spec_loads():
    home = load_home(SPEC)
    assert home.find_item("帮我把牛排拿出来").id == "beef"
    assert home.fixture_enclosing("freezer").id == "freezer_door"


def test_spec_rejects_unknown_container(raw):
    raw["items"][0]["container"] = "nope"
    with pytest.raises(SpecError, match="容器 nope 不存在"):
        parse_home(raw)


def test_spec_rejects_slot_collision(raw):
    raw["containers"][1]["slot"] = 0
    with pytest.raises(SpecError, match="同一个槽位"):
        parse_home(raw)


def test_spec_rejects_bad_waypoint(raw):
    raw["fixtures"][0]["waypoint"] = "nowhere"
    with pytest.raises(SpecError, match="没有点位 nowhere"):
        parse_home(raw)


# ---------- 端到端 ----------


def test_defrost_beef_end_to_end(world):
    [report] = run(world, "把牛肉拿出来解冻")
    assert report.ok, report.render()
    assert world.home.containers["box_f1"].location == "microwave"
    assert world.home.fixtures["freezer_door"].state == "closed"
    assert world.home.fixtures["microwave_door"].state == "closed"
    assert world.appliances["microwave"].status()["mode"] == "defrost"
    assert world.robot.holding is None


def test_fetch_milk(world):
    [report] = run(world, "把牛奶拿出来")
    assert report.ok
    assert world.robot.holding == "box_c1"


def test_unknown_instruction_fails_cleanly(world):
    [report] = run(world, "帮我跳个舞")
    assert not report.ok
    assert "规划失败" in report.summary


# ---------- 失败即停 ----------


def test_appliance_offline_stops(world):
    world.appliances["microwave"].online = False
    [report] = run(world, "把牛肉拿出来解冻")
    assert not report.ok
    assert report.results[-1].stage == "execution"
    assert "离线" in report.summary


def test_missing_container_stops_before_anything_else(world):
    world.missing_containers.add("box_f1")
    [report] = run(world, "把牛肉拿出来解冻")
    assert not report.ok
    assert len(report.results) == 3  # 停在 pick，后面的步骤都没做
    assert world.robot.holding is None


def test_queue_halts_after_failure(world):
    world.missing_containers.add("box_f1")
    reports = run(world, "把牛肉拿出来解冻", "把牛奶拿出来")
    assert len(reports) == 1


def test_precondition_blocks_pick_with_closed_door(world):
    s = Scheduler(world, RulePlanner())
    report = s.execute("手写计划", [Step("navigate", {"waypoint": "fridge_front"}), Step("pick", {"container": "box_f1"})])
    assert not report.ok
    assert report.results[-1].stage == "precondition"
    assert "冷冻室门没开" in report.results[-1].message


# ---------- 安全规则 ----------


def test_non_microwave_safe_container_blocked(world):
    [report] = run(world, "把饺子拿出来解冻")
    assert not report.ok
    assert report.results[-1].stage == "safety"
    assert "不能进微波炉" in report.summary


def test_forbidden_zone_blocked(world):
    report = Scheduler(world, RulePlanner()).execute("去灶台", [Step("navigate", {"waypoint": "stove_front"})])
    assert report.results[-1].stage == "safety"
    assert world.robot.waypoint == "entrance"


def test_cannot_open_running_microwave(world):
    run(world, "把牛肉拿出来解冻")
    report = Scheduler(world, RulePlanner()).execute("开门", [Step("open", {"fixture": "microwave_door"})])
    assert report.results[-1].stage == "safety"
    assert "正在运行" in report.summary


def test_no_hazard_start_while_holding(world):
    plan = [
        Step("navigate", {"waypoint": "fridge_front"}),
        Step("open", {"fixture": "chiller_door"}),
        Step("pick", {"container": "box_c1"}),
        Step("appliance", {"appliance": "microwave", "action": "heat"}),
    ]
    report = Scheduler(world, RulePlanner()).execute("边拿边开", plan)
    assert report.results[-1].stage == "safety"


def test_world_does_not_mutate_source_home():
    home = load_home(SPEC)
    before = copy.deepcopy(home)
    w = World.from_home(home, start_waypoint="entrance")
    run(w, "把牛肉拿出来解冻")
    assert home == before


# ---------- 技能清单 / CLI ----------


def test_catalog_lists_all_skills():
    names = {s["name"] for s in catalog()}
    assert names == {"navigate", "open", "close", "pick", "place", "inspect", "pour", "appliance"}


def test_cli_run_and_fault(capsys):
    assert main(["run", "把牛肉拿出来解冻"]) == 0
    assert main(["run", "把牛肉拿出来解冻", "--fault", "offline:microwave"]) == 1
    assert main(["validate"]) == 0
    assert "已停止" in capsys.readouterr().out

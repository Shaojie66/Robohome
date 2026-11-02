"""第 4 步：物理仿真。几何检查不需要 MuJoCo；其余测试没装 MuJoCo 时跳过。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robohome.cli import main
from robohome.mvp import HIDDEN_FAULTS, TASKS, fresh_world, run_task
from robohome.planner import RulePlanner, Step
from robohome.scheduler import Scheduler
from robohome.sim.layout import Reach, areas, check_geometry
from robohome.spec import SpecError, load_home, parse_home, save_home
from robohome.world import FLOOR

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"
FAULT_CASES = [(t, f, e) for t in TASKS for f, e in t.faults]


@pytest.fixture
def raw():
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _area(raw, area_id):
    return next(x for x in raw["storage"] + raw["appliances"] if x["id"] == area_id)


# ---------- 几何检查（不需要 MuJoCo） ----------


def test_sample_kitchen_geometry_is_consistent():
    assert check_geometry(load_home(SPEC)) == []


def test_slots_are_laid_out_inside_the_area():
    home = load_home(SPEC)
    a = areas(home)["freezer"]
    xs = [a.slot_local(s)[0] for s in range(a.slots)]
    ys = [a.slot_local(s)[1] for s in range(a.slots)]
    assert xs == pytest.approx([-0.175, 0.175, -0.175, 0.175])
    assert ys[0] < ys[2]  # 0 号在外面一行
    chiller = areas(home)["chiller"]
    assert [chiller.slot_local(s)[2] for s in (0, 2, 4)] == pytest.approx([0, 0.3, 0.6])  # 三层


def test_container_too_big_for_its_slot(raw):
    next(c for c in raw["containers"] if c["id"] == "box_c1")["size_m"] = [0.5, 0.55, 0.12]
    assert any("冷藏盒 1 号" in p and "放不进冷藏室" in p for p in check_geometry(parse_home(raw)))


def test_slot_out_of_reach(raw):
    raw["rooms"][0]["waypoints"]["sink_front"]["position"] = [4.6, -1.5, 0.0]
    problems = check_geometry(parse_home(raw))
    assert any("碗柜槽位" in p and "手够不着" in p for p in problems)
    assert any("碗柜门的把手" in p and "够不着" in p for p in problems)


def test_reach_counts_the_hand_above_the_container(raw):
    """槽位底面够得着还不够：取放时手在容器顶上。餐桌抬到 1.9 米，杯子顶上的手就超过 1.95 米了。"""
    pos = _area(raw, "dining_table")["geometry"]["position"]
    pos[2] = 1.9
    assert any("餐桌槽位" in p and "手够不着" in p for p in check_geometry(parse_home(raw)))


def test_handle_must_be_on_the_door(raw):
    next(f for f in raw["fixtures"] if f["id"] == "chiller_door")["handle_pose"]["position"] = [1.6, 0.6, 1.05]
    assert any("冷藏室门的把手" in p and "不在冷藏室的正面上" in p for p in check_geometry(parse_home(raw)))


def test_area_with_containers_needs_geometry(raw):
    del _area(raw, "upper_cabinet")["geometry"]
    assert any("吊柜" in p and "没有写三维尺寸" in p for p in check_geometry(parse_home(raw)))


def test_layout_must_hold_all_slots(raw):
    _area(raw, "chiller")["geometry"]["layout"] = [2, 1, 2]
    with pytest.raises(SpecError, match="排不下 6 个槽位"):
        parse_home(raw)


def test_reach_is_configurable():
    home = load_home(SPEC)
    assert any("手够不着" in p for p in check_geometry(home, Reach(horizontal_m=0.5)))


def test_geometry_survives_save_and_load(tmp_path):
    home = load_home(SPEC)
    save_home(home, tmp_path / "home.json")
    again = load_home(tmp_path / "home.json")
    assert again.storage["chiller"].geometry == home.storage["chiller"].geometry
    assert again.fixtures["dishwasher_door"].hinge == "bottom"


def test_cli_sim_check(capsys):
    assert main(["sim-check", str(SPEC)]) == 0
    assert "三维数据没问题" in capsys.readouterr().out


# ---------- 物理仿真 ----------

mujoco = pytest.importorskip("mujoco")
from robohome.sim.body import SimBody
from robohome.sim.render import Recorder


def attach(world):
    return SimBody.attach(world)


def sim_world(fault=None):
    from robohome.mvp import apply_fault

    world = fresh_world(SPEC)
    if fault:
        apply_fault(world, fault)
    return world, SimBody.attach(world)


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_task_in_simulation(task):
    run = run_task(task, RulePlanner(), SPEC, body=attach)
    assert run.passed, (run.problems, run.report.render())


@pytest.mark.parametrize("task,fault,expect", FAULT_CASES, ids=[f"{t.id}-{f}" for t, f, _ in FAULT_CASES])
def test_fault_in_simulation(task, fault, expect):
    run = run_task(task, RulePlanner(), SPEC, fault, expect, body=attach)
    assert run.passed, (run.problems, run.report.render())
    if fault.split(":")[0] in HIDDEN_FAULTS and expect == "stop":
        # 看不见的故障必须是物理上真的发生、被仿真发现的，不是模拟实现按设定报的错
        assert "仿真" in run.report.summary, run.report.summary


def test_attach_hands_faults_to_physics():
    world, body = sim_world("stuck:freezer_door")
    assert not world.stuck_fixtures and world.body is body
    j = body.model.joint("j_freezer_door")
    assert body.model.dof_frictionloss[j.dofadr[0]] > 1000
    assert world.sandbox().body is None  # 预演照旧只在内存里做


def _run(world, steps):
    return Scheduler(world, RulePlanner()).execute("手写计划", [Step(**s) for s in steps])


def _do(world, steps):
    """逐步执行，不做结束检查和收尾，方便看中间状态。"""
    report = Scheduler(world, RulePlanner(), cleanup=False).execute("手写计划", [Step(**s) for s in steps])
    assert all(r.ok for r in report.results), report.render()


def test_drawer_carries_its_contents():
    world, body = sim_world()
    before = body._container_pos("box_f1")
    _do(world, [{"skill": "navigate", "args": {"waypoint": "fridge_front"}},
                {"skill": "open", "args": {"fixture": "freezer_door"}}])
    moved = body._container_pos("box_f1") - before
    assert moved[1] == pytest.approx(-body.info.door_open["freezer_door"], abs=0.02)
    assert body.locate("box_f1") == ("freezer", 0)  # 跟着抽屉出来，还在自己的槽位里


def test_drawer_open_at_start_holds_its_contents(raw):
    """规范里冷冻室抽屉本来就开着：盒子要在拉出来的抽屉里，直接就能取。"""
    next(f for f in raw["fixtures"] if f["id"] == "freezer_door")["state"] = "open"
    world = fresh_world(SPEC)
    world.home = parse_home(raw)
    world.robot.waypoint = "entrance"
    body = SimBody.attach(world)
    assert body.locate("box_f1") == ("freezer", 0)
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "fridge_front"}},
                          {"skill": "inspect", "args": {"location": "freezer"}},
                          {"skill": "pick", "args": {"container": "box_f1"}}])
    assert all(r.ok for r in report.results), report.render()
    assert not report.photos[0].notes  # 照片和台账对得上


def test_door_really_swings_open():
    world, body = sim_world()
    _do(world, [{"skill": "navigate", "args": {"waypoint": "sink_front"}},
                {"skill": "open", "args": {"fixture": "dishwasher_door"}}])
    assert body._joint_q("dishwasher_door") > 0.8 * body.info.door_open["dishwasher_door"]


def test_dropped_container_ends_on_the_floor():
    world, body = sim_world("drop:cup_1")
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "table_front"}},
                          {"skill": "pick", "args": {"container": "cup_1"}},
                          {"skill": "navigate", "args": {"waypoint": "sink_front"}}])
    assert not report.ok and "从手里滑落" in report.summary
    assert body.locate("cup_1") == (FLOOR, 0)
    assert world.home.containers["cup_1"].location == FLOOR
    assert body._container_pos("cup_1")[2] < 0.1


def test_inspect_reports_what_is_physically_there():
    """有人把冷藏盒 1 号拿走了（台账没改）：照片里看不到，要提醒主人。"""
    world, body = sim_world()
    j = body.model.joint("fj_box_c1")
    body.data.qpos[j.qposadr[0] : j.qposadr[0] + 3] = [0.3, 0.3, 0.1]
    body._settle(0.5)
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "fridge_front"}},
                          {"skill": "open", "args": {"fixture": "chiller_door"}},
                          {"skill": "inspect", "args": {"location": "chiller"}},
                          {"skill": "close", "args": {"fixture": "chiller_door"}}])
    assert report.ok
    assert "台账里有冷藏盒 1 号，照片里没看到" in report.photos[0].notes


def test_pick_finds_a_different_container():
    """槽位里放的是别的盒子：如实报告，不把错的东西拿走。"""
    world, _ = sim_world()
    world.home.containers["box_f1"].slot, world.home.containers["box_f2"].slot = 1, 0  # 台账写反了
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "fridge_front"}},
                          {"skill": "open", "args": {"fixture": "freezer_door"}},
                          {"skill": "pick", "args": {"container": "box_f1"}}])
    assert not report.ok and "里是冷冻盒 2 号，不是冷冻盒 1 号" in report.summary
    assert world.robot.holding is None


def test_place_that_does_not_land_is_reported(monkeypatch):
    world, body = sim_world()
    monkeypatch.setattr(body, "place", lambda w, c, loc, slot: (FLOOR, 0))
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "table_front"}},
                          {"skill": "pick", "args": {"container": "cup_1"}},
                          {"skill": "place", "args": {"container": "cup_1", "location": "dining_table", "slot": 2}}])
    assert not report.ok and "没放稳" in report.summary
    assert world.home.containers["cup_1"].location == FLOOR


def test_out_of_reach_in_simulation(raw):
    raw = copy.deepcopy(raw)
    raw["rooms"][0]["waypoints"]["table_front"]["position"] = [2.6, 1.2, 0.0]
    world = fresh_world(SPEC)
    world.home = parse_home(raw)
    world.robot.waypoint = "entrance"
    SimBody.attach(world)
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "table_front"}},
                          {"skill": "pick", "args": {"container": "cup_1"}}])
    assert not report.ok and "手够不着" in report.summary


def test_pour_and_start_show_up_in_the_scene():
    world, body = sim_world()
    s = Scheduler(world, RulePlanner())
    s.submit(TASKS[2].instruction)
    [report] = s.run_pending()
    assert report.ok, report.render()
    assert body.model.geom_rgba[body.model.geom("food_air_fryer").id][3] == 1.0
    assert body.model.geom_rgba[body.model.geom("led_air_fryer").id][0] > 0.9  # 指示灯变红


def test_cli_mvp_in_simulation(capsys):
    assert main(["mvp", "--sim", "--task", "fridge"]) == 0
    assert "共 1/1 通过" in capsys.readouterr().out


def test_cli_run_in_simulation_with_fault(capsys):
    """门卡住这件事交给物理：执行时门真的拉不开。"""
    assert main(["run", "把冷冻的牛肉转到冷藏层解冻", "--sim", "--fault", "stuck:freezer_door"]) == 1
    out = capsys.readouterr().out
    assert "拉不开" in out and "仿真" in out


def test_recording(tmp_path):
    world = fresh_world(SPEC)
    rec = Recorder(tmp_path, fps=4)
    SimBody.attach(world, recorder=rec)
    s = Scheduler(world, RulePlanner())
    s.submit(TASKS[0].instruction)
    [report] = s.run_pending()
    assert report.ok
    if rec.failed:
        pytest.skip("这台机器上渲染不可用")
    assert all(p.image and Path(p.image).exists() for p in report.photos)
    gif = rec.save_gif(tmp_path / "fridge.gif")
    assert gif and Path(gif).stat().st_size > 10_000

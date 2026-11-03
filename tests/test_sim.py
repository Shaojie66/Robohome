"""第 4 步：物理仿真。几何检查不需要 MuJoCo；其余测试没装 MuJoCo 时跳过。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from robohome.cli import main
from robohome.mvp import HIDDEN_FAULTS, TASKS, fresh_world, run_task
from robohome.planner import RulePlanner, Step
from robohome.scheduler import Scheduler
from robohome.sim.floor import FORBIDDEN_RADIUS, G1_FOOTPRINT, FloorMap, pose_of
from robohome.sim.layout import G1_REACH, Reach, areas, check_geometry
from robohome.spec import SpecError, load_home, parse_home, save_home
from robohome.world import FLOOR

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"
FAULT_CASES = [(t, f, e) for t in TASKS for f, e in t.faults]


@pytest.fixture
def raw():
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _trail_points(trail, step=0.02):
    for (ax, ay), (bx, by) in zip(trail, trail[1:]):
        n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / step))
        for k in range(n + 1):
            yield ax + (bx - ax) * k / n, ay + (by - ay) * k / n


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
    raw["rooms"][0]["waypoints"]["dish_front"]["position"] = [5.3, -1.5, 0.0]
    problems = check_geometry(parse_home(raw))
    assert any("碗柜槽位" in p and "手够不着" in p for p in problems)
    assert any("碗柜门的把手" in p and "够不着" in p for p in problems)


def test_reach_counts_the_hand_above_the_container(raw):
    """槽位底面够得着还不够：取放时手在容器顶上。餐桌抬到 1.45 米还在 G1 的 1.55 米以内，杯子顶上的手就超了。"""
    pos = _area(raw, "dining_table")["geometry"]["position"]
    pos[2] = 1.45
    assert any("餐桌槽位" in p and "手够不着" in p for p in check_geometry(parse_home(raw)))


def test_handle_must_be_on_the_door(raw):
    next(f for f in raw["fixtures"] if f["id"] == "chiller_door")["handle_pose"]["position"] = [1.6, 0.6, 1.05]
    assert any("冷藏室门的把手" in p and "不在冷藏室的正面上" in p for p in check_geometry(parse_home(raw)))


def test_area_with_containers_needs_geometry(raw):
    del _area(raw, "pantry")["geometry"]
    assert any("橱柜" in p and "没有写三维尺寸" in p for p in check_geometry(parse_home(raw)))


def test_layout_must_hold_all_slots(raw):
    _area(raw, "chiller")["geometry"]["layout"] = [2, 1, 2]
    with pytest.raises(SpecError, match="排不下 6 个槽位"):
        parse_home(raw)


def test_reach_is_configurable():
    home = load_home(SPEC)
    assert any("手够不着" in p for p in check_geometry(home, Reach.box(0.5, 0.05, 1.95)))


def test_geometry_survives_save_and_load(raw, tmp_path):
    next(f for f in raw["fixtures"] if f["id"] == "microwave_door")["hinge"] = "bottom"
    home = parse_home(raw)
    save_home(home, tmp_path / "home.json")
    again = load_home(tmp_path / "home.json")
    assert again.storage["chiller"].geometry == home.storage["chiller"].geometry
    assert again.fixtures["microwave_door"].hinge == "bottom"


def test_g1_reach_depends_on_height():
    """G1 的手在腰的高度伸得最远（约 0.7 米），越往上越近，1.55 米以上够不着。"""
    assert G1_REACH.reaches(0.7, 0.9)
    assert not G1_REACH.reaches(0.7, 1.45) and G1_REACH.reaches(0.4, 1.45)
    assert not G1_REACH.reaches(0.1, 1.6)
    assert "宇树 G1 在这个高度" in G1_REACH.limit_text(1.2)


def test_waypoint_inside_furniture_is_caught(raw):
    raw["rooms"][0]["waypoints"]["counter_front"]["position"] = [2.4, 0.95, 0.0]
    assert any("点位 counter_front：机器人站在这里会碰到" in p for p in check_geometry(parse_home(raw)))


def test_open_drawer_must_not_hit_the_robot(raw):
    """冷冻室抽屉拉出来要占地方：站得太近，拉开时会撞到自己。"""
    raw["rooms"][0]["waypoints"]["freezer_front"]["position"] = [1.3, 0.7, 0.0]
    assert any("拉开的冷冻室门会碰到站在这里的机器人" in p for p in check_geometry(parse_home(raw)))


def test_route_keeps_away_from_stove_and_wall():
    """从操作台走到水槽要绕开灶台前的禁区，也不会从柜子后面的墙里穿过去。"""
    home = load_home(SPEC)
    fm = FloorMap.of(home)
    wps = home.rooms["kitchen"].waypoints
    pts, why = fm.route(pose_of(wps["counter_front"]), pose_of(wps["sink_front"]))
    assert pts is not None, why
    stove = wps["stove_front"].position
    near = min(math.hypot(x - stove[0], y - stove[1]) for x, y in _trail_points(pts))
    assert near > FORBIDDEN_RADIUS + G1_FOOTPRINT.radius - 0.03
    assert max(y for _, y in pts) < 1.0


def test_cli_sim_check(capsys):
    assert main(["sim-check", str(SPEC)]) == 0
    assert "三维数据没问题" in capsys.readouterr().out


# ---------- 物理仿真 ----------

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

from robohome.sim.body import SimBody  # noqa: E402
from robohome.sim.floor import Box2D  # noqa: E402
from robohome.sim.render import Recorder  # noqa: E402
from robohome.sim.scene import Obstacle  # noqa: E402


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


def test_containers_sit_a_little_off_center():
    """容器像人随手放的一样偏几厘米、转几度，但还在自己的格子里；不加偏差就摆在正中间。"""
    world, body = sim_world()
    tidy = SimBody(fresh_world(SPEC), jitter_seed=None)
    offsets = []
    for cid in body.info.containers:
        c = world.home.containers[cid]
        assert body.locate(cid) == (c.location, c.slot)
        offsets.append(float(np.linalg.norm(body._container_pos(cid)[:2] - tidy._container_pos(cid)[:2])))
    assert max(offsets) > 0.01 and max(offsets) < 0.045


def test_pick_aims_at_the_tag_not_the_ledger(monkeypatch):
    """盒子被推偏了 4 厘米：看标签就照样吸得住；要是按台账的槽位中心去吸，就会吸空。"""
    world, body = sim_world()
    j = body.model.joint("fj_box_f1")
    body.data.qpos[j.qposadr[0]] += 0.04
    body._settle(0.3)
    steps = [{"skill": "navigate", "args": {"waypoint": "freezer_front"}},
             {"skill": "open", "args": {"fixture": "freezer_door"}},
             {"skill": "pick", "args": {"container": "box_f1"}}]
    report = _run(world, steps + [{"skill": "close", "args": {"fixture": "freezer_door"}},
                                  {"skill": "place", "args": {"container": "box_f1", "location": "freezer", "slot": 0}}])
    assert report.results[2].ok, report.render()

    world, body = sim_world()
    j = body.model.joint("fj_box_f1")
    body.data.qpos[j.qposadr[0]] += 0.04
    body._settle(0.3)
    area = body.info.areas["freezer"]
    monkeypatch.setattr(body, "_read_tag", lambda cid: body._to_world(area, area.slot_local(0)) + np.array([0, 0, 0.05]))
    report = _run(world, steps)
    assert not report.ok and "没对准冷冻盒 1 号" in report.summary
    assert world.robot.holding is None


def test_covered_tag_means_hands_off():
    """标签上盖了块布：相机认不出是哪个盒子，就不伸手，停下来告诉主人。"""
    world, body = sim_world("covered:box_f1")
    assert not world.covered_tags  # 交给物理：布真的挡在标签前面
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "freezer_front"}},
                          {"skill": "open", "args": {"fixture": "freezer_door"}},
                          {"skill": "pick", "args": {"container": "box_f1"}}])
    assert not report.ok and "看不清上面的标签" in report.summary
    assert world.robot.holding is None and body.locate("box_f1") == ("freezer", 0)
    assert world.home.fixtures["freezer_door"].state == "closed"  # 收尾关好了门


def test_back_row_is_read_from_the_lid():
    """冷冻室抽屉里后排的盒子，正面的标签被前排挡住了，从上面看盖子上的标签。"""
    world, body = sim_world()
    _do(world, [{"skill": "navigate", "args": {"waypoint": "freezer_front"}},
                {"skill": "open", "args": {"fixture": "freezer_door"}}])
    area = body.info.areas["freezer"]
    back = next(cid for cid in body.info.containers if world.home.containers[cid].location == "freezer"
                and area.slot_local(world.home.containers[cid].slot)[1] > 0)
    c = world.home.containers[back]
    pre, _ = body._approach_points(area, area.slot_local(c.slot), area.slot_local(c.slot)[2] + c.size_m[2])
    body._move_hand(pre, 0.3)
    assert body._read_tag(back) is not None
    body.info.tags[back] = body.info.tags[back][:1]  # 只剩正面的标签
    assert body._read_tag(back) is None


def test_walks_around_a_box_on_the_floor():
    """去水槽的路上多了个纸箱（规范里没有）：激光看到后绕过去，照样走到，身体不碰到它。"""
    box = Obstacle(1.7, 0.0, size_m=(0.3, 0.3, 0.4))
    world = fresh_world(SPEC)
    body = SimBody.attach(world, obstacles=[box])
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "sink_front"}}])
    assert report.ok, report.render()
    assert body.floor.seen  # 是激光看到的
    footprint = Box2D(box.x, box.y, 0.15, 0.15, 0.0, "纸箱")
    gap = min(footprint.distance(x, y) for x, y in _trail_points(body.trail))
    assert gap > body.footprint.radius - 0.03
    goal = world.home.rooms["kitchen"].waypoints["sink_front"].position
    assert math.hypot(body.base[0] - goal[0], body.base[1] - goal[1]) < 1e-6


def test_blocked_way_stops_and_goes_back():
    """去入库台的路被纸箱挡死了：走到一半看清楚了绕不过去，就退回出发的地方，告诉主人。"""
    world, body = sim_world("blocked:intake_front")
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "intake_front"}}])
    assert not report.ok and "去 intake_front 的路被挡住了" in report.summary and "激光" in report.summary
    assert world.robot.waypoint == "entrance"
    assert math.hypot(body.base[0], body.base[1]) < 0.02 and abs(math.sin(body.base[2])) < 0.02


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
    _do(world, [{"skill": "navigate", "args": {"waypoint": "freezer_front"}},
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
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "freezer_front"}},
                          {"skill": "inspect", "args": {"location": "freezer"}},
                          {"skill": "pick", "args": {"container": "box_f1"}}])
    assert all(r.ok for r in report.results), report.render()
    assert not report.photos[0].notes  # 照片和台账对得上


def test_door_really_swings_open():
    world, body = sim_world()
    _do(world, [{"skill": "navigate", "args": {"waypoint": "fridge_front"}},
                {"skill": "open", "args": {"fixture": "chiller_door"}}])
    assert body._joint_q("chiller_door") > 0.8 * body.info.door_open["chiller_door"]


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
    body.data.qpos[j.qposadr[0] : j.qposadr[0] + 3] = [-3.0, -3.0, 0.1]  # 拿到厨房外面去了
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
    report = _run(world, [{"skill": "navigate", "args": {"waypoint": "freezer_front"}},
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
    raw["rooms"][0]["waypoints"]["table_front"]["position"] = [2.6, -0.2, 0.0]  # 站得离餐桌太远
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

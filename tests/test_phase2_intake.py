"""第二阶段第 1 周：入库闭环。

主人把买回来的东西摆在入库台上 → 机器人扫描标签、识别 → 挑容器 → 逐件装盒（stow）、放进对应温区 → 回报。
认不出、放不下的留在台上告诉主人；温区不对、混装由安全规则拦下。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robohome.cli import main
from robohome.intake.core import NewItem, parcels_on_counter, put_parcels
from robohome.intake.recognize import KeywordRecognizer, LLMRecognizer, split_pieces
from robohome.mvp import TASKS, apply_fault, fresh_world, run_task
from robohome.planner import IntakePlanner, LLMPlanner, PlanningError, RulePlanner, Step
from robohome.planner.intake import is_intake
from robohome.scheduler import Scheduler
from robohome.skills.library import SKILLS, catalog
from robohome.spec import SpecError, load_home, parse_home, save_home

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"
INTAKE_TASKS = [t for t in TASKS if t.parcels]


@pytest.fixture
def raw():
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture
def world():
    return fresh_world(SPEC)


def run(world, planner, instruction):
    s = Scheduler(world, planner)
    s.submit(instruction)
    [report] = s.run_pending()
    return report


def execute(world, steps):
    return Scheduler(world, None).execute("手写计划", [Step(**s) for s in steps])  # type: ignore[arg-type]


class FakeBackend:
    """假的大模型：记下收到的提示，按给定的答案回。"""

    name = model = "fake"

    def __init__(self, data):
        self.data = data
        self.prompts: list[str] = []

    def user_message(self, text, image_path=None):
        return {"role": "user", "content": text}

    def complete_json(self, system, messages, schema, schema_name):
        self.prompts.append(messages[-1]["content"])
        return self.data, {}


def entry(n, name, temp, volume=1.0, category="肉类", aliases=(), readable=True):
    return {"parcel": n, "readable": readable, "name": name, "category": category, "temperature": temp,
            "volume_l": volume, "aliases": list(aliases)}


# ---------- 规范：入库台和包装 ----------


def test_sample_kitchen_has_an_empty_intake_counter(world):
    area = world.home.storage["intake_counter"]
    assert (area.kind, area.waypoint, area.slots) == ("intake", "intake_front", 6)
    assert parcels_on_counter(world) == []


def test_parcels_load_without_capacity_and_keep_their_label(raw):
    raw["containers"].append({"id": "p1", "name": "袋子", "kind": "parcel", "size_m": [0.2, 0.14, 0.06],
                              "location": "intake_counter", "label": "一斤五花肉"})
    home = parse_home(raw)
    p = home.containers["p1"]
    assert (p.capacity_l, p.label, p.home) == (0.0, "一斤五花肉", None)  # 入库台上的东西没有原位


def test_parcel_must_sit_on_an_intake_counter(raw):
    raw["containers"].append({"id": "p1", "name": "袋子", "kind": "parcel", "size_m": [0.2, 0.14, 0.06],
                              "location": "dining_table", "slot": 2, "label": "牛奶"})
    with pytest.raises(SpecError, match="只能放在入库台"):
        parse_home(raw)


def test_unknown_storage_kind_is_rejected(raw):
    raw["storage"][0]["kind"] = "shelf"
    with pytest.raises(SpecError, match="kind 只能是"):
        parse_home(raw)


def test_saved_spec_keeps_parcels_and_omits_empty_labels(world, tmp_path):
    put_parcels(world, ["一斤五花肉"])
    path = tmp_path / "home.json"
    save_home(world.home, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    box = next(c for c in data["containers"] if c["id"] == "box_f1")
    assert "label" not in box
    assert load_home(path).containers["parcel_1"].label == "一斤五花肉"


def test_put_parcels_fills_free_slots_and_refuses_too_many(world):
    ids = put_parcels(world, ["牛奶", "酸奶"])
    assert ids == ["parcel_1", "parcel_2"]
    assert [p.slot for p in parcels_on_counter(world)] == [0, 1]
    assert put_parcels(world, ["鸡蛋"]) == ["parcel_3"]
    with pytest.raises(ValueError, match="只有 3 个空位"):
        put_parcels(world, ["苹果"] * 4)


def test_put_parcels_needs_an_intake_counter(raw):
    raw["storage"] = [s for s in raw["storage"] if s["id"] != "intake_counter"]
    world = fresh_world_from(raw)
    with pytest.raises(ValueError, match="没有登记入库台"):
        put_parcels(world, ["牛奶"])


def fresh_world_from(raw):
    from robohome.world import World

    return World.from_home(parse_home(raw), start_waypoint="entrance")


# ---------- 识别标签 ----------


def test_split_pieces():
    assert split_pieces("刚从超市回来，买了鸡翅、牛奶、西兰花、酸奶和面粉") == ["鸡翅", "牛奶", "西兰花", "酸奶", "面粉"]
    assert split_pieces("今天买了一斤五花肉、两盒酸奶和一袋苹果") == ["一斤五花肉", "两盒酸奶", "一袋苹果"]
    assert split_pieces("") == []


def test_keyword_recognizer_keeps_what_the_label_says_as_alias():
    items, _ = KeywordRecognizer().recognize("一斤五花肉、虾仁、牛奶")
    assert [(i.name, i.aliases) for i in items] == [("猪肉", ["五花肉"]), ("虾", ["虾仁"]), ("牛奶", [])]


def test_keyword_labels_one_item_per_parcel():
    out = KeywordRecognizer().recognize_labels(["一斤五花肉", None, "一包薯片", "牛奶和鸡蛋"])
    assert out[0].name == "猪肉" and out[0].temperature == "frozen"
    assert out[1:] == [None, None, None]  # 看不清、认不出、一个标签写了两样东西：都不猜


def test_llm_labels_are_matched_by_parcel_number():
    backend = FakeBackend({"parcels": [
        entry(2, "酸奶", "chilled", 1.0, "乳制品"),
        entry(1, "五花肉", "frozen", 0.5, aliases=["猪肉"]),
        entry(1, "牛奶", "chilled"),  # 同一个编号第二次出现：不要
        entry(9, "苹果", "chilled"),  # 没有这个编号
        entry(4, "薯片", "ambient", readable=False),
    ]})
    out = LLMRecognizer(backend).recognize_labels(["一斤五花肉", "两盒酸奶", None, "一包东西"], "今天买的", ["肉类"])
    assert [(i.name, i.volume_l) if i else None for i in out] == [("五花肉", 0.5), ("酸奶", 1.0), None, None]
    prompt = backend.prompts[0]
    assert "1. 一斤五花肉" in prompt and "4. 一包东西" in prompt and "3." not in prompt  # 看不清的不发给模型
    assert "主人说：今天买的" in prompt


def test_llm_labels_reject_bad_entries():
    backend = FakeBackend({"parcels": [entry(1, "五花肉", "warm"), entry(2, "酸奶", "chilled", volume=-1)]})
    assert LLMRecognizer(backend).recognize_labels(["一斤五花肉", "两盒酸奶"]) == [None, None]


def test_llm_labels_skip_the_model_when_nothing_is_readable():
    backend = FakeBackend({"parcels": []})
    assert LLMRecognizer(backend).recognize_labels([None, None]) == [None, None]
    assert backend.prompts == []


def test_llm_labels_bad_structure_is_an_error():
    with pytest.raises(RuntimeError, match="结构不对"):
        LLMRecognizer(FakeBackend({"items": []})).recognize_labels(["牛奶"])


# ---------- 入库规划 ----------


@pytest.mark.parametrize("text", ["今天买了牛奶", "帮我入库", "刚从超市回来", "新买的鸡蛋"])
def test_intake_phrases(text):
    assert is_intake(text)


@pytest.mark.parametrize("text", [t.instruction for t in TASKS if not t.parcels])
def test_other_tasks_are_not_intake(text):
    assert not is_intake(text)


def test_intake_plan_inspects_then_stows_each_parcel(world):
    put_parcels(world, ["一斤五花肉", "两盒酸奶"])
    steps = RulePlanner().plan("今天买了一斤五花肉和两盒酸奶", world)
    skills = [s.skill for s in steps]
    assert skills[:2] == ["navigate", "inspect"] and skills.count("stow") == 2
    stows = [s.args for s in steps if s.skill == "stow"]
    assert [(a["parcel"], a["container"], a["temperature"]) for a in stows] == [
        ("parcel_1", "box_f1", "frozen"),  # 和牛肉同是肉类
        ("parcel_2", "box_c1", "chilled"),  # 和牛奶同是乳制品
    ]
    assert world.home.find_item("五花肉") is None  # 规划不记账，装好才记


def test_intake_end_to_end_updates_ledger_and_clears_counter(world):
    put_parcels(world, ["一斤五花肉", "一袋苹果"])
    report = run(world, RulePlanner(), "今天买了一斤五花肉和一袋苹果")
    assert report.ok, report.render()
    assert parcels_on_counter(world) == [] and not world.opened_by_robot and world.robot.holding is None
    pork = world.home.find_item("五花肉")
    assert (pork.name, pork.container, pork.volume_l) == ("猪肉", "box_f1", 1.0)
    assert world.home.find_item("苹果").container == "box_c3"  # 水果开一个空的冷藏盒，不和牛奶混放
    assert "扫描包装 1 号「一斤五花肉」：猪肉，装进冷冻盒 1 号" in report.render()


def test_later_parcels_see_space_taken_by_earlier_ones(world):
    # 冷冻盒 1 号装着 1 升牛肉，还剩 3.5 升：第一件 3 升的肉进去后，第二件只能开空盒
    put_parcels(world, ["三斤排骨", "一斤牛肉"])
    planner = IntakePlanner(KeywordRecognizer())
    planner.plan("入库", world)
    assert [d.container for d in planner.decisions] == ["box_f1", "box_f3"]


def test_unrecognized_and_oversize_parcels_stay_and_owner_is_told(world):
    put_parcels(world, ["一斤五花肉", "一包薯片", "两箱牛奶"])
    apply_fault(world, "unreadable:parcel_1")
    put_parcels(world, ["一盒鸡蛋"])
    report = run(world, RulePlanner(), "帮我入库")
    assert report.ok, report.render()
    assert [p.id for p in parcels_on_counter(world)] == ["parcel_1", "parcel_2", "parcel_3"]
    notices = "\n".join(report.notices)
    assert "包装 1 号的标签看不清" in notices
    assert "包装 2 号「一包薯片」认不出是什么" in notices
    assert "包装 3 号「两箱牛奶」有 12 升，一个容器装不下，机器人不拆包装" in notices and "请主人拆开分装" in notices
    assert "parcel_4" not in world.home.containers  # 鸡蛋照常装好
    assert sum(i.name == "鸡蛋" and i.container == "box_c2" for i in world.home.items.values()) == 2


def test_full_zone_asks_owner_to_make_room(world):
    put_parcels(world, ["一袋青菜", "一袋苹果", "一袋橙子", "一袋葡萄"])
    report = run(world, RulePlanner(), "入库")
    assert report.ok
    # 冷藏室只有冷藏盒 3 号是空的，青菜占了之后水果没地方放
    assert any("冷藏区的容器都满了" in n and "请主人清理或添置容器" in n for n in report.notices), report.notices


def test_nothing_placeable_is_a_planning_failure(world):
    put_parcels(world, ["一包薯片"])
    with pytest.raises(PlanningError, match="都放不了：包装 1 号「一包薯片」认不出是什么"):
        RulePlanner().plan("入库", world)


def test_empty_counter_is_a_planning_failure(world):
    with pytest.raises(PlanningError, match="入库台上没有"):
        RulePlanner().plan("今天买了牛奶", world)


def test_llm_planner_hands_intake_to_the_recognizer(world):
    put_parcels(world, ["两盒酸奶"])
    backend = FakeBackend({"parcels": [entry(1, "酸奶", "chilled", 1.0, "乳制品", ["酸奶"])]})
    planner = LLMPlanner(backend)
    report = run(world, planner, "今天买了两盒酸奶")
    assert report.ok, report.render()
    assert world.home.find_item("酸奶").container == "box_c1"
    assert planner.attempts[0]["model"] == ["parcel_1「两盒酸奶」→ 酸奶（乳制品，冷藏，1 升） → box_c1"]
    assert "入库台上的包装" in backend.prompts[0]  # 问的是标签，不是让模型写计划


def test_llm_planner_reports_recognition_failure(world):
    put_parcels(world, ["一包东西"])
    planner = LLMPlanner(FakeBackend({"parcels": [entry(1, "?", "ambient", readable=False)]}))
    report = run(world, planner, "入库")
    assert not report.ok and "都放不了" in report.summary
    assert planner.attempts[0]["error"]


# ---------- 装盒技能和安全规则 ----------


def stow_args(**kw):
    args = {"parcel": "parcel_1", "container": "box_f1", "name": "猪肉", "category": "肉类",
            "temperature": "frozen", "volume_l": 1.0, "aliases": ["五花肉"]}
    return {**args, **kw}


def hold_parcel_at_fridge(world, door="freezer_door"):
    put_parcels(world, ["一斤五花肉"])
    steps = [
        {"skill": "navigate", "args": {"waypoint": "intake_front"}},
        {"skill": "pick", "args": {"container": "parcel_1"}},
        {"skill": "navigate", "args": {"waypoint": "fridge_front"}},
    ]
    if door:
        steps.append({"skill": "open", "args": {"fixture": door}})
    return steps


def test_stow_and_notify_are_not_offered_to_the_model():
    names = {s["name"] for s in catalog()}
    assert "stow" in SKILLS and "notify" in SKILLS and not {"stow", "notify"} & names


def test_stow_into_wrong_zone_is_blocked_by_safety(world):
    steps = hold_parcel_at_fridge(world, "chiller_door") + [{"skill": "stow", "args": stow_args(container="box_c3")}]
    report = execute(world, steps)
    assert not report.ok and report.results[-1].stage == "safety"
    assert "猪肉要冷冻，冷藏室是冷藏的，温区不对" in report.summary
    assert world.home.find_item("五花肉") is None
    assert any("手里还拿着包装 1 号" in n for n in report.needs_attention)


def test_stow_never_mixes_categories(world):
    steps = hold_parcel_at_fridge(world) + [{"skill": "stow", "args": stow_args(container="box_f2")}]
    report = execute(world, steps)
    assert not report.ok and "冷冻盒 2 号里装的是速冻，不能再装肉类，不混放" in report.summary


@pytest.mark.parametrize("args,reason", [
    (stow_args(volume_l=4.0), "只剩 3.5 升，装不下 4 升"),
    (stow_args(container="bowl_1"), "不是放在存放区里的食物容器"),
    (stow_args(container="box_x"), "容器 box_x 不存在"),
    (stow_args(temperature="warm"), "没说清楚"),
    (stow_args(volume_l=0), "体积不对"),
])
def test_stow_preconditions(world, args, reason):
    report = execute(world, hold_parcel_at_fridge(world) + [{"skill": "stow", "args": args}])
    assert not report.ok and report.results[-1].stage in ("precondition", "safety") and reason in report.summary


def test_stow_needs_the_door_open_and_the_parcel_in_hand(world):
    report = execute(world, hold_parcel_at_fridge(world, door=None) + [{"skill": "stow", "args": stow_args()}])
    assert "冷冻室门没开" in report.summary
    w2 = fresh_world(SPEC)
    put_parcels(w2, ["一斤五花肉"])
    report = execute(w2, [{"skill": "stow", "args": stow_args()}])
    assert "手里没有拿着包装 1 号" in report.summary


def test_notify_needs_a_message(world):
    report = execute(world, [{"skill": "notify", "args": {"message": " "}}])
    assert not report.ok and "提醒内容是空的" in report.summary


def test_unreadable_fault_only_applies_to_parcels(world):
    with pytest.raises(ValueError, match="入库台上没有这件包装"):
        apply_fault(world, "unreadable:box_f1")


# ---------- 验收检查本身要可靠 ----------


def intake1_world():
    world = fresh_world(SPEC)
    put_parcels(world, [label for label, _ in INTAKE_TASKS[0].parcels])
    return world


def test_check_flags_wrong_zone_and_unfindable_name():
    world = intake1_world()

    class Confused(KeywordRecognizer):  # 把五花肉认成要冷藏的，名字也起得和标签对不上
        def recognize_labels(self, labels, instruction="", categories=None):
            out = super().recognize_labels(labels, instruction, categories)
            out[0] = NewItem("猪五花", "冷藏肉", "chilled", 1.0)
            return out

    report = run(world, IntakePlanner(Confused()), "今天买了")
    assert report.ok, report.render()  # 执行本身没毛病：温区和容器是对得上的
    problems = INTAKE_TASKS[0].check(world, report)
    assert any("「一斤五花肉」放进了冷藏室，应该冷冻" in p for p in problems)
    assert any("登记成了猪五花，以后按包装上的叫法找不到" in p for p in problems)


def test_check_flags_parcels_left_silently():
    world = intake1_world()
    steps = [Step("navigate", {"waypoint": "intake_front"}), Step("inspect", {"location": "intake_counter"})]
    report = Scheduler(world, None).execute("今天买了", steps)  # type: ignore[arg-type]  # 只看了一眼，什么都没做
    problems = INTAKE_TASKS[0].check(world, report)
    assert sum("也没告诉主人" in p for p in problems) == 3


def test_intake_tasks_pass_with_keyword_recognizer():
    for task in INTAKE_TASKS:
        result = run_task(task, RulePlanner(), SPEC)
        assert result.passed, (result.problems, result.report.render())


# ---------- 命令行 ----------


def test_cli_intake_run(capsys):
    assert main(["intake", "今天买了一斤五花肉、两盒酸奶和一袋苹果", "--run"]) == 0
    out = capsys.readouterr().out
    assert "parcel_1「一斤五花肉」→ 猪肉（肉类，冷冻，1 升） → box_f1" in out
    assert "完成：共 19 步" in out


def test_cli_intake_run_reports_leftovers_and_saves(tmp_path, capsys):
    spec = tmp_path / "home.json"
    spec.write_text(SPEC.read_text(encoding="utf-8"), encoding="utf-8")
    code = main(["intake", "买了两箱牛奶和一斤五花肉", "--run", "--spec", str(spec), "--save", str(spec), "--yes"])
    out = capsys.readouterr().out
    assert code == 1 and "请主人拆开分装" in out and "台账已写入" in out
    home = load_home(spec)
    assert home.find_item("五花肉").container == "box_f1"
    assert [c.label for c in home.containers.values() if c.kind == "parcel"] == ["两箱牛奶"]  # 留在台上的也记着
    # 主人拆开分装后再说一次"入库"，台上剩下的会接着处理（这里还是整箱，所以仍然留着）
    assert main(["intake", "--run", "--spec", str(spec)]) == 1


def test_cli_intake_run_needs_something_to_do(capsys):
    assert main(["intake", "--run"]) == 2
    assert "没说买了什么" in capsys.readouterr().err


def test_cli_run_routes_intake(tmp_path, capsys):
    spec = tmp_path / "home.json"
    world = fresh_world(SPEC)
    put_parcels(world, ["一盒鸡蛋"])
    save_home(world.home, spec)
    assert main(["run", "入库", "--spec", str(spec)]) == 0
    assert "装进冷藏盒 2 号" in capsys.readouterr().out


# ---------- 仿真 ----------


def test_stow_in_simulation_absorbs_the_parcel(world):
    pytest.importorskip("mujoco")
    from robohome.sim.body import SimBody

    put_parcels(world, ["一斤五花肉"])
    body = SimBody.attach(world)
    report = run(world, RulePlanner(), "今天买了一斤五花肉")
    assert report.ok, report.render()
    assert "parcel_1" not in body.info.containers and world.home.find_item("五花肉").container == "box_f1"


def test_stow_in_simulation_fails_when_container_is_not_there(world):
    pytest.importorskip("mujoco")
    from robohome.sim.body import SimBody

    put_parcels(world, ["一斤五花肉"])
    apply_fault(world, "missing:box_f1")
    SimBody.attach(world)
    report = run(world, RulePlanner(), "今天买了一斤五花肉")
    assert not report.ok and "没找到冷冻盒 1 号" in report.summary and "仿真" in report.summary
    assert world.robot.holding == "parcel_1" and world.home.find_item("五花肉") is None
    assert any("手里还拿着包装 1 号" in n for n in report.needs_attention)


def test_sim_check_covers_the_intake_counter():
    from robohome.sim.layout import areas

    area = areas(load_home(SPEC))["intake_counter"]
    assert area.style == "surface" and area.waypoint == "intake_front"


def test_parcels_are_not_offered_as_food_containers(world):
    from robohome.intake.core import choose_container

    put_parcels(world, ["牛奶"])
    container, _ = choose_container(world, NewItem("苹果", "水果", "ambient", 1.0))
    assert container == "bin_a2"


# ---------- 类别对齐台账 ----------


@pytest.mark.parametrize("name,category,aliases,want", [
    ("饺子", "速冻食品", [], "速冻"),  # 台账里有"速冻饺子"，别名就是饺子
    ("牛奶", "饮品", [], "乳制品"),  # 台账里已经有牛奶
    ("酸奶", "乳制品类", [], "乳制品"),  # 和已有类别互相包含
    ("五花肉", "冷冻肉类", [], "肉类"),
    ("虾仁", "海鲜", [], "海鲜"),  # 新类别保留
    ("挂面", "面", [], "面"),  # 一个字的类别不去猜
])
def test_settle_category(world, name, category, aliases, want):
    from robohome.intake.core import settle_category

    assert settle_category(world, NewItem(name, category, "chilled", 1.0, aliases)).category == want


def test_llm_category_variants_still_share_a_container(world):
    put_parcels(world, ["一袋饺子"])
    backend = FakeBackend({"parcels": [entry(1, "饺子", "frozen", 1.5, "速冻食品")]})
    planner = IntakePlanner(LLMRecognizer(backend))
    planner.plan("入库", world)
    assert planner.decisions[0].container == "box_f2" and planner.decisions[0].item.category == "速冻"


# ---------- 本地模型思考打转 ----------


def _sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def _reasoning_server(thoughts, content='{"parcels": []}'):
    from test_local_llm import StubServer

    def script(handler, n):
        try:
            for t in thoughts:
                handler.wfile.write(_sse({"choices": [{"delta": {"content": "", "reasoning": t}, "finish_reason": None}]}))
            handler.wfile.write(_sse({"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]}))
            handler.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端发现打转后提前断开

    return StubServer([], script=script)


def test_local_backend_gives_up_when_thinking_loops():
    """Mac 上 qwen3:8b 实测：为"苹果归哪个类别"反复念同一句话，5 分钟没出正式答案。"""
    from robohome.llm import LLMError, LocalJSON

    loop = "最终，可能将苹果归入“生鲜”类别，但用户提供的类别中没有，所以可能需要归入“其他”。"
    server = _reasoning_server(["先看五花肉，应该冷冻。" * 3] + [loop] * 200)
    try:
        backend = LocalJSON(base_url=server.url, retries=1)
        started = time.monotonic()
        with pytest.raises(LLMError, match="思考陷入重复"):
            backend.complete_json("s", [backend.user_message("u")], {"type": "object"}, "x")
        assert time.monotonic() - started < 5
        assert len(server.requests) == 1  # 温度是 0，重试也一样，不重试
    finally:
        server.close()


def test_local_backend_accepts_normal_thinking():
    from robohome.llm import LocalJSON

    thoughts = [f"第 {n} 件：看标签，判断温区和体积。" for n in range(1, 60)]
    server = _reasoning_server(thoughts, '{"ok": true}')
    try:
        data, _ = LocalJSON(base_url=server.url).complete_json("s", [{"role": "user", "content": "u"}], {"type": "object"}, "x")
        assert data == {"ok": True}
    finally:
        server.close()


def test_local_backend_caps_very_long_thinking():
    from robohome.llm import LLMError, LocalJSON

    thoughts = [f"想法 {n}：{'这件东西可能是什么' * 3}{n * 7919}" for n in range(2000)]
    server = _reasoning_server(thoughts)
    try:
        with pytest.raises(LLMError, match="思考超过 3000 字"):
            LocalJSON(base_url=server.url, max_reasoning_chars=3000).complete_json(
                "s", [{"role": "user", "content": "u"}], {"type": "object"}, "x")
    finally:
        server.close()

"""第 3 步（上）：按 PR #2 审阅意见补的加固。

覆盖：加热时长范围、失败后安全收尾、家电状态读不到时的保守处理、入库识别与分装、
本地模型流式连接的各种异常、Claude 后端的报错、命令行的预览与防误写。
"""

import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from robohome.appliances.base import MockAppliance
from robohome.cli import DEFAULT_SPEC, main
from robohome.intake import KeywordRecognizer, LLMRecognizer, NewItem, assign
from robohome.intake.recognize import validate_items
from robohome.llm import ClaudeJSON, LLMError, LocalJSON
from robohome.planner import LLMPlanner, Step
from robohome.planner.context import describe_world
from robohome.scheduler import Scheduler, simulate
from robohome.scheduler.safety import DEFAULT_RULES
from robohome.spec import load_home
from robohome.world import World

from test_local_llm import StubServer, plan_text
from test_step2 import GOOD_PLAN

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"


@pytest.fixture
def world():
    return World.from_home(load_home(SPEC), start_waypoint="entrance")


def steps(raw):
    return [Step(**s) for s in raw]


def with_minutes(minutes):
    plan = [dict(s) for s in GOOD_PLAN]
    plan[-1] = {"skill": "appliance",
                "args": {"appliance": "microwave", "action": "defrost", "params": {"minutes": minutes}}}
    return steps(plan)


class FixedPlanner:
    def __init__(self, plan):
        self.plan_steps = plan

    def plan(self, instruction, world):
        return self.plan_steps


# ---------- 加热时长 ----------


@pytest.mark.parametrize("minutes", [True, 0, -3, 31, 8.5, "8"])
def test_bad_run_time_is_rejected(world, minutes):
    report = simulate(world, with_minutes(minutes))
    assert not report.ok
    assert report.results[-1].stage == "precondition"
    assert "分钟" in report.results[-1].message


@pytest.mark.parametrize("minutes", [1, 8, 30])
def test_run_time_within_range_is_accepted(world, minutes):
    assert simulate(world, with_minutes(minutes)).ok


def test_max_minutes_comes_from_spec(world):
    world.home.appliances["microwave"].max_minutes = 5
    report = simulate(world, with_minutes(8))
    assert not report.ok and "1 到 5 分钟" in report.results[-1].message


def test_stop_needs_no_run_time(world):
    world.appliances["microwave"].running = True
    s = Scheduler(world, FixedPlanner(steps([
        {"skill": "navigate", "args": {"waypoint": "counter_front"}},
        {"skill": "appliance", "args": {"appliance": "microwave", "action": "stop", "params": {}}},
    ])))
    s.submit("停下微波炉")
    [report] = s.run_pending()
    assert report.ok, report.render()
    assert not world.appliances["microwave"].running


# ---------- 失败后的安全收尾 ----------


def test_failure_closes_door_the_robot_opened(world):
    world.missing_containers.add("box_f1")
    s = Scheduler(world, FixedPlanner(with_minutes(8)))
    s.submit("把牛肉拿出来解冻")
    [report] = s.run_pending()
    assert not report.ok
    assert report.results[-1].stage == "execution"  # 去拿时发现盒子不在
    assert [r.step.skill for r in report.cleanup] == ["close"]
    assert all(r.ok for r in report.cleanup)
    assert report.needs_attention == []
    assert world.home.fixtures["freezer_door"].state == "closed"
    assert world.opened_by_robot == set()
    assert "收尾 ✓" in report.render()


def test_failure_reports_door_it_could_not_close(world):
    world.missing_containers.add("box_f1")
    jammed = lambda w, skill, args: "门被挡住了" if skill == "close" else None
    s = Scheduler(world, FixedPlanner(with_minutes(8)), rules=[*DEFAULT_RULES, jammed])
    s.submit("把牛肉拿出来解冻")
    [report] = s.run_pending()
    assert not report.ok
    assert report.cleanup and not report.cleanup[-1].ok
    assert report.needs_attention == ["冷冻室门"]
    assert "需要人处理：冷冻室门还开着" in report.render()


def test_plan_that_leaves_door_open_fails_and_is_cleaned_up(world):
    s = Scheduler(world, FixedPlanner(steps([
        {"skill": "navigate", "args": {"waypoint": "fridge_front"}},
        {"skill": "open", "args": {"fixture": "freezer_door"}},
    ])))
    s.submit("打开冷冻室")
    [report] = s.run_pending()
    assert not report.ok and "还开着" in report.summary
    assert world.home.fixtures["freezer_door"].state == "closed"


def test_offline_appliance_after_doors_closed_needs_no_cleanup(world):
    world.appliances["microwave"].online = False
    s = Scheduler(world, FixedPlanner(with_minutes(8)))
    s.submit("把牛肉拿出来解冻")
    [report] = s.run_pending()
    assert not report.ok and "离线" in report.summary
    assert report.cleanup == [] and report.needs_attention == []


def test_simulate_does_not_clean_up_or_touch_real_world(world):
    world.missing_containers.add("box_f1")  # 预演看不到这个故障
    report = simulate(world, steps(GOOD_PLAN[:2]))
    assert not report.ok and report.cleanup == []
    assert world.home.fixtures["freezer_door"].state == "closed"


def test_cli_json_includes_cleanup(capsys):
    code = main(["run", "把牛肉拿出来解冻", "--fault", "missing:box_f1", "--json"])
    assert code == 1
    [report] = json.loads(capsys.readouterr().out)
    assert report["cleanup"] and report["cleanup"][0]["ok"]
    assert report["needs_attention"] == []


# ---------- 家电状态读不到 ----------


class BrokenAppliance(MockAppliance):
    def status(self):
        raise ConnectionError("家电网关没响应")


def test_unreadable_hazard_appliance_is_not_touched(world, caplog):
    world.appliances["microwave"] = BrokenAppliance(appliance_id="microwave", capabilities=["defrost", "heat", "stop"])
    s = Scheduler(world, FixedPlanner(with_minutes(8)))
    s.submit("把牛肉拿出来解冻")
    with caplog.at_level(logging.WARNING):
        [report] = s.run_pending()
    failed = report.results[-1]
    assert failed.stage == "safety" and "读不到微波炉的状态" in failed.message
    assert "读不到 microwave" in caplog.text
    assert "读不到状态" in report.final_state["appliances"]["microwave"]["error"]


def test_sandbox_treats_unreadable_appliance_as_offline(world, caplog):
    world.appliances["microwave"] = BrokenAppliance(appliance_id="microwave", capabilities=["defrost"])
    with caplog.at_level(logging.WARNING):
        sandbox = world.sandbox()
    assert sandbox.appliances["microwave"].online is False
    assert "预演时读不到 microwave" in caplog.text


def test_sandbox_copies_offline_state(world):
    world.appliances["microwave"].online = False
    assert world.sandbox().appliances["microwave"].online is False
    report = simulate(world, with_minutes(8))
    assert not report.ok and "离线" in report.results[-1].message


def test_context_tells_model_about_doors_limits_and_status(world):
    world.opened_by_robot.add("freezer_door")
    world.appliances["microwave"].online = False
    ctx = describe_world(world)
    assert ctx["robot"]["opened_by_robot"] == ["freezer_door"]
    microwave = next(a for a in ctx["appliances"] if a["id"] == "microwave")
    assert microwave["max_minutes"] == 30
    assert microwave["online"] is False and microwave["running"] is False


# ---------- 入库：关键词识别 ----------


@pytest.mark.parametrize("text,expected", [
    ("一箱牛奶", [("牛奶", 6.0)]),
    ("两盒酸奶", [("酸奶", 1.0)]),
    ("3斤五花肉", [("猪肉", 3.0)]),
    ("从超市回来买了虾", [("虾", 1.0)]),
    ("米", [("大米", 2.5)]),
    ("一打鸡蛋", [("鸡蛋", 18.0)]),
])
def test_keyword_quantities(text, expected):
    items, unknown = KeywordRecognizer().recognize(text)
    assert [(i.name, i.volume_l) for i in items] == expected
    assert unknown == []


def test_keyword_refuses_ambiguous_words():
    items, unknown = KeywordRecognizer().recognize("糖醋里脊、玉米、牛奶糖、米饭")
    assert items == []
    assert unknown == ["糖醋里脊", "玉米", "牛奶糖", "米饭"]


def test_keyword_single_char_needs_exact_match():
    items, unknown = KeywordRecognizer().recognize("鱼丸和盐")
    assert [i.name for i in items] == ["调料"]
    assert unknown == ["鱼丸"]


# ---------- 入库：大模型识别结果的检查 ----------


def test_validate_items_rejects_bad_entries():
    items, rejected = validate_items([
        {"name": "牛奶", "category": "乳制品", "temperature": "chilled", "volume_l": 1, "aliases": []},
        {"name": "", "category": "x", "temperature": "chilled", "volume_l": 1},
        {"name": "冰棍", "category": "冷饮", "temperature": "cold", "volume_l": 0.5},
        {"name": "西瓜", "category": "水果", "temperature": "chilled", "volume_l": -2},
        {"name": "整只羊", "category": "肉类", "temperature": "frozen", "volume_l": 80},
        {"name": "面包", "category": "", "temperature": "ambient", "volume_l": "abc"},
        {"name": "黄油", "category": "", "temperature": "chilled", "volume_l": "0.25", "aliases": ["", "牛油"]},
    ])
    assert [(i.name, i.category, i.volume_l, i.aliases) for i in items] == [
        ("牛奶", "乳制品", 1.0, []),
        ("黄油", "未分类", 0.25, ["牛油"]),
    ]
    assert rejected == [
        "（没有名字的物品）",
        "冰棍（温区不明：cold）",
        "西瓜（体积不合理：-2 升）",
        "整只羊（体积不合理：80 升）",
        "面包（体积不合理：abc 升）",
    ]


class FakeBackend:
    model = "text-only"

    def __init__(self, data=None, vision=None):
        self.data = data
        self.vision = vision
        self.calls = 0

    def supports_vision(self):
        return self.vision

    def user_message(self, text, image_path=None):
        return {"role": "user", "content": text}

    def complete_json(self, system, messages, schema, schema_name):
        self.calls += 1
        return self.data, {}


def test_llm_recognizer_refuses_empty_input():
    backend = FakeBackend({"items": [{"name": "牛奶"}], "unrecognized": []})
    with pytest.raises(RuntimeError, match="既没有文字也没有照片"):
        LLMRecognizer(backend).recognize("   ")
    assert backend.calls == 0  # 没有去问模型，避免它凭空编物品


def test_llm_recognizer_refuses_photo_for_text_only_model(tmp_path):
    img = tmp_path / "bag.jpg"
    img.write_bytes(b"jpg")
    backend = FakeBackend(vision=False)
    with pytest.raises(RuntimeError, match="不支持看图"):
        LLMRecognizer(backend).recognize("", str(img))
    assert backend.calls == 0


def test_llm_recognizer_rejects_wrong_structure():
    with pytest.raises(RuntimeError, match="结构不对"):
        LLMRecognizer(FakeBackend({"things": []})).recognize("买了牛奶")


def test_llm_recognizer_passes_rejected_items_to_owner():
    data = {"items": [{"name": "冰棍", "category": "冷饮", "temperature": "cold", "volume_l": 0.5, "aliases": []}],
            "unrecognized": ["一个看不清的袋子"]}
    items, unknown = LLMRecognizer(FakeBackend(data)).recognize("买了冰棍")
    assert items == []
    assert unknown == ["一个看不清的袋子", "冰棍（温区不明：cold）"]


# ---------- 入库：分装 ----------


def milk(volume):
    return NewItem("牛奶", "乳制品", "chilled", volume)


def test_too_big_item_is_split_across_containers(world):
    result = assign(world, [milk(12.0)])
    assert result.ok
    assert [(p.container, p.item.volume_l) for p in result.placements] == [("box_c3", 7.2), ("box_c1", 4.8)]
    assert "分装 1/2" in result.placements[0].note
    stored = sum(i.volume_l for i in world.home.items.values() if i.name == "牛奶")
    assert stored == pytest.approx(2.0 + 12.0)


def test_split_reports_what_does_not_fit(world):
    result = assign(world, [milk(18.0)])
    assert not result.ok
    placed = [p for p in result.placements if p.ok]
    assert sum(p.item.volume_l for p in placed) == pytest.approx(12.4)
    assert "还剩 5.6 升放不下" in result.placements[-1].note


def test_split_never_mixes_categories(world):
    result = assign(world, [NewItem("酸奶", "酸奶类", "chilled", 10.0)])
    # 只有冷藏盒 3 号是空的，另外两个分别装着牛奶和鸡蛋，不能混装
    assert not result.ok
    assert [p.container for p in result.placements] == [None]
    assert len(world.home.items) == 5  # 整件放不下就不拆，台账不动


def test_preview_on_sandbox_leaves_ledger_alone(world):
    preview = assign(world.sandbox(), [milk(1.0)])
    assert preview.ok
    assert len(world.home.items) == 5


# ---------- 本地模型：流式连接的各种异常 ----------


def sse(chunk):
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()


def content_chunks(text, finish="stop"):
    pieces = [text[i:i + 40] for i in range(0, len(text), 40)]
    for n, piece in enumerate(pieces):
        yield sse({"choices": [{"delta": {"content": piece}, "finish_reason": finish if n == len(pieces) - 1 else None}]})


def send_ok(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()


GOOD = plan_text(GOOD_PLAN)


def test_dropped_connection_is_retried(world):
    def script(h, n):
        send_ok(h)
        chunks = list(content_chunks(GOOD))
        if n == 1:
            h.wfile.write(chunks[0])  # 只发了一截就断开：没有 finish_reason，也没有 [DONE]
            return
        for c in chunks:
            h.wfile.write(c)
        h.wfile.write(b"data: [DONE]\n\n")

    server = StubServer([], script=script)
    try:
        result = LLMPlanner(LocalJSON(base_url=server.url, idle_timeout=2)).plan("把牛肉拿出来解冻", world)
    finally:
        server.close()
    assert len(result) == len(GOOD_PLAN)
    assert len(server.requests) == 2


def test_stream_that_never_finishes_fails_after_retries():
    def script(h, n):
        send_ok(h)
        h.wfile.write(next(content_chunks('{"feasible": true', finish=None)))

    server = StubServer([], script=script)
    try:
        with pytest.raises(LLMError, match="没有正常结束.*已重试 2 次"):
            LocalJSON(base_url=server.url, retries=2, idle_timeout=2).complete_json("s", [], {}, "t")
    finally:
        server.close()
    assert len(server.requests) == 3


def test_finish_reason_without_done_is_accepted():
    def script(h, n):
        send_ok(h)
        for c in content_chunks('{"ok": true}'):
            h.wfile.write(c)

    server = StubServer([], script=script)
    try:
        data, _ = LocalJSON(base_url=server.url, idle_timeout=2).complete_json("s", [], {}, "t")
    finally:
        server.close()
    assert data == {"ok": True}


def test_usage_chunk_and_comments_are_skipped():
    def script(h, n):
        send_ok(h)
        h.wfile.write(b": keep-alive\n\n")
        for c in content_chunks('{"ok": 1}'):
            h.wfile.write(c)
        h.wfile.write(sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}))
        h.wfile.write(b"data: [DONE]\n\n")

    server = StubServer([], script=script)
    try:
        data, _ = LocalJSON(base_url=server.url, idle_timeout=2).complete_json("s", [], {}, "t")
    finally:
        server.close()
    assert data == {"ok": 1}


def test_error_event_in_stream_is_not_retried():
    def script(h, n):
        send_ok(h)
        h.wfile.write(sse({"error": {"message": "model requires more system memory"}}))

    server = StubServer([], script=script)
    try:
        with pytest.raises(LLMError, match="more system memory"):
            LocalJSON(base_url=server.url, idle_timeout=2).complete_json("s", [], {}, "t")
    finally:
        server.close()
    assert len(server.requests) == 1


def test_total_timeout_is_not_retried():
    def script(h, n):
        send_ok(h)
        for _ in range(20):  # 一直慢慢出字，但永远写不完
            h.wfile.write(sse({"choices": [{"delta": {"content": " "}, "finish_reason": None}]}))
            h.wfile.flush()
            time.sleep(0.1)

    server = StubServer([], script=script)
    try:
        started = time.monotonic()
        with pytest.raises(LLMError, match="秒内没有完成"):
            LocalJSON(base_url=server.url, idle_timeout=1, total_timeout=0.5, retries=3).complete_json("s", [], {}, "t")
        assert time.monotonic() - started < 1.5
    finally:
        server.close()
    assert len(server.requests) == 1


def test_cold_start_waits_longer_for_first_byte():
    def script(h, n):
        time.sleep(1.0)  # 模型加载中，迟迟不出第一个字
        send_ok(h)
        for c in content_chunks('{"ok": true}'):
            h.wfile.write(c)
        h.wfile.write(b"data: [DONE]\n\n")

    server = StubServer([], script=script)
    try:
        backend = LocalJSON(base_url=server.url, idle_timeout=0.3, first_byte_timeout=3, retries=0)
        data, _ = backend.complete_json("s", [], {}, "t")
    finally:
        server.close()
    assert data == {"ok": True}
    assert len(server.requests) == 1


def test_idle_timeout_applies_after_first_byte():
    def script(h, n):
        send_ok(h)
        h.wfile.write(next(content_chunks('{"ok": true}', finish=None)))
        h.wfile.flush()
        time.sleep(2)  # 出了一个字就卡住

    server = StubServer([], script=script)
    try:
        backend = LocalJSON(base_url=server.url, idle_timeout=0.3, first_byte_timeout=5, retries=0)
        started = time.monotonic()
        with pytest.raises(LLMError, match="像是卡住了"):
            backend.complete_json("s", [], {}, "t")
        assert time.monotonic() - started < 1.5
    finally:
        server.close()


def test_chunked_transfer_like_ollama():
    """真实的 Ollama 用 HTTP/1.1 分块传输。"""
    def script(h, n):
        h.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
        for c in [*content_chunks('{"ok": "分块"}'), b"data: [DONE]\n\n"]:
            h.wfile.write(f"{len(c):x}\r\n".encode() + c + b"\r\n")
        h.wfile.write(b"0\r\n\r\n")

    server = StubServer([], script=script)
    try:
        data, _ = LocalJSON(base_url=server.url, idle_timeout=2).complete_json("s", [], {}, "t")
    finally:
        server.close()
    assert data == {"ok": "分块"}


def test_negative_retries_are_clamped():
    assert LocalJSON(retries=-1).retries == 0
    assert LocalJSON(idle_timeout=10).first_byte_timeout == 30


@pytest.mark.parametrize("caps,expected", [(["completion", "vision"], True), (["completion"], False), (None, None)])
def test_supports_vision_asks_ollama(caps, expected):
    server = StubServer([], capabilities=caps)
    try:
        backend = LocalJSON(model="qwen2.5vl:7b", base_url=server.url)
        assert backend.supports_vision() is expected
        assert backend.supports_vision() is expected  # 第二次用缓存
    finally:
        server.close()
    assert [r["path"] for r in server.requests] == ["/api/show"]
    assert server.requests[0]["body"] == {"model": "qwen2.5vl:7b"}


def test_supports_vision_unknown_when_server_is_down():
    assert LocalJSON(base_url="http://127.0.0.1:9/v1").supports_vision() is None


def test_photo_with_text_only_local_model_fails_before_sending(tmp_path):
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG fake")
    server = StubServer([], capabilities=["completion"])
    try:
        with pytest.raises(RuntimeError, match="不支持看图"):
            LLMRecognizer(LocalJSON(base_url=server.url)).recognize("", str(img))
    finally:
        server.close()
    assert [r["path"] for r in server.requests] == ["/api/show"]


# ---------- Claude 后端 ----------


class MessagesStub:
    """假的 Anthropic Messages API：用真实 SDK 发请求，检查我们传的参数 SDK 都认。"""

    def __init__(self, text=None, status=200):
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "body": body, "beta": self.headers.get("anthropic-beta")})
                if status == 200:
                    payload = {
                        "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
                        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                        "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10},
                    }
                else:
                    payload = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
                data = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_real_sdk_accepts_our_request(world):
    anthropic = pytest.importorskip("anthropic")
    stub = MessagesStub(plan_text(GOOD_PLAN))
    try:
        client = anthropic.Anthropic(api_key="test-key", base_url=stub.url, max_retries=0)
        result = LLMPlanner(ClaudeJSON(client=client)).plan("把牛肉拿出来解冻", world)
    finally:
        stub.close()
    assert len(result) == len(GOOD_PLAN)
    req = stub.requests[0]
    assert req["path"].startswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in req["beta"]
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"


def test_sdk_errors_become_llm_errors():
    anthropic = pytest.importorskip("anthropic")
    stub = MessagesStub(status=401)
    try:
        client = anthropic.Anthropic(api_key="bad-key", base_url=stub.url, max_retries=0)
        with pytest.raises(LLMError, match="调用 Claude 失败"):
            ClaudeJSON(client=client).complete_json("s", [{"role": "user", "content": "x"}], {"type": "object"}, "t")
    finally:
        stub.close()


def test_missing_api_key_message(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    client = anthropic.Anthropic(base_url="http://127.0.0.1:9", max_retries=0)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        ClaudeJSON(client=client).complete_json("s", [{"role": "user", "content": "x"}], {"type": "object"}, "t")


def test_missing_sdk_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMError, match=r'pip install -e "\.\[claude\]"'):
        ClaudeJSON()


def test_cli_reports_missing_sdk(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert main(["run", "把牛肉拿出来解冻", "--planner", "claude"]) == 2
    assert "pip install" in capsys.readouterr().err
    assert main(["intake", "买了牛奶", "--recognizer", "claude"]) == 2
    assert "pip install" in capsys.readouterr().err


# ---------- 命令行：入库预览与防误写 ----------


def test_cli_refuses_to_overwrite_sample(capsys):
    before = DEFAULT_SPEC.read_bytes()
    assert main(["intake", "买了牛奶", "--save", str(DEFAULT_SPEC), "--yes"]) == 2
    assert "样例文件" in capsys.readouterr().err
    assert DEFAULT_SPEC.read_bytes() == before


def test_cli_force_allows_sample_path(tmp_path, monkeypatch, capsys):
    copy = tmp_path / "kitchen.json"
    copy.write_bytes(DEFAULT_SPEC.read_bytes())
    monkeypatch.setattr("robohome.cli.DEFAULT_SPEC", copy)
    assert main(["intake", "买了牛奶", "--spec", str(copy), "--save", str(copy), "--yes", "--force"]) == 0
    assert sum(i.name == "牛奶" for i in load_home(copy).items.values()) == 2


def test_cli_missing_photo(capsys):
    assert main(["intake", "--recognizer", "local", "--image", "/no/such/photo.jpg"]) == 2
    assert "找不到照片" in capsys.readouterr().err


def test_saved_ledger_is_compact_and_loads_back(tmp_path):
    out = tmp_path / "home.json"
    assert main(["intake", "买了两盒酸奶", "--save", str(out), "--yes"]) == 0
    text = out.read_text(encoding="utf-8")
    assert '"position": [' in text and "\n            0.0" not in text  # 坐标写在一行里
    home = load_home(out)
    assert any(i.name == "酸奶" for i in home.items.values())


def test_sample_file_is_unchanged_by_test_suite():
    # 防止哪个测试误写了仓库自带的样例
    assert len(load_home(DEFAULT_SPEC).items) == 5

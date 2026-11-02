"""本地模型后端：起一个假的 OpenAI 兼容服务，走真实的 HTTP 请求。"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from robohome.cli import main
from robohome.intake import LLMRecognizer
from robohome.llm import LLMError, LocalJSON
from robohome.planner import LLMPlanner, PlanningError
from robohome.scheduler import Scheduler
from robohome.spec import load_home
from robohome.world import World

from test_step2 import BAD_PLAN, GOOD_PLAN

SPEC = Path(__file__).resolve().parents[1] / "data" / "kitchen_example.json"


class StubServer:
    """按顺序返回预设的 content 文本，记录收到的请求体。"""

    def __init__(self, contents, status=200, finish_reason="stop", stall_first=0.0, capabilities=None, script=None):
        self.contents = list(contents)
        self.capabilities = capabilities  # /api/show 返回的模型能力；None 表示不是 Ollama（404）
        self.script = script  # 自定义流式输出：给定时按它写原始字节，用来模拟断线等异常
        self.requests = []
        self.stall_first = stall_first  # 第一次请求先卡住这么多秒（模拟 Ollama 挂住）
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                if self.path == "/api/show":
                    caps = stub.capabilities
                    data = json.dumps({"capabilities": caps} if caps is not None else {"error": "not found"}).encode()
                    self.send_response(200 if caps is not None else 404)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if stub.script is not None and body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    stub.script(self, len(stub.requests))
                    return
                if stub.stall_first and len(stub.requests) == 1:
                    time.sleep(stub.stall_first)
                    return
                if status == 200 and body.get("stream"):
                    content = stub.contents.pop(0)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    pieces = [content[i:i + 40] for i in range(0, len(content), 40)] or [""]
                    for n, piece in enumerate(pieces):
                        last = n == len(pieces) - 1
                        chunk = {"choices": [{"delta": {"content": piece},
                                              "finish_reason": finish_reason if last else None}]}
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                if status != 200:
                    payload = {"error": "model not found"}
                else:
                    content = stub.contents.pop(0)
                    payload = {"choices": [{"message": {"role": "assistant", "content": content},
                                            "finish_reason": finish_reason}]}
                data = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def plan_text(steps, feasible=True, reason="ok", think=False):
    text = json.dumps({"feasible": feasible, "reason": reason, "steps": steps}, ensure_ascii=False)
    return f"<think>先开冷冻室……</think>\n{text}" if think else text


@pytest.fixture
def world():
    return World.from_home(load_home(SPEC), start_waypoint="entrance")


def test_local_planner_over_http(world):
    server = StubServer([plan_text(GOOD_PLAN)])
    try:
        backend = LocalJSON(model="qwen3:8b", base_url=server.url)
        steps = LLMPlanner(backend).plan("把牛肉拿出来解冻", world)
    finally:
        server.close()
    assert len(steps) == len(GOOD_PLAN)
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "qwen3:8b"
    assert req["body"]["stream"] is True
    assert req["body"]["messages"][0]["role"] == "system"
    assert req["body"]["response_format"]["type"] == "json_schema"
    assert req["body"]["response_format"]["json_schema"]["schema"]["required"] == ["feasible", "reason", "steps"]


def test_local_planner_replans_with_text_history(world):
    server = StubServer([plan_text(BAD_PLAN), plan_text(GOOD_PLAN, think=True)])
    try:
        steps = LLMPlanner(LocalJSON(base_url=server.url), normalize_plans=False).plan("把牛肉拿出来解冻", world)
    finally:
        server.close()
    assert len(steps) == len(GOOD_PLAN)
    msgs = server.requests[1]["body"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert isinstance(msgs[2]["content"], str) and "fridge_front" in msgs[2]["content"]
    assert "冷冻室门没开" in msgs[3]["content"]


def test_local_end_to_end_through_scheduler(world):
    server = StubServer([plan_text(GOOD_PLAN)])
    try:
        s = Scheduler(world, LLMPlanner(LocalJSON(base_url=server.url)))
        s.submit("把牛肉拿出来解冻")
        [report] = s.run_pending()
    finally:
        server.close()
    assert report.ok, report.render()


def test_local_errors_become_planning_errors(world):
    server = StubServer([], status=404)
    try:
        with pytest.raises(PlanningError, match="404"):
            LLMPlanner(LocalJSON(base_url=server.url)).plan("x", world)
    finally:
        server.close()

    server = StubServer(["这不是 JSON"])
    try:
        with pytest.raises(PlanningError, match="合法 JSON"):
            LLMPlanner(LocalJSON(base_url=server.url)).plan("x", world)
    finally:
        server.close()

    server = StubServer(['{"feasible": tr'], finish_reason="length")
    try:
        with pytest.raises(PlanningError, match="截断"):
            LLMPlanner(LocalJSON(base_url=server.url)).plan("x", world)
    finally:
        server.close()


def test_connection_refused_message():
    backend = LocalJSON(base_url="http://127.0.0.1:9/v1", idle_timeout=2)
    with pytest.raises(LLMError, match="ollama pull"):
        backend.complete_json("s", [{"role": "user", "content": "x"}], {"type": "object"}, "t")


def test_local_recognizer_and_api_key(tmp_path):
    payload = {"items": [{"name": "虾仁", "category": "水产", "temperature": "frozen", "volume_l": 0.5,
                          "aliases": ["虾"]}], "unrecognized": []}
    server = StubServer([json.dumps(payload, ensure_ascii=False)])
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG fake")
    try:
        backend = LocalJSON(base_url=server.url, api_key="secret")
        items, _ = LLMRecognizer(backend).recognize("买了虾仁", image_path=str(img))
    finally:
        server.close()
    assert items[0].name == "虾仁"
    assert server.requests[0]["path"] == "/api/show"  # 先问能不能看图；不是 Ollama 就照常尝试
    req = server.requests[1]
    assert req["auth"] == "Bearer secret"
    content = req["body"]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_cli_local_planner(capsys):
    server = StubServer([plan_text(GOOD_PLAN)])
    try:
        code = main(["run", "把牛肉拿出来解冻", "--planner", "local", "--base-url", server.url])
    finally:
        server.close()
    assert code == 0
    assert "完成" in capsys.readouterr().out


def test_stalled_stream_is_retried(world):
    """在 Mac 上遇到过 Ollama 连接挂住不出字；现在等 idle_timeout 秒就重试，不再干等 10 分钟。"""
    server = StubServer([plan_text(GOOD_PLAN)], stall_first=3)
    try:
        backend = LocalJSON(base_url=server.url, idle_timeout=0.5, retries=1)
        started = time.monotonic()
        steps = LLMPlanner(backend).plan("把牛肉拿出来解冻", world)
        elapsed = time.monotonic() - started
    finally:
        server.close()
    assert len(steps) == len(GOOD_PLAN)
    assert len(server.requests) == 2
    assert elapsed < 3


def test_stall_without_retry_fails_fast(world):
    server = StubServer([], stall_first=3)
    try:
        backend = LocalJSON(base_url=server.url, idle_timeout=0.5, retries=0)
        with pytest.raises(PlanningError, match="卡住|没有响应"):
            LLMPlanner(backend).plan("x", world)
    finally:
        server.close()

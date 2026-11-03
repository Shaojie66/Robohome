"""家庭服务器：手机下指令、看进度和回报、回答机器人的问题；服务重启后台账和没做完的任务都还在。"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from robohome.cli import DEFAULT_SPEC
from robohome.intake.recognize import KeywordRecognizer
from robohome.mvp import apply_fault
from robohome.planner.base import Step
from robohome.planner.rules import RulePlanner
from robohome.server import Service, make_server, on_home_network
from robohome.server.service import step_text
from robohome.spec.dump import home_to_dict
from robohome.spec.loader import SpecError, load_home, parse_home


def make(tmp_path, **kw) -> Service:
    return Service(tmp_path / "home", DEFAULT_SPEC, RulePlanner(), KeywordRecognizer(), **kw)


def run_all(service: Service, limit: int = 20) -> None:
    for _ in range(limit):
        if not service.run_once():
            return


def task(service: Service, task_id: str) -> dict:
    return next(t for t in service.state()["tasks"] if t["id"] == task_id)


def question(service: Service) -> dict:
    [q] = service.state()["questions"]
    return q


# ---------- 下指令、看进度和回报 ----------


def test_task_runs_and_reports(tmp_path):
    s = make(tmp_path)
    t = s.submit("把冷冻牛肉转到冷藏层解冻")
    assert t["status"] == "queued"
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "done"
    assert t["report"]["ok"]
    assert [p["state"] for p in t["plan"]] == ["ok"] * len(t["plan"])
    assert t["plan"][0]["text"] == "走到冷冻室门前"  # 给人看的说法，不是 navigate(...)
    assert "牛肉" in [i for c in s.state()["storage"][1]["containers"] for i in c["items"]]


def test_progress_is_reported_step_by_step(tmp_path):
    s = make(tmp_path)
    states = []
    original = s._progress

    def spy(t, i, result):
        original(t, i, result)
        states.append([p["state"] for p in t["plan"]])

    s._progress = spy
    s.submit("看看冰箱里还有什么")
    run_all(s)
    assert states[0][0] == "doing" and states[1][0] == "ok"
    assert states[-1] == ["ok"] * len(states[-1])


def test_failure_report_reads_like_a_person(tmp_path):
    s = make(tmp_path)
    apply_fault(s.world, "stuck:freezer_door")
    t = s.submit("把冷冻牛肉转到冷藏层解冻")
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "stopped"
    assert t["report"]["summary"].startswith("第 2 步（打开冷冻室门）执行失败")
    assert t["report"]["scene"] is not None
    assert s.state()["questions"] == []  # 门卡住不是主人能在手机上拿主意的事，照常停下回报


def test_empty_and_too_long_instructions_are_refused(tmp_path):
    s = make(tmp_path)
    with pytest.raises(ValueError):
        s.submit("   ")
    with pytest.raises(ValueError):
        s.submit("把" * 301)


def test_step_text_covers_every_skill(tmp_path):
    s = make(tmp_path)
    home = s.world.home
    steps = {
        "navigate": Step("navigate", {"waypoint": "freezer_front"}),
        "open": Step("open", {"fixture": "freezer_door"}),
        "close": Step("close", {"fixture": "freezer_door"}),
        "pick": Step("pick", {"container": "box_f1"}),
        "place": Step("place", {"container": "box_f1", "location": "chiller", "slot": 0}),
        "inspect": Step("inspect", {"location": "freezer"}),
        "pour": Step("pour", {"container": "jar_f1", "into": "air_fryer"}),
        "appliance": Step("appliance", {"appliance": "air_fryer", "action": "air_fry",
                                        "params": {"minutes": 12, "temp_c": 180}}),
        "stow": Step("stow", {"parcel": "p", "container": "box_f1", "name": "五花肉"}),
        "notify": Step("notify", {"message": "包装 2 号认不出"}),
    }
    texts = {k: step_text(v, home, s._place) for k, v in steps.items()}
    assert texts["appliance"] == "让空气炸锅炸（12 分钟，180 度）"
    assert texts["place"] == "把冷冻盒 1 号放进冷藏室"
    assert all("(" not in t for t in texts.values()), texts


# ---------- 入库：先给主人看，主人点了才装 ----------


def test_intake_asks_before_stowing(tmp_path):
    s = make(tmp_path)
    t = s.submit("今天买了一斤五花肉、两盒酸奶和一袋苹果")
    run_all(s)
    assert task(s, t["id"])["status"] == "waiting"
    q = question(s)
    assert q["kind"] == "intake" and len(q["lines"]) == 3 and all(line["ok"] for line in q["lines"])
    assert q["fields"] == []
    assert len([c for c in s.world.home.containers.values() if c.kind == "parcel"]) == 3  # 还没动手
    s.answer(q["id"], "go")
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "done", t["events"]
    assert not [c for c in s.world.home.containers.values() if c.kind == "parcel"]
    names = {i.name for i in s.world.home.items.values()}
    assert {"猪肉", "酸奶", "苹果"} <= names


def test_owner_names_what_robot_could_not_recognize(tmp_path):
    s = make(tmp_path)
    t = s.submit("买了一袋苹果和一包榴莲干")
    run_all(s)
    q = question(s)
    assert [f["id"] for f in q["fields"]] == ["parcel_2"]
    assert q["lines"][1]["ok"] is False
    s.answer(q["id"], "go", {"parcel_2": "葡萄"})
    run_all(s)
    q2 = question(s)
    assert q2["id"] != q["id"]
    assert "你说是「葡萄」" in q2["lines"][1]["text"] and q2["lines"][1]["ok"]
    s.answer(q2["id"], "go")
    run_all(s)
    assert task(s, t["id"])["status"] == "done"
    grapes = s.world.home.find_item("葡萄")
    assert grapes is not None and s.world.home.storage_of(grapes.container).temperature == "chilled"


def test_unrecognized_left_on_counter_and_told(tmp_path):
    s = make(tmp_path)
    t = s.submit("买了一袋苹果和一包榴莲干")
    run_all(s)
    s.answer(question(s)["id"], "go")  # 不填，认不出的先留在台上
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "done"
    assert any("包装 2 号" in n for n in t["report"]["notices"])
    assert [c.id for c in s.world.home.containers.values() if c.kind == "parcel"] == ["parcel_2"]


def test_nothing_recognizable_only_offers_naming(tmp_path):
    s = make(tmp_path)
    s.submit("买了一包榴莲干")
    run_all(s)
    q = question(s)
    go = next(o for o in q["options"] if o["id"] == "go")
    assert go["label"] == "再认一遍" and go["needs_fields"]


def test_cancel_intake_leaves_things_on_counter(tmp_path):
    s = make(tmp_path)
    t = s.submit("今天买了一袋饺子")
    run_all(s)
    s.cancel(t["id"])
    assert task(s, t["id"])["status"] == "cancelled"
    assert s.state()["questions"] == []
    assert [c.label for c in s.world.home.containers.values() if c.kind == "parcel"] == ["一袋饺子"]
    with pytest.raises(ValueError):
        s.cancel(t["id"])


def test_confirmed_intake_uses_what_the_owner_saw(tmp_path):
    """主人确认的是哪样就装哪样：确认之后不再让识别器认一遍（大模型每次认的可能不一样）。"""
    s = make(tmp_path)
    s.submit("今天买了两盒酸奶")
    run_all(s)
    q = question(s)

    class Broken:
        def recognize_labels(self, *a, **k):
            raise AssertionError("确认后不该再认一遍")

    s.recognizer = Broken()
    s.answer(q["id"], "go")
    run_all(s)
    assert s.state()["tasks"][0]["status"] == "done"


# ---------- 听不懂的指令、主人开着的家电 ----------


def test_rephrase_when_planner_cannot(tmp_path):
    s = make(tmp_path)
    t = s.submit("帮我跳个舞")
    run_all(s)
    q = question(s)
    assert q["kind"] == "rephrase" and "不认识这句话" in q["text"]
    s.answer(q["id"], "go", {"text": "看看冰箱里还有什么"})
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "done" and t["instruction"] == "看看冰箱里还有什么"
    assert any("你改成：看看冰箱里还有什么" in e["text"] for e in t["events"])


def test_asks_before_stopping_owners_appliance(tmp_path):
    s = make(tmp_path)
    s.set_appliance("dishwasher", True)  # 主人自己开的洗碗机
    t = s.submit("把洗好的碗从洗碗机收到柜子")
    run_all(s)
    q = question(s)
    assert q["kind"] == "stop" and q["title"] == "洗碗机正开着，要停掉吗？"
    assert s.world.robot.waypoint == "entrance"  # 预演就发现了，还没走过去
    assert s.world.appliance_running("dishwasher")
    s.answer(q["id"], "stop")
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "done", t["events"]
    assert not s.world.appliance_running("dishwasher")
    assert any("按你说的停掉了洗碗机" in e["text"] for e in t["events"])


def test_owner_says_leave_the_appliance_running(tmp_path):
    s = make(tmp_path)
    s.set_appliance("air_fryer", True)
    t = s.submit("把鸡米花放进空气炸锅并启动")
    run_all(s)
    s.answer(question(s)["id"], "cancel")
    run_all(s)
    assert task(s, t["id"])["status"] == "cancelled"
    assert s.world.appliance_running("air_fryer")  # 主人说不用停，就不停


def test_queue_waits_while_owner_decides(tmp_path):
    s = make(tmp_path)
    first = s.submit("今天买了一袋饺子")
    second = s.submit("看看冰箱里还有什么")
    run_all(s)
    assert task(s, first["id"])["status"] == "waiting"
    assert task(s, second["id"])["status"] == "queued"  # 主人还没拿主意，后面的先等着
    s.answer(question(s)["id"], "go")
    run_all(s)
    assert task(s, first["id"])["status"] == "done"
    assert task(s, second["id"])["status"] == "done"


def test_answer_checks(tmp_path):
    s = make(tmp_path)
    s.submit("今天买了一袋饺子")
    run_all(s)
    q = question(s)
    with pytest.raises(ValueError):
        s.answer(q["id"], "nope")
    s.answer(q["id"], "go", {"not_a_field": "x"})
    with pytest.raises(ValueError):
        s.answer(q["id"], "go")  # 已经回答过了
    with pytest.raises(KeyError):
        s.answer("q999", "go")


def test_owner_appliance_switch_is_refused_while_robot_works(tmp_path):
    s = make(tmp_path)
    s.world_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="机器人正在做事"):
            s.set_appliance("dishwasher", True)
    finally:
        s.world_lock.release()
    with pytest.raises(KeyError):
        s.set_appliance("toaster", True)


# ---------- 重启 ----------


def test_restart_keeps_ledger_and_unfinished_tasks(tmp_path):
    s = make(tmp_path)
    s.submit("今天买了一斤五花肉")
    run_all(s)
    s.answer(question(s)["id"], "go")
    run_all(s)
    queued = s.submit("把冷冻牛肉转到冷藏层解冻")
    waiting = s.submit("今天买了两盒酸奶")
    s.set_appliance("dishwasher", True)

    s2 = make(tmp_path)  # 服务重启
    assert s2.key == s.key
    assert s2.world.home.find_item("五花肉") is not None
    assert s2.world.appliance_running("dishwasher")
    assert task(s2, queued["id"])["status"] == "queued"
    run_all(s2)
    assert task(s2, queued["id"])["status"] == "done"
    assert task(s2, waiting["id"])["status"] == "waiting"

    s3 = make(tmp_path)  # 等回答时又重启了一次
    q = question(s3)
    s3.answer(q["id"], "go")
    run_all(s3)
    assert task(s3, waiting["id"])["status"] == "done"
    assert s3.world.home.find_item("酸奶") is not None
    beef = s3.world.home.find_item("牛肉")
    assert s3.world.home.storage_of(beef.container).temperature == "chilled"


def test_restart_mid_task_wraps_up(tmp_path):
    """做到一半服务停了（断电、Ctrl+C）：重启后这个任务算中断，关上机器人开着的门，列出要人处理的事。"""
    s = make(tmp_path)
    original = s._progress

    def crash(t, i, result):
        original(t, i, result)
        if result is not None and i == 1:  # 冷冻室门刚打开
            raise KeyboardInterrupt

    s._progress = crash
    t = s.submit("把冷冻牛肉转到冷藏层解冻")
    with pytest.raises(KeyboardInterrupt):
        run_all(s)
    assert s.store.read("state.json")["opened_by_robot"] == ["freezer_door"]

    s2 = make(tmp_path)
    t = task(s2, t["id"])
    assert t["status"] == "interrupted"
    assert "做到第 3 步" in t["report"]["summary"]
    assert [c["ok"] for c in t["report"]["cleanup"]] == [True]
    assert not s2.world.opened_by_robot
    s2.retry(t["id"])
    run_all(s2)
    assert s2.state()["tasks"][0]["status"] == "done"


def test_restart_while_holding_something(tmp_path):
    """机器人手里拿着东西时的台账也能存、能读回来（规范文件里不允许，运行状态里允许）。"""
    s = make(tmp_path)
    original = s._progress

    def crash(t, i, result):
        original(t, i, result)
        if result is not None and i == 2:  # 刚拿起冷冻盒
            raise KeyboardInterrupt

    s._progress = crash
    s.submit("把冷冻牛肉转到冷藏层解冻")
    with pytest.raises(KeyboardInterrupt):
        run_all(s)
    data = s.store.read("home.json")
    with pytest.raises(SpecError):
        parse_home(data)  # 当规范文件用不行
    s2 = make(tmp_path)
    assert s2.world.robot.holding == "box_f1"
    assert any("拿着冷冻盒 1 号" in n for n in s2.state()["tasks"][0]["report"]["needs_attention"])


def test_service_ledger_matches_spec_format(tmp_path):
    s = make(tmp_path)
    load_home(s.store.root / "home.json")  # 没在做事时，台账就是一份合法的规范文件
    assert home_to_dict(s.world.home)["name"] == "样例厨房"


# ---------- 网页和接口 ----------


MAC_IFCONFIG = """lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
\tinet6 ::1 prefixlen 128
\tinet6 fe80::1%lo0 prefixlen 64 scopeid 0x1
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet6 fe80::1c2b:3a4d:5e6f:7081%en0 prefixlen 64 secured scopeid 0xb
\tinet 100.66.10.20 netmask 0xfffffc00 broadcast 100.66.11.255
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 169.254.12.3 netmask 0xffff0000 broadcast 169.254.255.255
utun10: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 100.79.1.2 --> 100.79.1.2 netmask 0xffffffff
\tinet6 fd7a:115c:a1e0::1234 prefixlen 48
"""

LINUX_IP = """1: lo    inet 127.0.0.1/8 scope host lo       valid_lft forever preferred_lft forever
2: wlan0    inet 192.168.31.20/24 brd 192.168.31.255 scope global dynamic wlan0
2: wlan0    inet6 fd00:1:2:3::5/64 scope global dynamic
2: wlan0    inet6 fe80::5/64 scope link
5: tailscale0    inet 100.101.2.3/32 scope global tailscale0
6: wg0    inet 10.8.0.2/24 scope global wg0
"""


def test_home_network_is_this_computers_subnet():
    """家里的 Wi-Fi 不一定是 192.168.x.x，有的路由器分 100.64–127.x.x。按这台电脑网卡所在的网段算。"""
    from robohome.server.http import home_networks, parse_interfaces

    nets = home_networks(parse_interfaces(MAC_IFCONFIG))
    assert [str(n) for n in nets] == ["100.66.8.0/22"]  # 本机、链路本地、VPN 网卡（这里是 Tailscale）不算
    for ok in ("100.66.8.9", "100.66.11.254", "127.0.0.1", "::1", "fe80::1", "169.254.1.1", "::ffff:100.66.9.1"):
        assert on_home_network(ok, nets), ok
    for bad in ("100.66.12.1", "100.79.1.3", "fd7a:115c:a1e0::9", "192.168.1.23", "8.8.8.8", "not-an-ip"):
        assert not on_home_network(bad, nets), bad

    nets = home_networks(parse_interfaces(LINUX_IP))
    assert [str(n) for n in nets] == ["192.168.31.0/24", "fd00:1:2:3::/64"]
    assert on_home_network("192.168.31.77", nets) and on_home_network("fd00:1:2:3::9", nets)
    assert not on_home_network("192.168.32.1", nets) and not on_home_network("10.8.0.5", nets)


def test_home_network_fallback_without_interface_list():
    for ok in ("192.168.1.23", "10.0.0.8", "172.20.1.2", "127.0.0.1", "::1", "fe80::1", "::ffff:192.168.1.5", "fd12::3"):
        assert on_home_network(ok), ok
    for bad in ("8.8.8.8", "100.64.1.1", "172.32.0.1", "2001:db8::1", "::ffff:1.2.3.4", "not-an-ip"):
        assert not on_home_network(bad), bad


@pytest.fixture
def server(tmp_path):
    s = make(tmp_path)
    httpd = make_server(s, "127.0.0.1", 0, networks=())
    stop = threading.Event()
    threads = [threading.Thread(target=httpd.serve_forever, daemon=True),
               threading.Thread(target=s.run_forever, args=(stop,), daemon=True)]
    for th in threads:
        th.start()
    yield s, f"http://127.0.0.1:{httpd.server_address[1]}"
    stop.set()
    httpd.shutdown()
    httpd.server_close()


def call(url, body=None, key=None, raw=False):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if key:
        req.add_header("X-Robohome-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = r.read()
            return r.status, (payload if raw else json.loads(payload))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_for(fn, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("等太久了")


def test_http_end_to_end(server):
    s, base = server
    status, html = call(base + "/", raw=True)
    assert status == 200 and "机器人管家".encode() in html
    assert call(base + "/api/state")[0] == 401
    assert call(base + "/api/state", key="wrong")[0] == 401
    status, state = call(base + "/api/state", key=s.key)
    assert status == 200 and state["home"] == "样例厨房" and state["robot"]["place"] == "门口"
    assert call(f"{base}/api/state?key={s.key}")[0] == 200  # 照片用 ?key=，接口也认

    status, t = call(base + "/api/tasks", {"instruction": "今天买了一斤五花肉和一包榴莲干"}, key=s.key)
    assert status == 201
    q = wait_for(lambda: call(base + "/api/state", key=s.key)[1]["questions"])[0]
    status, _ = call(f"{base}/api/questions/{q['id']}", {"choice": "go", "fields": {"parcel_2": "牛肉"}}, key=s.key)
    assert status == 200
    q2 = wait_for(lambda: [x for x in call(base + "/api/state", key=s.key)[1]["questions"] if x["id"] != q["id"]])[0]
    call(f"{base}/api/questions/{q2['id']}", {"choice": "go"}, key=s.key)
    done = wait_for(lambda: [x for x in call(base + "/api/state", key=s.key)[1]["tasks"]
                             if x["id"] == t["id"] and x["status"] == "done"])[0]
    assert done["report"]["ok"]
    assert "data" not in q and "names" not in done and "put_parcels" not in done  # 内部字段不给出去

    status, again = call(f"{base}/api/tasks/{t['id']}/retry", {}, key=s.key)
    assert status == 200 and again["instruction"] == t["instruction"]


def test_http_rejects_bad_requests(server):
    s, base = server
    assert call(base + "/api/tasks", b"not json", key=s.key)[0] == 400
    assert call(base + "/api/tasks", [1, 2], key=s.key)[0] == 400
    assert call(base + "/api/tasks", {"instruction": ""}, key=s.key)[0] == 400
    assert call(base + "/api/tasks/t999/cancel", {}, key=s.key)[0] == 404
    assert call(base + "/api/questions/q1", {"choice": "go", "fields": "x"}, key=s.key)[0] == 400
    assert call(base + "/api/nothing", {}, key=s.key)[0] == 404
    assert call(f"{base}/photos/..%2Fkey?key={s.key}")[0] == 404
    assert call(f"{base}/photos/../key?key={s.key}")[0] == 404
    (s.store.root / "secret.png").write_bytes(b"x")
    assert call(f"{base}/photos/..%2Fsecret.png?key={s.key}")[0] == 404
    (s.store.photos / "ok.png").write_bytes(b"png")
    assert call(f"{base}/photos/ok.png?key={s.key}", raw=True) == (200, b"png")
    assert call(f"{base}/photos/ok.png")[0] == 401


def test_http_appliance_switch(server):
    s, base = server
    status, out = call(base + "/api/appliances/dishwasher", {"running": True}, key=s.key)
    assert status == 200 and out["running"]
    appl = {a["id"]: a for a in call(base + "/api/state", key=s.key)[1]["appliances"]}
    assert appl["dishwasher"]["running"] and not appl["dishwasher"]["by_robot"]
    assert call(base + "/api/appliances/toaster", {"running": True}, key=s.key)[0] == 404


def test_http_too_many_wrong_keys(server):
    s, base = server
    for _ in range(20):
        assert call(base + "/api/state", key="guess")[0] == 401
    assert call(base + "/api/state", key="guess")[0] == 429
    assert call(base + "/api/state", key=s.key)[0] == 429  # 这个地址先歇一会儿


# ---------- 仿真：回报带照片 ----------


def test_sim_report_has_photos(tmp_path):
    from robohome import sim  # 先于 mujoco 导入，没有显示器的 Linux 才会用软件渲染

    if not sim.available():
        pytest.skip(sim.INSTALL_HINT)
    s = make(tmp_path, sim=True)
    t = s.submit("看看冰箱里还有什么")
    run_all(s)
    t = task(s, t["id"])
    assert t["status"] == "done"
    photos = t["report"]["photos"]
    if not any(p["image"] for p in photos):
        pytest.skip("这台机器上渲染不可用")
    for p in photos:
        assert p["image"].startswith("/photos/") and (s.store.photos / p["image"][len("/photos/"):]).is_file()

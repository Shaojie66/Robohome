"""家庭服务器的核心：收指令、排队、一件一件交给机器人做，要主人拿主意的事停下来问。

一次只做一件事，前一件做完（或停下）才做下一件；有问题等主人回答时，后面的任务也先等着，
免得机器人在主人还没拿主意时把家里的状态改了。

要主人拿主意的三种事：
- 入库：认出来的东西打算放哪，先给主人看一眼，主人点"开始装"才动手；
  标签看不清、认不出的，主人可以直接告诉机器人这是什么
- 听不懂或做不了的指令：说明原因，请主人换个说法
- 要停掉主人自己开着的家电（比如洗碗机还在洗）：机器人不会自己停，问主人要不要停

所有改动世界的事都在 worker 里做（主线程），网页那边只往任务表里加东西、填答案。
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

from ..appliances.base import MockAppliance
from ..intake.core import TEMP_NAMES, NewItem, parcels_on_counter, put_parcels
from ..intake.recognize import split_pieces
from ..planner.base import Planner, Step
from ..planner.intake import Decision, IntakePlanner, LabelRecognizer, is_intake
from ..scheduler.core import Report, Scheduler, StepResult, simulate
from ..spec.loader import load_home
from ..spec.models import FLOOR
from ..world import World
from .store import Store

log = logging.getLogger(__name__)

STATUS_NAMES = {
    "queued": "排队中", "planning": "在想怎么做", "waiting": "等你回答", "running": "正在做",
    "done": "做完了", "stopped": "停下了", "cancelled": "取消了", "interrupted": "中断了",
}
ACTIVE = ("planning", "running")
FINISHED = ("done", "stopped", "cancelled", "interrupted")
MAX_TASKS_SHOWN = 30
MAX_INSTRUCTION = 300


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


ACTION_NAMES = {"defrost": "解冻", "heat": "加热", "air_fry": "炸", "wash": "洗", "on": "打开", "off": "关掉",
                "stop": "停下"}


def step_text(step: Step, home, place) -> str:
    """一步计划给人看的说法，比如"走到冷冻室门前""打开冷冻室门"。"""
    a = step.args

    def name(obj_id) -> str:
        for group in (home.containers, home.storage, home.appliances, home.fixtures):
            if obj_id in group:
                return group[obj_id].name
        return str(obj_id)

    try:
        match step.skill:
            case "navigate":
                return f"走到{place(a['waypoint'])}"
            case "open" | "close":
                return f"{'打开' if step.skill == 'open' else '关上'}{name(a['fixture'])}"
            case "pick":
                return f"拿起{name(a['container'])}"
            case "place":
                return f"把{name(a['container'])}放进{name(a['location'])}"
            case "inspect":
                return f"拍一张{name(a['location'])}"
            case "pour":
                return f"把{name(a['container'])}里的东西倒进{name(a['into'])}"
            case "appliance":
                params = a.get("params") or {}
                extra = "，".join(x for x in (f"{params['minutes']} 分钟" if params.get("minutes") else "",
                                              f"{params['temp_c']} 度" if params.get("temp_c") else "") if x)
                act = ACTION_NAMES.get(a["action"], a["action"])
                return f"让{name(a['appliance'])}{act}" + (f"（{extra}）" if extra else "")
            case "stow":
                return f"把{a.get('name') or name(a['parcel'])}装进{name(a['container'])}"
            case "notify":
                return f"告诉你：{a['message']}"
    except (KeyError, TypeError):
        pass
    return step.describe()


class FixedRecognizer:
    """主人确认过的识别结果：照着确认时的样子装，不再让模型认一遍（模型每次认的可能不一样）。"""

    def __init__(self, items: list[NewItem | None]):
        self.items = items

    def recognize_labels(self, labels, instruction="", categories=None) -> list[NewItem | None]:
        return list(self.items)


class Service:
    def __init__(
        self,
        data_dir: str | Path,
        spec: str | Path,
        planner: Planner,
        recognizer: LabelRecognizer,
        sim: bool = False,
        video: bool = False,
        examples: list[str] | None = None,
        planner_name: str = "rule",
    ):
        self.store = Store(data_dir)
        self.planner = planner
        self.recognizer = recognizer
        self.sim = sim
        self.video = video
        self.planner_name = planner_name
        self.examples = examples or []
        self.lock = threading.RLock()  # 任务表和问题
        self.world_lock = threading.Lock()  # 机器人在做事时拿着；网页模拟主人开关家电要先拿到
        self.changed = threading.Condition(self.lock)
        self.version = 0
        self.key = self.store.key()

        if self.store.has_world():
            self.world = self.store.load_world()
        else:
            home = load_home(spec)
            first = next(iter(home.rooms.values()))
            self.world = World.from_home(home, start_waypoint="entrance" if "entrance" in first.waypoints else None)
            self.store.save_world(self.world)

        saved = self.store.read("tasks.json", {}) or {}
        self.tasks: list[dict] = saved.get("tasks", [])
        self.questions: list[dict] = saved.get("questions", [])
        self.next_id = saved.get("next_id", 1)
        with self.lock:
            self.view = self._world_view()
            self._recover()
            self.view = self._world_view()
            self._save_tasks()

    # ---------- 网页调用（任何线程） ----------

    def submit(self, instruction: str, put: bool = True) -> dict:
        """put：入库指令要不要先把说到的东西摆上入库台（仿真里代替主人摆）；重做时不再摆一次。"""
        instruction = " ".join((instruction or "").split())
        if not instruction:
            raise ValueError("指令是空的")
        if len(instruction) > MAX_INSTRUCTION:
            raise ValueError(f"指令太长了（最多 {MAX_INSTRUCTION} 个字）")
        with self.lock:
            task = {
                "id": self._new_id("t"), "instruction": instruction, "kind": "intake" if is_intake(instruction) else "task",
                "status": "queued", "created": _now(), "updated": _now(), "put_parcels": put,
                "events": [], "plan": [], "report": None, "question": None, "names": {},
            }
            self._event(task, f"收到：{instruction}")
            self.tasks.append(task)
            self._changed()
            return self._public_task(task)

    def retry(self, task_id: str) -> dict:
        with self.lock:
            old = self._task(task_id)
            if old["status"] not in FINISHED:
                raise ValueError("这个任务还没结束")
            return self.submit(old["instruction"], put=False)

    def cancel(self, task_id: str) -> dict:
        with self.lock:
            task = self._task(task_id)
            if task["status"] not in ("queued", "waiting"):
                raise ValueError(f"任务{STATUS_NAMES.get(task['status'], task['status'])}，不能取消")
            q = self._question(task["question"]) if task["question"] else None
            if q and q["status"] == "pending":
                q["status"] = "cancelled"
            left = " 东西还在入库台上。" if task["kind"] == "intake" and task["status"] == "waiting" else ""
            self._finish(task, "cancelled", f"你取消了。{left}".strip())
            self._changed()
            return self._public_task(task)

    def answer(self, question_id: str, choice: str, fields: dict | None = None) -> dict:
        with self.lock:
            q = self._question(question_id)
            if q["status"] != "pending":
                raise ValueError("这个问题已经回答过了")
            if choice not in [o["id"] for o in q["options"]]:
                raise ValueError(f"没有这个选项：{choice}")
            allowed = {f["id"] for f in q["fields"]}
            clean = {k: " ".join(str(v).split())[:60] for k, v in (fields or {}).items() if k in allowed and str(v).strip()}
            q["status"], q["answer"], q["answered"] = "answered", {"choice": choice, "fields": clean}, _now()
            self._changed()
            return self._public_question(q)

    def set_appliance(self, appliance_id: str, running: bool) -> dict:
        """模拟主人自己按了家电的开关（真家电接进来以后，这一步由家电自己上报）。"""
        if not self.world_lock.acquire(timeout=0.5):
            raise BusyError("机器人正在做事，等它做完再试")
        try:
            adapter = self.world.appliances.get(appliance_id)
            a = self.world.home.appliances.get(appliance_id)
            if a is None or not isinstance(adapter, MockAppliance):
                raise KeyError(appliance_id)
            if running:
                if not adapter.online:
                    raise ValueError(f"{a.name}离线")
                if not adapter.door_closed:
                    raise ValueError(f"{a.name}的门开着，不能启动")
                adapter.running, adapter.mode = True, (adapter.capabilities or ["run"])[0]
            else:
                adapter.running, adapter.mode = False, None
            self.world.started_by_robot.discard(appliance_id)  # 主人按的，就不算机器人启动的
            self.store.save_world(self.world)
            with self.lock:
                self.view = self._world_view()
                self._changed()
            return {"id": appliance_id, "running": adapter.running}
        finally:
            self.world_lock.release()

    def state(self) -> dict:
        with self.lock:
            tasks = [self._public_task(t) for t in reversed(self.tasks[-MAX_TASKS_SHOWN:])]
            pending = [self._public_question(q) for q in self.questions if q["status"] == "pending"]
            busy = any(t["status"] in ACTIVE for t in self.tasks)
            view = {**self.view, "attention": [] if busy else self.view["attention"]}
            # 复制一份再交出去：网页那边转 JSON 的时候，worker 可能正在改任务
            return copy.deepcopy({
                "version": self.version, **view, "tasks": tasks, "questions": pending,
                "examples": self.examples, "sim": self.sim, "planner": self.planner_name,
                "status_names": STATUS_NAMES,
            })

    # ---------- worker ----------

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if not self.run_once():
                with self.changed:
                    self.changed.wait(timeout=1.0)

    def run_once(self) -> bool:
        """做一件事：处理一个回答，或者开始一个排队的任务。没事做返回 False。"""
        with self.lock:
            job = self._next_job()
        if job is None:
            return False
        kind, task, q = job
        with self.world_lock:
            try:
                if kind == "answer":
                    self._handle_answer(task, q)
                else:
                    self._start(task)
            except Exception as e:
                log.exception("任务 %s 出错", task["id"])
                with self.lock:
                    self._finish(task, "stopped", f"出了没想到的错：{e}")
            finally:
                self.world.body = None
                self.store.save_world(self.world)
                with self.lock:
                    self.view = self._world_view()
                    self._changed()
        return True

    def _next_job(self):
        for q in self.questions:
            if q["status"] == "answered":
                q["status"] = "handled"
                return "answer", self._task(q["task"]), q
        if any(t["status"] in ("waiting", *ACTIVE) for t in self.tasks):
            return None  # 等主人回答时，后面的任务先等着
        for t in self.tasks:
            if t["status"] == "queued":
                t["status"] = "planning"
                self._touch(t)
                return "start", t, None
        return None

    # ---------- 做任务 ----------

    def _start(self, task: dict) -> None:
        self._attach(task)
        if task["kind"] == "intake":
            self._intake_preview(task)
        else:
            self._plan_and_run(task, task["instruction"])

    def _plan_and_run(self, task: dict, instruction: str) -> None:
        with self.lock:
            task["status"] = "planning"
            self._touch(task)
            self._changed()
        try:
            plan = self.planner.plan(instruction, self.world)
        except Exception as e:  # noqa: BLE001 规划器（尤其是大模型）的各种错误都当成"没法照这句做"
            reason = str(e) or type(e).__name__
            with self.lock:
                self._event(task, f"没法照这句做：{reason}")
                if not self._ask_stop(task, reason):
                    self._ask_rephrase(task, reason)
            return
        # 先在副本上预演一遍：要停主人开着的家电才做得成的，动手之前就问，不白跑一趟
        preview = simulate(self.world, plan, instruction=instruction)
        if not preview.ok and self._owner_appliance(preview.summary):
            with self.lock:
                self._event(task, f"预演发现：{self._humane(preview.summary, plan)}")
                self._ask_stop(task, preview.summary)
            return
        self._execute(task, instruction, plan)

    def _execute(self, task: dict, instruction: str, plan: list[Step]) -> Report:
        with self.lock:
            task["status"] = "running"
            task["plan"] = [{"text": step_text(s, self.world.home, self._place), "state": "todo", "message": ""}
                            for s in plan]
            task["report"] = None
            self._touch(task)
            self._changed()
        scheduler = Scheduler(self.world, self.planner, on_progress=lambda i, s, r: self._progress(task, i, r))
        report = scheduler.execute(instruction, plan)
        with self.lock:
            task["report"] = self._report_view(report)
            summary = task["report"]["summary"]
            if report.ok:
                self._finish(task, "done", f"做完了：{summary}")
            elif not self._ask_stop(task, report.summary):
                self._finish(task, "stopped", f"停下了：{summary}")
        return report

    def _progress(self, task: dict, i: int, result: StepResult | None) -> None:
        with self.lock:
            if not 0 <= i < len(task["plan"]):
                return
            row = task["plan"][i]
            if result is None:
                row["state"] = "doing"
            else:
                row["state"], row["message"] = ("ok" if result.ok else "fail"), result.message
            self._touch(task)
            if result is not None:
                self.view = self._world_view()
            self._changed()
        if result is not None:
            self.store.save_world(self.world)  # 每做完一步存一次，断电重启后知道做到哪了

    # ---------- 入库 ----------

    def _intake_preview(self, task: dict) -> None:
        world = self.world
        if task.get("put_parcels"):
            pieces = split_pieces(task["instruction"])
            if pieces:
                try:
                    ids = put_parcels(world, pieces)  # 仿真里代替主人把买回来的东西摆上入库台
                except ValueError as e:
                    with self.lock:
                        self._finish(task, "stopped", f"停下了：{e}")
                    return
                with self.lock:
                    self._event(task, f"摆上入库台：{'、'.join(pieces)}（{len(ids)} 件）")
            with self.lock:
                task["put_parcels"] = False  # 已经摆过了，重启后接着问也不再摆一次
                self._save_tasks()
            self.store.save_world(world)
        with self.lock:
            task["status"] = "planning"
            self._touch(task)
            self._changed()
        planner = IntakePlanner(self.recognizer, names=task.get("names"))
        parcels = parcels_on_counter(world)
        error = ""
        try:
            planner.plan(task["instruction"], world)
        except Exception as e:  # noqa: BLE001 识别出错、什么都放不了：都要告诉主人
            error = str(e)
        with self.lock:
            if not planner.decisions:
                self._finish(task, "stopped", f"停下了：{error or '入库台上没有东西'}")
                return
            self._ask_intake(task, planner.decisions, parcels, error)

    def _ask_intake(self, task: dict, decisions: list[Decision], parcels, error: str) -> None:
        home = self.world.home
        lines, fields = [], []
        for d in decisions:
            name = home.containers[d.parcel].name
            said = f"你说是「{d.label}」" if d.named_by_owner else (f"「{d.label}」" if d.label else "标签看不清")
            if d.item is None:
                lines.append({"ok": False, "text": f"{name}：{said}，认不出是什么"})
                fields.append({"id": d.parcel, "label": f"{name}是什么？", "placeholder": "比如：牛肉、酸奶"})
            elif d.container is None:
                lines.append({"ok": False, "text": f"{name}：{said} → {d.item.name}，{d.note.strip()}，{d.ask}"})
            else:
                c = home.containers[d.container]
                area = home.storage.get(c.location)
                where = f"{area.name if area else c.location}的{c.name}"
                lines.append({"ok": True, "text": (f"{name}：{said} → {d.item.name}（{TEMP_NAMES[d.item.temperature]}，"
                                                   f"{d.item.volume_l:g} 升）→ {where}{d.note}")})
        placeable = any(d.container for d in decisions)
        # 填了认不出的东西是什么，点主按钮就是"再认一遍"；什么都放不了时，主按钮只能用来再认一遍
        go = {"id": "go", "label": "开始装" if placeable else "再认一遍", "primary": True, "needs_fields": not placeable}
        options = ([go] if placeable or fields else []) + [{"id": "cancel", "label": "先不装"}]
        if fields:
            text = "认不出的东西，告诉我它是什么，我再认一遍；不填就先留在入库台上。"
        else:
            text = "按下面这样装，可以吗？"
        if not placeable:
            text = (error + "。" if error else "") + "填上它们是什么，我再认一遍。"
        items = {p.id: None for p in parcels}
        for d in decisions:
            items[d.parcel] = asdict(d.item) if d.item else None
        self._ask(task, {
            "kind": "intake", "title": f"入库台上有 {len(decisions)} 件东西", "text": text,
            "lines": lines, "fields": fields, "options": options, "retry_label": "再认一遍",
            "data": {"items": items},
        })

    def _intake_go(self, task: dict, q: dict) -> None:
        saved = q["data"]["items"]
        parcels = parcels_on_counter(self.world)
        if [p.id for p in parcels] != list(saved):
            with self.lock:
                self._event(task, "入库台上的东西变了，重新看一遍")
            self._intake_preview(task)
            return
        items = [NewItem(**saved[p.id]) if saved[p.id] else None for p in parcels]
        planner = IntakePlanner(FixedRecognizer(items), names=task.get("names"))
        try:
            plan = planner.plan(task["instruction"], self.world)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self._finish(task, "stopped", f"停下了：{e}")
            return
        self._execute(task, task["instruction"], plan)

    # ---------- 问主人 ----------

    def _ask(self, task: dict, spec: dict) -> dict:
        q = {"id": self._new_id("q"), "task": task["id"], "status": "pending", "created": _now(), "answer": None,
             "fields": [], "lines": [], **spec}
        self.questions.append(q)
        task["status"], task["question"] = "waiting", q["id"]
        self._event(task, f"问你：{q['title']}")
        self._touch(task)
        self._changed()
        return q

    def _ask_rephrase(self, task: dict, reason: str) -> None:
        self._ask(task, {
            "kind": "rephrase", "title": "这句我没法照着做", "text": reason,
            "fields": [{"id": "text", "label": "换个说法", "placeholder": task["instruction"]}],
            "options": [{"id": "go", "label": "按新说法做", "primary": True}, {"id": "cancel", "label": "算了"}],
        })

    def _owner_appliance(self, text: str):
        """失败原因里提到的、主人自己开着的家电（不是机器人启动的）。"""
        for a in self.world.home.appliances.values():
            if a.name not in text or a.id in self.world.started_by_robot:
                continue
            if not self.world.appliance_running(a.id):
                continue
            stop = next((s for s in ("stop", "off") if s in a.capabilities), None)
            if stop:
                return a, stop
        return None

    def _ask_stop(self, task: dict, reason: str) -> bool:
        found = self._owner_appliance(reason)
        if found is None:
            return False
        a, action = found
        self._ask(task, {
            "kind": "stop", "title": f"{a.name}正开着，要停掉吗？",
            "text": f"{a.name}正在运行，不是我开的，我不会自己停。停掉它才能接着做「{task['instruction']}」。",
            "options": [{"id": "stop", "label": f"停掉{a.name}，接着做", "primary": True},
                        {"id": "cancel", "label": "不用，先不做"}],
            "data": {"appliance": a.id, "action": action},
        })
        return True

    def _handle_answer(self, task: dict, q: dict) -> None:
        choice, fields = q["answer"]["choice"], q["answer"]["fields"]
        with self.lock:
            task["question"] = None
            if task["status"] != "waiting":
                return  # 已经取消了
            task["status"] = "planning"
            self._touch(task)
            self._changed()
        if choice == "cancel":
            with self.lock:
                left = "东西还在入库台上。" if q["kind"] == "intake" else ""
                self._finish(task, "cancelled", f"你说先不做。{left}".strip())
            return
        self._attach(task)
        if q["kind"] == "intake":
            if fields:
                with self.lock:
                    task["names"] = {**task.get("names", {}), **fields}
                    told = "、".join(f"{self.world.home.containers[k].name}是{v}" for k, v in fields.items()
                                    if k in self.world.home.containers)
                    self._event(task, f"你说：{told}")
                self._intake_preview(task)
            else:
                with self.lock:
                    self._event(task, "你说：开始装")
                self._intake_go(task, q)
        elif q["kind"] == "rephrase":
            text = " ".join(fields.get("text", "").split())[:MAX_INSTRUCTION]
            if not text:
                with self.lock:
                    self._ask_rephrase(task, "你没有写新的说法")
                return
            with self.lock:
                self._event(task, f"你改成：{text}")
                task["instruction"], task["kind"] = text, "intake" if is_intake(text) else "task"
                task["put_parcels"] = True
            self._start(task)
        elif q["kind"] == "stop":
            aid, action = q["data"]["appliance"], q["data"]["action"]
            name = self.world.home.appliances[aid].name
            # 主人点了"停掉"，就是主人明确让停：用主人的话当指令，安全规则照常检查
            report = Scheduler(self.world, self.planner).execute(
                f"停掉{name}", [Step("appliance", {"appliance": aid, "action": action})])
            with self.lock:
                if not report.ok:
                    task["report"] = self._report_view(report)
                    self._finish(task, "stopped", f"没停下来：{report.summary}")
                    return
                self._event(task, f"按你说的停掉了{name}，接着做")
            self._plan_and_run(task, task["instruction"])

    # ---------- 重启 ----------

    def _recover(self) -> None:
        """服务重启：正在做的任务算中断（做安全收尾），排队的接着排，等回答的接着等。"""
        interrupted = [t for t in self.tasks if t["status"] in ACTIVE]
        for q in self.questions:
            # 回答刚开始处理就重启了：任务还在等，回答重新处理一遍
            if q["status"] == "handled" and self._task(q["task"])["status"] == "waiting":
                q["status"] = "answered"
        for t in interrupted:
            was_running = t["status"] == "running"
            done = sum(1 for row in t.get("plan", []) if row["state"] == "ok")
            where = f"做到第 {done + 1} 步" if was_running and t.get("plan") else "还在想怎么做"
            for row in t.get("plan", []):
                if row["state"] == "doing":
                    row["state"] = "fail"
                    row["message"] = "做到一半服务重启了"
            summary = f"服务重启时这件事{where}，停下了"
            if was_running:
                self._attach(t)
                report = Scheduler(self.world, self.planner).abort(t["instruction"], summary)
                self.world.body = None
                t["report"] = self._report_view(report)
            self._finish(t, "interrupted", f"{summary}，可以点「重做」")
        if interrupted:
            self.store.save_world(self.world)

    # ---------- 仿真 ----------

    def _attach(self, task: dict) -> None:
        """仿真模式下每个任务接一个新的仿真（按当前台账搭场景），照片存到数据目录。"""
        if not self.sim:
            return
        from ..sim.body import SimBody
        from ..sim.render import Recorder

        rec = Recorder(self.store.photos, prefix=f"{task['id']}-{int(time.time())}-", video=self.video)
        task["_recorder"] = rec
        SimBody.attach(self.world, recorder=rec)

    # ---------- 小工具 ----------

    def _finish(self, task: dict, status: str, text: str) -> None:
        task["status"] = status
        self._event(task, text)
        rec = task.pop("_recorder", None)
        if rec is not None and self.video and status in ("done", "stopped") and rec.frames:
            path = rec.save_gif(self.store.photos / f"{task['id']}-{int(time.time())}.gif")
            if path and task.get("report") is not None:
                task["report"]["video"] = self._photo_url(path)
        if rec is not None:
            rec.close()
        self._touch(task)
        self._changed()

    def _event(self, task: dict, text: str) -> None:
        task["events"].append({"time": _now(), "text": text})

    def _touch(self, task: dict) -> None:
        task["updated"] = _now()

    def _changed(self) -> None:
        self.version += 1
        self._save_tasks()
        self.changed.notify_all()

    def _save_tasks(self) -> None:
        tasks = [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.tasks]
        self.store.write("tasks.json", {"next_id": self.next_id, "tasks": tasks, "questions": self.questions})

    def _new_id(self, prefix: str) -> str:
        n = self.next_id
        self.next_id += 1
        return f"{prefix}{n}"

    def _task(self, task_id: str) -> dict:
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        raise KeyError(task_id)

    def _question(self, question_id: str) -> dict:
        for q in self.questions:
            if q["id"] == question_id:
                return q
        raise KeyError(question_id)

    def _public_task(self, task: dict) -> dict:
        out = {k: v for k, v in task.items() if not k.startswith("_") and k not in ("names", "put_parcels")}
        out["status_name"] = STATUS_NAMES.get(task["status"], task["status"])
        return out

    def _public_question(self, q: dict) -> dict:
        task = self._task(q["task"])
        return {k: v for k, v in q.items() if k != "data"} | {"instruction": task["instruction"]}

    def _photo_url(self, path: str | None) -> str | None:
        if not path:
            return None
        p = Path(path)
        return f"/photos/{p.name}" if p.parent.resolve() == self.store.photos.resolve() else None

    def _report_view(self, report: Report) -> dict:
        d = report.to_dict()
        for photo in [*d["photos"], *([d["scene"]] if d["scene"] else [])]:
            photo["image"] = self._photo_url(photo["image"])
        for row, r in zip([*d["steps"], *d["cleanup"]], [*report.results, *report.cleanup]):
            row["text"] = step_text(r.step, self.world.home, self._place)
        d["summary"] = self._humane(d["summary"], [r.step for r in report.results])
        d.pop("final_state", None)
        return d

    def _humane(self, text: str, steps: list[Step]) -> str:
        """回报里的"第 2 步（open(fixture=...)）"换成"第 2 步（打开洗碗机抽屉）"。"""
        for step in steps:
            text = text.replace(f"（{step.describe()}）", f"（{step_text(step, self.world.home, self._place)}）")
        return text

    def _place(self, waypoint: str) -> str:
        """点位的叫法：站在哪个东西前面。"""
        home = self.world.home
        if waypoint == "entrance":
            return "门口"
        for group in (home.storage.values(), home.appliances.values(), home.fixtures.values()):
            for obj in group:
                if getattr(obj, "waypoint", None) == waypoint:
                    return f"{obj.name}前"
        return waypoint

    def _world_view(self) -> dict:
        """手机上显示的家：机器人在哪、家电开着没、每个存放区里有什么、要人处理的事。"""
        world, home = self.world, self.world.home
        holding = home.containers[world.robot.holding].name if world.robot.holding in home.containers else None
        appliances = []
        for a in home.appliances.values():
            status = World._status_or_error(world.appliances[a.id])
            inside = [i.name for i in home.items.values() if i.container == a.id]
            inside += [c.name for c in home.containers.values() if c.location == a.id]
            appliances.append({
                "id": a.id, "name": a.name, "running": bool(status.get("running")), "mode": status.get("mode"),
                "online": status.get("online", "error" not in status), "by_robot": a.id in world.started_by_robot,
                "inside": inside,
            })
        storage = []
        for s in home.storage.values():
            boxes = []
            for c in sorted((c for c in home.containers.values() if c.location == s.id), key=lambda c: c.slot):
                items = [i.name for i in home.items.values() if i.container == c.id]
                boxes.append({"name": c.name, "items": items, "label": c.label if c.kind == "parcel" else ""})
            storage.append({"id": s.id, "name": s.name, "temperature": TEMP_NAMES.get(s.temperature, ""),
                            "kind": s.kind, "containers": boxes})
        # 机器人做事的时候门开着、手里拿着东西是正常的，state() 只在它停下来时才把这些给主人看
        attention = [f"{home.fixtures[f].name}还开着（机器人开的）" for f in sorted(world.opened_by_robot)]
        attention += [f"{c.name}掉在地上了" for c in home.containers.values() if c.location == FLOOR]
        if holding:
            attention.append(f"机器人手里拿着{holding}，停在{self._place(world.robot.waypoint)}")
        return {
            "home": home.name,
            "robot": {"place": self._place(world.robot.waypoint), "waypoint": world.robot.waypoint, "holding": holding},
            "appliances": appliances, "storage": storage, "attention": attention,
            "open_doors": [f.name for f in home.fixtures.values() if f.state == "open"],
        }


class BusyError(RuntimeError):
    """机器人正在做事，现在不能改家里的状态。"""

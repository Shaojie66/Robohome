"""命令行入口。

    robohome validate data/kitchen_example.json
    robohome run "把牛肉拿出来解冻"
    robohome run "把牛肉拿出来解冻" --fault offline:microwave
    robohome run "把牛肉拿出来解冻" --fault missing:box_f1
    robohome run "把桌上散落的常用物品收回原位" --fault drop:cup_1
    robohome mvp                                         # MVP 五个任务各跑一次
    robohome mvp --faults                                # 每个任务注入故障，检查是否安全停下
    robohome mvp --planner local --repeat 20             # 用本地模型连跑 20 次，看成功率
    robohome run "把牛肉拿出来解冻" --planner local       # 用本地模型规划（默认 Ollama + qwen3:8b）
    robohome run "把牛肉拿出来解冻" --planner claude      # 用 Claude 规划（需要 ANTHROPIC_API_KEY）
    robohome intake "今天买了一斤五花肉、两盒酸奶和一袋苹果"           # 只预览，不改台账
    robohome intake "..." --spec my_home.json --save my_home.json --yes  # 确认后写入台账
    robohome intake --recognizer local --model qwen2.5vl:7b --image receipt.jpg
    robohome skills
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .intake.core import assign
from .intake.recognize import KeywordRecognizer, LLMRecognizer
from .llm.base import LLMError
from .mvp import FAULT_HELP, TASKS, apply_fault, run_mvp, summarize
from .planner.base import Planner
from .planner.rules import RulePlanner
from .scheduler.core import Scheduler
from .skills.library import catalog
from .spec.dump import save_home
from .spec.loader import SpecError, load_home
from .world import World

DEFAULT_SPEC = Path(__file__).resolve().parents[2] / "data" / "kitchen_example.json"


def _make_planner(name: str, model: str | None, base_url: str | None) -> Planner:
    if name == "rule":
        return RulePlanner()
    from .llm import make_backend
    from .planner.llm import LLMPlanner

    return LLMPlanner(make_backend(name, model, base_url, effort="medium"))


def _start_world(home) -> World:
    first_room = next(iter(home.rooms.values()))
    return World.from_home(home, start_waypoint="entrance" if "entrance" in first_room.waypoints else None)


def _step_json(s) -> dict:
    return {"step": s.step.describe(), "ok": s.ok, "stage": s.stage, "message": s.message}


def _photo_json(p) -> dict:
    return {"location": p.location, "title": p.title, "seen": p.seen, "notes": p.notes}


def _report_json(r) -> dict:
    return {
        "instruction": r.instruction, "ok": r.ok, "summary": r.summary,
        "steps": [_step_json(s) for s in r.results],
        "photos": [_photo_json(p) for p in r.photos],
        "scene": _photo_json(r.scene) if r.scene else None,
        "notices": r.notices,
        "cleanup": [_step_json(s) for s in r.cleanup],
        "needs_attention": r.needs_attention,
        "final_state": r.final_state,
    }


def _mvp(args) -> int:
    try:
        planner = _make_planner(args.planner, args.model, args.base_url)
    except LLMError as e:
        print(e, file=sys.stderr)
        return 2

    def show(run) -> None:
        tag = f"［故障 {run.fault}，应{'停下' if run.expect == 'stop' else '提醒'}］" if run.fault else ""
        mark = "通过" if run.passed else "未通过：" + "；".join(run.problems)
        lines = [f"\n{tag}{run.report.render()}"]
        if args.show_plans or not run.passed:
            for n, a in enumerate(run.attempts, 1):
                lines.append(f"  规划第 {n} 轮：模型给了 {len(a['model'])} 步，补全后 {len(a['plan'])} 步，"
                             + ("预演通过" if a["error"] is None else f"预演失败：{a['error']}"))
                if args.show_plans:
                    lines += [f"    模型：{s}" for s in a["model"]]
        lines.append(f"→ {mark}（{run.seconds:.1f} 秒）")
        print("\n".join(lines), flush=True)  # 本地模型一次要几十秒，及时打印，免得看起来像卡住

    runs = run_mvp(planner, args.spec, args.task or None, faults=args.faults, repeat=args.repeat, on_run=show)
    print(summarize(runs))
    return 0 if all(r.passed for r in runs) else 1


def _intake(args, home, world: World) -> int:
    if args.save and Path(args.save).resolve() == DEFAULT_SPEC.resolve() and not args.force:
        print(f"不会改写仓库自带的样例文件 {DEFAULT_SPEC.name}；先复制一份再用，或者加 --force", file=sys.stderr)
        return 2
    if args.image and not Path(args.image).is_file():
        print(f"找不到照片：{args.image}", file=sys.stderr)
        return 2
    if args.recognizer in ("local", "claude"):
        from .llm import make_backend

        categories = sorted({it.category for it in home.items.values()})
        try:
            backend = make_backend(args.recognizer, args.model, args.base_url, effort="low")
            items, unknown = LLMRecognizer(backend).recognize(args.text, args.image, categories)
        except (LLMError, RuntimeError) as e:
            print(e, file=sys.stderr)
            return 2
    else:
        if args.image:
            print("识别照片需要 --recognizer local 或 claude", file=sys.stderr)
            return 2
        items, unknown = KeywordRecognizer().recognize(args.text)
    result = assign(world, items)
    result.unrecognized = unknown
    print(result.render(world))
    if args.save:
        if args.yes:
            save_home(world.home, args.save)
            print(f"台账已写入 {args.save}")
        else:
            print(f"以上只是预览，台账没有改动。确认无误后加 --yes 写入 {args.save}")
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robohome", description="家庭专属机器人管家（模拟运行）")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="校验家用规范文件")
    v.add_argument("spec", nargs="?", default=str(DEFAULT_SPEC))

    r = sub.add_parser("run", help="执行一条或多条指令")
    r.add_argument("instructions", nargs="+")
    r.add_argument("--spec", default=str(DEFAULT_SPEC))
    r.add_argument("--fault", action="append", default=[], help=f"注入故障：{FAULT_HELP}")
    r.add_argument("--json", action="store_true", help="输出 JSON 结果")
    r.add_argument("--planner", choices=["rule", "local", "claude"], default="rule",
                   help="规划器：rule 规则（离线）/ local 本地模型 / claude")
    r.add_argument("--model", help="模型名；local 默认 qwen3:8b，claude 默认 claude-opus-5-5")
    r.add_argument("--base-url", help="本地模型服务地址，默认 http://localhost:11434/v1（Ollama）")

    i = sub.add_parser("intake", help="入库：识别新买的物品并分配容器")
    i.add_argument("text", nargs="?", default="")
    i.add_argument("--spec", default=str(DEFAULT_SPEC))
    i.add_argument("--recognizer", choices=["keyword", "local", "claude"], default="keyword")
    i.add_argument("--image", help="小票或购物袋照片（需要 local 视觉模型或 claude）")
    i.add_argument("--model", help="模型名；local 默认 qwen3:8b，claude 默认 claude-opus-5-5")
    i.add_argument("--base-url", help="本地模型服务地址，默认 http://localhost:11434/v1（Ollama）")
    i.add_argument("--save", help="把更新后的台账写到这个文件（要加 --yes 才真正写入）")
    i.add_argument("--yes", action="store_true", help="确认写入台账；不加时只预览")
    i.add_argument("--force", action="store_true", help="允许覆盖仓库自带的样例文件")

    m = sub.add_parser("mvp", help="端到端跑 MVP 的五个任务（可注入故障、可重复多次）")
    m.add_argument("--spec", default=str(DEFAULT_SPEC))
    m.add_argument("--task", action="append", choices=[t.id for t in TASKS], help="只跑某个任务，可写多次")
    m.add_argument("--faults", action="store_true", help="每个任务注入几种故障，检查是否安全停下")
    m.add_argument("--repeat", type=int, default=1, help="每个任务重复几次（不注入故障时）")
    m.add_argument("--planner", choices=["rule", "local", "claude"], default="rule")
    m.add_argument("--show-plans", action="store_true", help="打印大模型每一轮给的计划")
    m.add_argument("--model")
    m.add_argument("--base-url")

    sub.add_parser("skills", help="列出技能清单")

    args = p.parse_args(argv)

    if args.cmd == "skills":
        print(json.dumps(catalog(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "mvp":
        return _mvp(args)

    try:
        home = load_home(args.spec)
    except SpecError as e:
        print(e, file=sys.stderr)
        return 2

    if args.cmd == "validate":
        print(f"✓ {home.name}：{len(home.rooms)} 个房间，{len(home.fixtures)} 个部件，"
              f"{len(home.containers)} 个容器，{len(home.items)} 件物品，{len(home.appliances)} 台家电")
        return 0

    world = _start_world(home)

    if args.cmd == "intake":
        return _intake(args, home, world)

    for f in args.fault:
        try:
            apply_fault(world, f)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
    try:
        planner = _make_planner(args.planner, args.model, args.base_url)
    except LLMError as e:
        print(e, file=sys.stderr)
        return 2
    scheduler = Scheduler(world, planner)
    for ins in args.instructions:
        scheduler.submit(ins)
    reports = scheduler.run_pending()

    if args.json:
        print(json.dumps([_report_json(r) for r in reports], ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(r.render() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())

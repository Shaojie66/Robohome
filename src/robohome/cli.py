"""命令行入口。

    robohome validate data/kitchen_example.json
    robohome run "把牛肉拿出来解冻"
    robohome run "把牛肉拿出来解冻" --fault offline:microwave
    robohome run "把牛肉拿出来解冻" --fault missing:box_f1
    robohome run "把牛肉拿出来解冻" --planner local       # 用本地模型规划（默认 Ollama + qwen3:8b）
    robohome run "把牛肉拿出来解冻" --planner claude      # 用 Claude 规划（需要 ANTHROPIC_API_KEY）
    robohome intake "今天买了一斤五花肉、两盒酸奶和一袋苹果"
    robohome intake "..." --recognizer claude --image receipt.jpg --save data/my_home.json
    robohome skills
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .appliances.base import MockAppliance
from .intake.core import assign
from .intake.recognize import KeywordRecognizer, LLMRecognizer
from .planner.base import Planner
from .planner.rules import RulePlanner
from .scheduler.core import Scheduler
from .skills.library import catalog
from .spec.dump import save_home
from .spec.loader import SpecError, load_home
from .world import World

DEFAULT_SPEC = Path(__file__).resolve().parents[2] / "data" / "kitchen_example.json"


def _apply_fault(world: World, fault: str) -> None:
    kind, _, target = fault.partition(":")
    if kind == "offline":
        adapter = world.appliances.get(target)
        if not isinstance(adapter, MockAppliance):
            raise SystemExit(f"没有可以设为离线的家电：{target}")
        adapter.online = False
    elif kind == "missing":
        if target not in world.home.containers:
            raise SystemExit(f"没有这个容器：{target}")
        world.missing_containers.add(target)
    else:
        raise SystemExit(f"不认识的故障类型：{fault}（可用 offline:<家电> / missing:<容器>）")


def _make_planner(name: str, model: str | None, base_url: str | None) -> Planner:
    if name == "rule":
        return RulePlanner()
    from .llm import make_backend
    from .planner.llm import LLMPlanner

    return LLMPlanner(make_backend(name, model, base_url, effort="medium"))


def _start_world(home) -> World:
    first_room = next(iter(home.rooms.values()))
    return World.from_home(home, start_waypoint="entrance" if "entrance" in first_room.waypoints else None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robohome", description="家庭专属机器人管家（模拟运行）")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="校验家用规范文件")
    v.add_argument("spec", nargs="?", default=str(DEFAULT_SPEC))

    r = sub.add_parser("run", help="执行一条或多条指令")
    r.add_argument("instructions", nargs="+")
    r.add_argument("--spec", default=str(DEFAULT_SPEC))
    r.add_argument("--fault", action="append", default=[], help="注入故障：offline:<家电> 或 missing:<容器>")
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
    i.add_argument("--save", help="把更新后的台账写到这个文件")

    sub.add_parser("skills", help="列出技能清单")

    args = p.parse_args(argv)

    if args.cmd == "skills":
        print(json.dumps(catalog(), ensure_ascii=False, indent=2))
        return 0

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
        if args.recognizer in ("local", "claude"):
            from .llm import make_backend

            categories = sorted({it.category for it in home.items.values()})
            backend = make_backend(args.recognizer, args.model, args.base_url, effort="low")
            try:
                items, unknown = LLMRecognizer(backend).recognize(args.text, args.image, categories)
            except RuntimeError as e:
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
            save_home(world.home, args.save)
            print(f"台账已写入 {args.save}")
        return 0 if result.ok else 1

    for f in args.fault:
        _apply_fault(world, f)
    scheduler = Scheduler(world, _make_planner(args.planner, args.model, args.base_url))
    for ins in args.instructions:
        scheduler.submit(ins)
    reports = scheduler.run_pending()

    if args.json:
        print(json.dumps([
            {"instruction": r.instruction, "ok": r.ok, "summary": r.summary,
             "steps": [{"step": s.step.describe(), "ok": s.ok, "stage": s.stage, "message": s.message} for s in r.results],
             "final_state": r.final_state}
            for r in reports
        ], ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(r.render() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())

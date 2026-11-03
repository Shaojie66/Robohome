"""入库规划器：主人把买回来的东西摆在入库台上，机器人逐件装盒、放进对应温区。

    今天买了一斤五花肉、两盒酸奶和一袋苹果
    → 拍入库台，读每件包装的标签
    → 识别每件是什么、要放哪个温区、大概多大（关键词表，或大模型）
    → 按 intake.core 的规则挑容器：温区对、同类优先、不混装、不硬塞
    → 每件：拿起包装 → 走到容器跟前、开门 → 装进去（stow，装好才记账）→ 关门
    → 认不出、放不下的留在入库台上，告诉主人

大模型只负责"认"：标签上是什么东西、该冷冻还是冷藏。挑容器、走动、开关门都按固定规矩，
和其他任务一样交给调度器逐步检查后执行。
机器人不拆包装（不碰刀具）：一件东西一个容器装不下，就请主人拆开分装。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..intake.core import (
    TEMP_NAMES,
    NewItem,
    choose_container,
    largest_container,
    parcels_on_counter,
    record_item,
    settle_category,
)
from ..world import World
from .base import PlanningError, Step
from .builder import Builder

BUY_WORDS = ("买了", "买回", "买的", "新买", "超市回来", "采购")
# 提到买，但要做的是别的事，比如"把昨天买的牛肉拿出来解冻"
OTHER_TASK_WORDS = (
    "拿出", "取出", "解冻", "空气炸锅", "炸一下", "加热", "热一下", "启动", "洗碗机", "原位", "归位", "看看", "查看",
)


def is_intake(instruction: str) -> bool:
    """这句话是不是让机器人把买回来的东西收起来。"""
    if "入库" in instruction:
        return True
    return any(w in instruction for w in BUY_WORDS) and not any(w in instruction for w in OTHER_TASK_WORDS)


class LabelRecognizer(Protocol):
    def recognize_labels(
        self, labels: list[str | None], instruction: str = "", categories: list[str] | None = None
    ) -> list[NewItem | None]: ...


@dataclass
class Decision:
    """一件包装怎么处理：装进哪个容器，或者为什么留在入库台上。"""

    parcel: str
    label: str
    item: NewItem | None = None
    container: str | None = None
    note: str = ""  # 留在台上的原因，或者挑这个容器的理由
    ask: str = "请主人看一下"  # 留在台上时，请主人做什么

    def describe(self) -> str:
        if not self.label:
            return f"{self.parcel} → 标签看不清"
        if self.item is None:
            return f"{self.parcel}「{self.label}」→ 认不出"
        i = self.item
        what = f"{i.name}（{i.category}，{TEMP_NAMES[i.temperature]}，{i.volume_l:g} 升）"
        return f"{self.parcel}「{self.label}」→ {what} → {self.container or '留在台上：' + self.note}"


class IntakePlanner:
    def __init__(self, recognizer: LabelRecognizer):
        self.recognizer = recognizer
        self.decisions: list[Decision] = []  # 最近一次规划每件包装的处理，方便调试和回报

    def plan(self, instruction: str, world: World) -> list[Step]:
        home = world.home
        parcels = parcels_on_counter(world)
        if not parcels:
            raise PlanningError("入库台上没有要入库的东西")
        labels = [p.label.strip() or None for p in parcels]
        categories = sorted({i.category for i in home.items.values()})
        try:
            items = self.recognizer.recognize_labels(labels, instruction, categories)
        except RuntimeError as e:
            raise PlanningError(str(e)) from e
        if len(items) != len(parcels):
            raise PlanningError(f"识别结果数量不对：{len(parcels)} 件包装，识别出 {len(items)} 条")

        # 在副本上逐件分配，后面的东西能看到前面的占掉了多少地方；真正记账等装好那一刻
        sandbox = world.sandbox()
        self.decisions = []
        for p, label, item in zip(parcels, labels, items):
            d = Decision(p.id, label or "")
            self.decisions.append(d)
            if label is None:
                d.note = "的标签看不清，认不出是什么"
                continue
            if item is None:
                d.note = "认不出是什么"
                continue
            d.item = item = settle_category(sandbox, item)
            container, why = choose_container(sandbox, item)
            if container is None:
                biggest = largest_container(sandbox, item.temperature)
                zone = TEMP_NAMES[item.temperature]
                if not biggest:
                    d.note = why  # 家里没有这个温区
                elif item.volume_l > biggest:
                    d.note, d.ask = f"有 {item.volume_l:g} 升，一个容器装不下，机器人不拆包装", "请主人拆开分装"
                else:
                    d.note, d.ask = f"{zone}区的容器都满了，放不下 {item.volume_l:g} 升", "请主人清理或添置容器"
                continue
            record_item(sandbox, item, container, item.volume_l)
            d.container, d.note = container, why

        todo = [d for d in self.decisions if d.container]
        if not todo:
            reasons = "；".join(f"{home.containers[d.parcel].name}{_quote(d.label)}{d.note}" for d in self.decisions)
            raise PlanningError(f"入库台上的东西都放不了：{reasons}")

        b = Builder(world)
        for location in dict.fromkeys(p.location for p in parcels):
            b.inspect(location)  # 先拍一张，看清台上摆了什么、标签写的什么
        for d in self.decisions:
            if d.container is None:
                b.notify(f"{home.containers[d.parcel].name}{_quote(d.label)}{d.note}，先留在入库台上，{d.ask}")
        for d in todo:
            b.pick(d.parcel)
            b.stow(d.parcel, d.container, d.item)
        return b.finish()


def _quote(label: str) -> str:
    return f"「{label}」" if label else ""

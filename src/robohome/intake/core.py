"""入库：把新买回来的东西装进规范的容器里，并记进台账。

分三步：
1. 识别：一句话（或一张照片）→ 物品清单，每件物品带类别、温区、大概体积
2. 分配：按温区找存放区，按类别和剩余容量找容器
3. 记账：在物品台账里加上新物品和它所在的容器

分配规则：
- 温区必须对上：冷冻的进冷冻室，冷藏的进冷藏室，常温的进橱柜
- 同类优先：已经装着同类物品、还放得下的容器优先，挑剩余空间最小的，尽量装满一个再开下一个
- 不混装：一个容器只装一个类别（生肉和饺子不放一起）
- 都不行就开一个空容器，挑放得下的最小那个
- 还是放不下就如实报告，不硬塞

assign 只更新台账（预览用）。让机器人真的做一遍见 planner/intake.py：
主人把买回来的东西摆在入库台上，机器人逐件扫描标签、识别、按上面的规则挑容器，
拿起包装走到容器跟前装进去（技能 stow），台账在装好的那一刻才记上。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from ..spec.models import FOOD_KINDS, PARCEL, Container, Item
from ..world import World

Temperature = Literal["ambient", "chilled", "frozen"]
TEMP_NAMES = {"ambient": "常温", "chilled": "冷藏", "frozen": "冷冻"}


@dataclass
class NewItem:
    name: str
    category: str
    temperature: Temperature
    volume_l: float
    aliases: list[str] = field(default_factory=list)


@dataclass
class Placement:
    item: NewItem
    container: str | None = None
    item_id: str | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.container is not None


@dataclass
class IntakeResult:
    placements: list[Placement] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.placements) and not self.unrecognized

    def render(self, world: World) -> str:
        lines = ["入库结果："]
        for p in self.placements:
            if p.ok:
                c = world.home.containers[p.container]
                storage = world.home.storage.get(c.location)
                where = f"{storage.name if storage else c.location} 槽位 {c.slot}"
                lines.append(f"  ✓ {p.item.name}（{p.item.category}，{TEMP_NAMES[p.item.temperature]}，"
                             f"{p.item.volume_l:g} 升）→ {c.name}（{where}）{p.note}")
            else:
                lines.append(f"  ✗ {p.item.name}：{p.note}")
        for text in self.unrecognized:
            lines.append(f"  ? 没认出来：{text}")
        return "\n".join(lines)


def used_volume(world: World, container_id: str) -> float:
    return sum(i.volume_l for i in world.home.items.values() if i.container == container_id)


def categories_in(world: World, container_id: str) -> set[str]:
    return {i.category for i in world.home.items.values() if i.container == container_id}


def _new_item_id(world: World) -> str:
    n = len(world.home.items) + 1
    while f"item_{n:03d}" in world.home.items:
        n += 1
    return f"item_{n:03d}"


def _food_containers(world: World, temperature: str) -> list:
    """这个温区里能装食物的容器：只算保鲜盒和密封罐，碗杯不算；放在餐桌这类台面上的也不算。"""
    home = world.home
    return [
        c for c in home.containers.values()
        if c.kind in FOOD_KINDS
        and c.location in home.storage
        and home.storage[c.location].kind == "storage"
        and home.storage[c.location].temperature == temperature
    ]


def choose_container(world: World, item: NewItem) -> tuple[str | None, str]:
    """为一件物品挑容器，返回 (容器 id, 说明)。找不到时容器 id 为 None。"""
    home = world.home
    candidates = _food_containers(world, item.temperature)
    if not candidates:
        return None, f"家里没有{TEMP_NAMES[item.temperature]}的存放区"

    def remaining(cid: str) -> float:
        return home.containers[cid].capacity_l - used_volume(world, cid)

    same = [c for c in candidates if categories_in(world, c.id) == {item.category} and remaining(c.id) >= item.volume_l]
    if same:
        best = min(same, key=lambda c: remaining(c.id))
        return best.id, "（和同类放一起）"

    empty = [c for c in candidates if not categories_in(world, c.id) and c.capacity_l >= item.volume_l]
    if empty:
        best = min(empty, key=lambda c: (c.capacity_l, c.id))
        return best.id, "（新开一个空容器）"

    return None, (f"{TEMP_NAMES[item.temperature]}区没有放得下 {item.volume_l:g} 升的空容器，"
                  "需要清理或添置容器")


def settle_category(world: World, item: NewItem) -> NewItem:
    """类别对齐台账：同类才能放一起，"速冻食品"和"速冻"要算同一类。

    1. 台账里已经有这样东西（名字或别名完全一样，如"饺子"），就用它的类别
    2. 否则和已有类别互相包含（"速冻食品"⊃"速冻"、"乳制品类"⊃"乳制品"），就用已有的名字
    3. 都不是就保留，入库时会给它开一个空容器
    """
    names = {item.name, *item.aliases}
    for known in world.home.items.values():
        if known.name in names or names & set(known.aliases):
            return replace(item, category=known.category)
    existing = {i.category for i in world.home.items.values()}
    if item.category in existing:
        return item
    close = [c for c in existing if len(c) >= 2 and len(item.category) >= 2 and (c in item.category or item.category in c)]
    if close:
        return replace(item, category=max(close, key=len))
    return item


def largest_container(world: World, temperature: str) -> float:
    """这个温区里最大的食物容器能装多少升；东西比它还大，就只能拆开分装。"""
    return max((c.capacity_l for c in _food_containers(world, temperature)), default=0.0)


def record_item(world: World, item: NewItem, container: str, volume: float) -> str:
    item_id = _new_item_id(world)
    world.home.items[item_id] = Item(
        id=item_id,
        name=item.name,
        category=item.category,
        container=container,
        aliases=list(item.aliases),
        volume_l=volume,
    )
    return item_id


def _split(world: World, item: NewItem) -> tuple[list[tuple[str, float]], float]:
    """一个容器放不下时（比如一箱牛奶），分装到同温区的多个容器里。返回 [(容器, 体积)] 和放不下的剩余体积。"""
    home = world.home
    remaining = item.volume_l
    parts: list[tuple[str, float]] = []
    used: dict[str, float] = {}

    def room_left(cid: str) -> float:
        return home.containers[cid].capacity_l - used_volume(world, cid) - used.get(cid, 0.0)

    while remaining > 1e-9:
        candidates = [
            c.id for c in _food_containers(world, item.temperature)
            if categories_in(world, c.id) <= {item.category}  # 空的或同类的，不混装
            and room_left(c.id) > 1e-9
        ]
        if not candidates:
            break
        best = max(candidates, key=lambda cid: (room_left(cid), cid))
        portion = round(min(remaining, room_left(best)), 3)
        parts.append((best, portion))
        used[best] = used.get(best, 0.0) + portion
        remaining = round(remaining - portion, 3)
    return parts, remaining


def assign(world: World, items: list[NewItem], commit: bool = True) -> IntakeResult:
    """逐件分配容器。commit=True 时直接写进台账，后面的物品会看到前面物品占掉的空间。

    要预览而不改台账，就在 world.sandbox() 上调用。
    """
    result = IntakeResult()
    for item in items:
        container, note = choose_container(world, item)
        if container:
            placement = Placement(item, container, note=note)
            if commit:
                placement.item_id = record_item(world, item, container, item.volume_l)
            result.placements.append(placement)
            continue

        parts, left = _split(world, item) if item.volume_l > 0 else ([], item.volume_l)
        if len(parts) < 2 and left > 0:
            # 分装也装不下（或根本没有可用容器），整件报告，不拆
            result.placements.append(Placement(item, None, note=note))
            continue
        for n, (cid, volume) in enumerate(parts, 1):
            part = NewItem(item.name, item.category, item.temperature, volume, list(item.aliases))
            placement = Placement(part, cid, note=f"（分装 {n}/{len(parts)}）")
            if commit:
                placement.item_id = record_item(world, part, cid, volume)
            result.placements.append(placement)
        if left > 0:
            rest = NewItem(item.name, item.category, item.temperature, left, list(item.aliases))
            result.placements.append(Placement(rest, None, note=f"还剩 {left:g} 升放不下，需要清理或添置容器"))
    return result


PARCEL_SIZE = (0.2, 0.14, 0.06)  # 仿真里包装的大小：一个扁平的袋子


def intake_areas(world: World) -> list[str]:
    return [s.id for s in world.home.storage.values() if s.kind == "intake"]


def put_parcels(world: World, labels: list[str], area: str | None = None) -> list[str]:
    """主人把买回来的东西摆在入库台上：每件一个包装，标签就是这件东西的说法。返回包装 id。"""
    home = world.home
    areas = [area] if area else intake_areas(world)
    if not areas or areas[0] not in home.storage:
        raise ValueError("家里没有登记入库台（kind 为 intake 的存放区），不知道东西摆在哪")
    area = areas[0]
    free = [s for s in range(home.storage[area].slots) if world.container_at(area, s) is None]
    if len(labels) > len(free):
        raise ValueError(f"{home.storage[area].name}只有 {len(free)} 个空位，摆不下 {len(labels)} 件东西")
    ids = []
    n = 1
    for label, slot in zip(labels, free):
        while f"parcel_{n}" in home.containers:
            n += 1
        pid = f"parcel_{n}"
        home.containers[pid] = Container(
            id=pid, name=f"包装 {n} 号", size_m=PARCEL_SIZE, location=area, slot=slot, kind=PARCEL, label=label
        )
        ids.append(pid)
    return ids


def parcels_on_counter(world: World) -> list[Container]:
    """入库台上等着入库的包装，按台面和槽位排好。"""
    home = world.home
    return sorted(
        (c for c in home.containers.values()
         if c.kind == PARCEL and c.location in home.storage and home.storage[c.location].kind == "intake"),
        key=lambda c: (c.location, c.slot),
    )

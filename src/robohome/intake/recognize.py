"""识别新买的物品：把一句话或一张照片变成物品清单。

- KeywordRecognizer：离线、零成本，靠关键词表，适合测试和演示
- LLMRecognizer：用大模型识别（Claude 或本地模型），能处理任意说法和照片（小票、购物袋）
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.base import JsonLLM, LLMError
from ..llm.claude import ClaudeJSON
from .core import NewItem

TEMPERATURES = ("ambient", "chilled", "frozen")

# 关键词 → (标准名, 类别, 温区, 每份体积升)
# 演示用识别器：只认表里的词，认不出的一律交给主人确认，宁可不认也不乱放。
KEYWORDS: list[tuple[tuple[str, ...], str, str, str, float]] = [
    (("猪肉", "五花肉", "排骨", "里脊", "肉末", "肉馅", "猪蹄"), "猪肉", "肉类", "frozen", 1.0),
    (("牛肉", "牛排", "牛腩"), "牛肉", "肉类", "frozen", 1.0),
    (("鸡翅", "鸡腿", "鸡胸", "鸡肉", "整鸡"), "鸡肉", "肉类", "frozen", 1.0),
    (("羊肉", "羊排"), "羊肉", "肉类", "frozen", 1.0),
    (("虾仁", "大虾", "虾"), "虾", "水产", "frozen", 1.0),
    (("鱼", "三文鱼", "鱼片"), "鱼", "水产", "frozen", 1.5),
    (("饺子", "汤圆", "馄饨"), "速冻饺子", "速冻", "frozen", 1.5),
    (("酸奶",), "酸奶", "乳制品", "chilled", 0.5),
    (("牛奶",), "牛奶", "乳制品", "chilled", 1.0),
    (("鸡蛋",), "鸡蛋", "蛋类", "chilled", 1.5),
    (("苹果",), "苹果", "水果", "chilled", 2.0),
    (("橙子", "橘子"), "橙子", "水果", "chilled", 2.0),
    (("葡萄",), "葡萄", "水果", "chilled", 1.0),
    (("青菜", "白菜", "菠菜", "生菜"), "青菜", "蔬菜", "chilled", 2.0),
    (("西兰花",), "西兰花", "蔬菜", "chilled", 1.5),
    (("大米", "米"), "大米", "干货", "ambient", 2.5),
    (("面粉",), "面粉", "干货", "ambient", 2.0),
    (("挂面", "面条"), "面条", "干货", "ambient", 1.0),
    (("盐", "糖", "酱油", "醋"), "调料", "调料", "ambient", 0.5),
]

# 含这些词的说法是加工食品或别的东西，关键词表判断不了，交给主人确认
AMBIGUOUS = ("糖醋", "奶糖", "糖果", "玉米", "米饭", "米粉", "鱼丸", "虾片", "虾条", "鱼豆腐", "熟")

_SPLIT = re.compile(r"[，,、；;。\s]+|和|还有|以及")
_PREFIX = re.compile(r"^(今天|刚才|刚)?(从超市回来|回来)?(买了|买的|新买的|买)?")
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}
_QTY = re.compile(r"^(\d+(?:\.\d+)?|[一两二三四五六七八九十半])(斤|公斤|千克|克|盒|袋|箱|瓶|包|罐|个|只|条|块|份|把|颗|根|板|提|桶|打)?")
# 单位 → 相当于几份
_UNIT_PORTIONS = {"箱": 6, "提": 6, "打": 12, "板": 1, "公斤": 2, "千克": 2, "克": 0.002}


def _parse_quantity(piece: str) -> tuple[float, str]:
    """"两盒酸奶" → (2, "酸奶")；"一箱牛奶" → (6, "牛奶")；没有数量就是 1 份。"""
    m = _QTY.match(piece)
    if not m:
        return 1.0, piece
    raw, unit = m.group(1), m.group(2) or ""
    number = float(raw) if raw[0].isdigit() else float(_CN_NUM[raw])
    portions = number * _UNIT_PORTIONS.get(unit, 1)
    return max(portions, 0.5), piece[m.end():].strip()


def _match(core: str):
    """多字关键词按包含匹配，单字关键词（米、盐、虾、鱼……）必须整个词就是它，避免"玉米"变成大米。"""
    if any(a in core for a in AMBIGUOUS):
        return None
    best, best_len = None, 0
    for entry in KEYWORDS:
        for k in entry[0]:
            hit = (k in core) if len(k) > 1 else (core == k)
            if hit and len(k) > best_len:
                best, best_len = entry, len(k)
    return best


class KeywordRecognizer:
    def recognize(self, text: str) -> tuple[list[NewItem], list[str]]:
        items, unknown = [], []
        for piece in _SPLIT.split(text):
            piece = _PREFIX.sub("", piece.strip())
            if not piece:
                continue
            portions, core = _parse_quantity(piece)
            entry = _match(core)
            if entry is None:
                unknown.append(piece)
                continue
            _, name, category, temp, vol = entry
            items.append(NewItem(name, category, temp, round(vol * portions, 2)))  # type: ignore[arg-type]
        return items, unknown


RECOGNIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "物品的通用名，如 牛奶、五花肉"},
                    "category": {"type": "string", "description": "类别，优先用已有类别"},
                    "temperature": {"type": "string", "enum": ["ambient", "chilled", "frozen"]},
                    "volume_l": {"type": "number", "description": "大概占用体积，单位升"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "category", "temperature", "volume_l", "aliases"],
                "additionalProperties": False,
            },
        },
        "unrecognized": {"type": "array", "items": {"type": "string"}, "description": "看不清或无法判断的东西"},
    },
    "required": ["items", "unrecognized"],
    "additionalProperties": False,
}

RECOGNIZE_PROMPT = """你在帮一个家庭机器人做"入库"：主人刚买回一些东西，需要列出每一件物品，供机器人分配到标准容器里。

每件物品给出：通用名、类别、存放温区（frozen 冷冻 / chilled 冷藏 / ambient 常温）、大概占用体积（升，按包装估计），以及主人可能用到的别名。
同一种东西买了多份时合并成一件，体积相加。生鲜肉类和水产默认冷冻，除非主人说马上要吃。
家里已有的类别：{categories}。能归入已有类别就用已有的名字。"""


MAX_ITEM_VOLUME_L = 50.0


def validate_items(raw_items: list[dict]) -> tuple[list[NewItem], list[str]]:
    """大模型给的物品逐个检查，不合格的不入账，放进"认不出"让主人确认。"""
    items, rejected = [], []
    for i in raw_items:
        name = str(i.get("name") or "").strip()
        temp = i.get("temperature")
        try:
            volume = float(i.get("volume_l"))
        except (TypeError, ValueError):
            volume = -1.0
        if not name:
            rejected.append("（没有名字的物品）")
        elif temp not in TEMPERATURES:
            rejected.append(f"{name}（温区不明：{temp}）")
        elif not 0 < volume <= MAX_ITEM_VOLUME_L:
            rejected.append(f"{name}（体积不合理：{i.get('volume_l')} 升）")
        else:
            category = str(i.get("category") or "未分类").strip() or "未分类"
            aliases = [str(a) for a in (i.get("aliases") or []) if str(a).strip()]
            items.append(NewItem(name, category, temp, volume, aliases))
    return items, rejected


class LLMRecognizer:
    def __init__(
        self, backend: JsonLLM | None = None, *, client: Any = None, model: str = "claude-opus-5-5", effort: str = "low"
    ):
        self.backend = backend or ClaudeJSON(client=client, model=model, effort=effort)

    def recognize(
        self, text: str = "", image_path: str | None = None, categories: list[str] | None = None
    ) -> tuple[list[NewItem], list[str]]:
        text = (text or "").strip()
        if not text and not image_path:
            # 什么都没给时大模型会凭空编出物品，必须拦住
            raise RuntimeError("识别失败：既没有文字也没有照片，没东西可识别")
        if image_path:
            supports = getattr(self.backend, "supports_vision", None)
            if supports is not None and supports() is False:
                raise RuntimeError(f"识别失败：模型 {getattr(self.backend, 'model', '')} 不支持看图，请换一个视觉模型（如 qwen2.5vl）")
        system = RECOGNIZE_PROMPT.format(categories="、".join(categories or []) or "暂无")
        message = self.backend.user_message(text or "请识别照片里新买的东西。", image_path)
        try:
            data, _ = self.backend.complete_json(system, [message], RECOGNIZE_SCHEMA, "intake_items")
        except LLMError as e:
            raise RuntimeError(f"识别失败：{e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise RuntimeError("识别失败：返回的结构不对")
        items, rejected = validate_items(data["items"])
        unknown = [str(u) for u in (data.get("unrecognized") or [])]
        return items, unknown + rejected

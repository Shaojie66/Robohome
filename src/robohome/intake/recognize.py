"""识别新买的物品：把一句话或一张照片变成物品清单。

- KeywordRecognizer：离线、零成本，靠关键词表，适合测试和演示
- LLMRecognizer：用 Claude 识别，能处理任意说法和照片（小票、购物袋）
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from .core import NewItem

# 关键词 → (标准名, 类别, 温区, 默认体积升)
KEYWORDS: list[tuple[tuple[str, ...], str, str, str, float]] = [
    (("猪肉", "五花肉", "排骨"), "猪肉", "肉类", "frozen", 1.0),
    (("牛肉", "牛排"), "牛肉", "肉类", "frozen", 1.0),
    (("鸡翅", "鸡腿", "鸡胸", "鸡肉"), "鸡肉", "肉类", "frozen", 1.0),
    (("羊肉",), "羊肉", "肉类", "frozen", 1.0),
    (("虾",), "虾", "水产", "frozen", 1.0),
    (("鱼",), "鱼", "水产", "frozen", 1.5),
    (("饺子", "汤圆", "馄饨"), "速冻饺子", "速冻", "frozen", 1.5),
    (("酸奶",), "酸奶", "乳制品", "chilled", 1.0),
    (("牛奶",), "牛奶", "乳制品", "chilled", 2.0),
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

_SPLIT = re.compile(r"[，,、；;。\s]+|和|还有|以及")
_PREFIX = re.compile(r"^(今天|刚才|刚)?(买了|买的|新买的|买)?")


class KeywordRecognizer:
    def recognize(self, text: str) -> tuple[list[NewItem], list[str]]:
        items, unknown = [], []
        for piece in _SPLIT.split(text):
            piece = _PREFIX.sub("", piece.strip())
            if not piece:
                continue
            for keys, name, category, temp, vol in KEYWORDS:
                if any(k in piece for k in keys):
                    items.append(NewItem(name, category, temp, vol))  # type: ignore[arg-type]
                    break
            else:
                unknown.append(piece)
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


class LLMRecognizer:
    def __init__(self, client: Any = None, model: str = "claude-opus-5-5", effort: str = "low"):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def recognize(
        self, text: str = "", image_path: str | None = None, categories: list[str] | None = None
    ) -> tuple[list[NewItem], list[str]]:
        content: list[dict] = []
        if image_path:
            media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        content.append({"type": "text", "text": text or "请识别照片里新买的东西。"})

        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=RECOGNIZE_PROMPT.format(categories="、".join(categories or []) or "暂无"),
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": RECOGNIZE_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"识别失败：{response.stop_reason}")
        text_out = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text_out)
        items = [
            NewItem(i["name"], i["category"], i["temperature"], float(i["volume_l"]), list(i["aliases"]))
            for i in data["items"]
        ]
        return items, list(data["unrecognized"])

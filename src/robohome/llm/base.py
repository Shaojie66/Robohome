"""大模型后端的统一接口。

规划层和入库模块只需要一件事：给一段系统提示、一串消息和一个 JSON Schema，拿回一份符合 Schema 的 JSON。
不同后端（Claude、本地 Ollama / llama.cpp / vLLM）各自实现这个接口，上层代码不用关心用的是哪个。
"""

from __future__ import annotations

import re
from typing import Protocol


class LLMError(RuntimeError):
    """大模型调用失败：拒绝、截断、连不上、返回的不是合法 JSON。"""


class JsonLLM(Protocol):
    name: str

    def user_message(self, text: str, image_path: str | None = None) -> dict:
        """按这个后端的格式构造一条用户消息（可带一张图片）。"""
        ...

    def complete_json(self, system: str, messages: list[dict], schema: dict, schema_name: str) -> tuple[dict, dict]:
        """返回 (解析好的 JSON, 要追加到对话里的 assistant 消息)。"""
        ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def clean_json_text(text: str) -> str:
    """去掉部分本地模型会带上的思考段落和代码块围栏。"""
    text = _THINK.sub("", text).strip()
    return _FENCE.sub("", text).strip()

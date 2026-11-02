"""大模型后端：Claude（云端）或本地模型（Ollama 等），接口一致，可以随时切换。"""

from __future__ import annotations

from .base import JsonLLM, LLMError
from .claude import ClaudeJSON
from .local import LocalJSON


def make_backend(kind: str, model: str | None = None, base_url: str | None = None, effort: str = "medium") -> JsonLLM:
    if kind == "claude":
        return ClaudeJSON(model=model or "claude-opus-5-5", effort=effort)
    if kind == "local":
        kwargs = {"model": model} if model else {}
        if base_url:
            kwargs["base_url"] = base_url
        return LocalJSON(**kwargs)
    raise ValueError(f"不认识的后端：{kind}")


__all__ = ["ClaudeJSON", "JsonLLM", "LLMError", "LocalJSON", "make_backend"]

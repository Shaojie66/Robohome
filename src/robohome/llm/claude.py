"""Claude 后端：Anthropic SDK + 结构化输出。"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from .base import LLMError

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeJSON:
    name = "claude"

    def __init__(self, client: Any = None, model: str = DEFAULT_MODEL, effort: str = "medium"):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise LLMError('没有安装 Anthropic SDK，先运行：pip install -e ".[claude]"') from e
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def user_message(self, text: str, image_path: str | None = None) -> dict:
        if not image_path:
            return {"role": "user", "content": text}
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
        return {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": text},
            ],
        }

    def complete_json(self, system: str, messages: list[dict], schema: dict, schema_name: str) -> tuple[dict, dict]:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                cache_control={"type": "ephemeral"},
                # 安全分类器误拒时，由服务端自动换模型重试
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except TypeError as e:
            if "authentication" in str(e):
                raise LLMError("没有找到 Claude 的密钥，先设置环境变量 ANTHROPIC_API_KEY") from e
            raise
        except Exception as e:
            if _is_sdk_error(e):
                raise LLMError(f"调用 Claude 失败：{e}") from e
            raise
        if response.stop_reason == "refusal":
            raise LLMError("大模型拒绝了这条指令")
        if response.stop_reason == "max_tokens":
            raise LLMError("输出被截断")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("大模型没有返回内容")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"返回的不是合法 JSON：{e}") from e
        # 把回复原样放回对话（只追加，不改写）
        return data, {"role": "assistant", "content": response.content}


def _is_sdk_error(e: Exception) -> bool:
    """Anthropic SDK 的异常（连不上、限流、密钥无效……）都转成 LLMError，上层统一处理。"""
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(e, anthropic.AnthropicError)

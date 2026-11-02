"""本地大模型后端：任何兼容 OpenAI Chat Completions 接口的本地服务。

默认连 Ollama（http://localhost:11434/v1），同样适用于 llama.cpp server、vLLM、LM Studio。
用 response_format 的 json_schema 约束输出，服务端会按 Schema 做语法约束解码，小模型也能稳定输出合法 JSON。
用流式返回：只要模型还在吐字就继续等，超过 idle_timeout 秒一个字都没有就判定卡住，重试一次。
（在 Mac 上实测时遇到过 Ollama 连接挂住不出字，非流式要等满整个超时才发现。）
只用标准库，不需要额外依赖。

    ollama pull qwen3:8b
    robohome run "把牛肉拿出来解冻" --planner local
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from .base import LLMError, clean_json_text

DEFAULT_BASE_URL = os.environ.get("ROBOHOME_LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("ROBOHOME_LLM_MODEL", "qwen3:8b")


class LocalJSON:
    name = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        idle_timeout: float = 60.0,
        total_timeout: float = 300.0,
        retries: int = 1,
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.idle_timeout = idle_timeout  # 多久没有任何输出就算卡住
        self.total_timeout = total_timeout  # 单次请求最长允许多久
        self.retries = retries  # 卡住或超时后重试几次
        self.api_key = api_key or os.environ.get("ROBOHOME_LLM_API_KEY")
        self.temperature = temperature

    def user_message(self, text: str, image_path: str | None = None) -> dict:
        if not image_path:
            return {"role": "user", "content": text}
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
                {"type": "text", "text": text},
            ],
        }

    def complete_json(self, system: str, messages: list[dict], schema: dict, schema_name: str) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        }
        text = ""
        for attempt in range(self.retries + 1):
            try:
                text, finish_reason = self._stream("/chat/completions", body)
                break
            except _Stalled as e:
                if attempt == self.retries:
                    raise LLMError(f"本地模型{e}，已重试 {self.retries} 次") from e
        if finish_reason == "length":
            raise LLMError("输出被截断")
        try:
            data = json.loads(clean_json_text(text))
        except json.JSONDecodeError as e:
            raise LLMError(f"返回的不是合法 JSON：{e}；原文开头：{text[:200]}") from e
        return data, {"role": "assistant", "content": text}

    def _stream(self, path: str, body: dict) -> tuple[str, str | None]:
        """流式请求，返回 (完整文本, finish_reason)。卡住或超时抛 _Stalled。"""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps({**body, "stream": True}, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )
        started = time.monotonic()
        parts: list[str] = []
        finish_reason = None
        try:
            # timeout 作用于每一次读取：模型持续输出就不会触发，停住超过 idle_timeout 才触发
            with urllib.request.urlopen(req, timeout=self.idle_timeout) as resp:
                for raw in resp:
                    if time.monotonic() - started > self.total_timeout:
                        raise _Stalled(f"{self.total_timeout:g} 秒内没有完成")
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choice = chunk["choices"][0]
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                        raise LLMError(f"本地模型返回格式不对：{data[:200]}") from e
                    parts.append((choice.get("delta") or {}).get("content") or "")
                    finish_reason = choice.get("finish_reason") or finish_reason
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"本地模型服务报错 {e.code}：{detail}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise _Stalled(f"{self.idle_timeout:g} 秒没有响应") from e
            raise LLMError(
                f"连不上本地模型服务 {self.base_url}（{e.reason}）。先启动 Ollama 并拉取模型：ollama pull {self.model}"
            ) from e
        except (TimeoutError, socket.timeout) as e:
            raise _Stalled(f"{self.idle_timeout:g} 秒没有任何输出，像是卡住了") from e
        return "".join(parts), finish_reason


class _Stalled(Exception):
    """本地模型卡住或超时，可以重试。"""

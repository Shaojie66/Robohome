"""本地大模型后端：任何兼容 OpenAI Chat Completions 接口的本地服务。

默认连 Ollama（http://localhost:11434/v1），同样适用于 llama.cpp server、vLLM、LM Studio。
用 response_format 的 json_schema 约束输出，服务端会按 Schema 做语法约束解码，小模型也能稳定输出合法 JSON。
只用标准库，不需要额外依赖。

    ollama pull qwen3:8b
    robohome run "把牛肉拿出来解冻" --planner local
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
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
        timeout: float = 600.0,
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
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
        reply = self._post("/chat/completions", body)
        try:
            choice = reply["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"本地模型返回格式不对：{str(reply)[:200]}") from e
        if choice.get("finish_reason") == "length":
            raise LLMError("输出被截断")
        try:
            data = json.loads(clean_json_text(text))
        except json.JSONDecodeError as e:
            raise LLMError(f"返回的不是合法 JSON：{e}；原文开头：{text[:200]}") from e
        return data, {"role": "assistant", "content": text}

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"本地模型服务报错 {e.code}：{detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"连不上本地模型服务 {self.base_url}（{e.reason}）。先启动 Ollama 并拉取模型：ollama pull {self.model}"
            ) from e
        except TimeoutError as e:
            raise LLMError(f"本地模型 {self.timeout:g} 秒内没有返回") from e

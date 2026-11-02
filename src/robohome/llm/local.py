"""本地大模型后端：任何兼容 OpenAI Chat Completions 接口的本地服务。

默认连 Ollama（http://localhost:11434/v1），同样适用于 llama.cpp server、vLLM、LM Studio。
用 response_format 的 json_schema 约束输出，服务端会按 Schema 做语法约束解码，小模型也能稳定输出合法 JSON。

用流式返回，分三种等待：
- 第一个字之前最多等 first_byte_timeout 秒（冷启动要先把模型读进内存，可能比较久）；
- 开始出字后，超过 idle_timeout 秒一个字都没有就判定卡住；
- 卡住、连接中途断开、流没正常结束，都重试 retries 次；整次请求超过 total_timeout 秒直接报错，不重试。
（在 Mac 上实测时遇到过 Ollama 连接挂住不出字，非流式要等满整个超时才发现。）
只用标准库，不需要额外依赖。

    ollama pull qwen3:8b
    robohome run "把牛肉拿出来解冻" --planner local
"""

from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

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
        first_byte_timeout: float | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.idle_timeout = idle_timeout  # 出字之后，多久没有任何输出就算卡住
        # 第一个字之前允许等多久；冷启动加载模型要时间，默认是 idle_timeout 的 3 倍
        self.first_byte_timeout = first_byte_timeout if first_byte_timeout is not None else idle_timeout * 3
        self.total_timeout = total_timeout  # 单次请求最长允许多久，超过不重试
        self.retries = max(0, int(retries))  # 卡住或断开后重试几次
        self.api_key = api_key or os.environ.get("ROBOHOME_LLM_API_KEY")
        self.temperature = temperature
        self._vision: dict[str, bool | None] = {}

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

    def supports_vision(self) -> bool | None:
        """问 Ollama 这个模型能不能看图。True / False；不是 Ollama 或问不到时返回 None（不确定，照常尝试）。"""
        if self.model in self._vision:
            return self._vision[self.model]
        result: bool | None = None
        root = self.base_url.removesuffix("/v1")
        try:
            status, raw = self._post_once(root + "/api/show", {"model": self.model}, timeout=5)
            if status == 200:
                caps = json.loads(raw).get("capabilities")
                if isinstance(caps, list):
                    result = "vision" in caps
        except (OSError, http.client.HTTPException, ValueError, AttributeError, LLMError, _Retryable):
            result = None
        self._vision[self.model] = result
        return result

    def complete_json(self, system: str, messages: list[dict], schema: dict, schema_name: str) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self.temperature,
            "stream": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        }
        text, finish_reason = "", None
        for attempt in range(self.retries + 1):
            try:
                text, finish_reason = self._stream("/chat/completions", body)
                break
            except _Retryable as e:
                if attempt == self.retries:
                    tail = f"，已重试 {self.retries} 次" if self.retries else ""
                    raise LLMError(f"本地模型{e}{tail}") from e
        if finish_reason == "length":
            raise LLMError("输出被截断")
        try:
            data = json.loads(clean_json_text(text))
        except json.JSONDecodeError as e:
            raise LLMError(f"返回的不是合法 JSON：{e}；原文开头：{text[:200]}") from e
        return data, {"role": "assistant", "content": text}

    # ---- HTTP ----

    def _headers(self, stream: bool) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _connect(self, url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
        parts = urlsplit(url)
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        conn = cls(parts.hostname or "localhost", parts.port, timeout=timeout)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            conn.connect()
        except TimeoutError as e:
            conn.close()
            raise _Retryable(f"{timeout:g} 秒内连不上") from e
        except OSError as e:
            conn.close()
            raise LLMError(
                f"连不上本地模型服务 {self.base_url}（{e}）。先启动 Ollama 并拉取模型：ollama pull {self.model}"
            ) from e
        return conn, target

    def _post_once(self, url: str, body: dict, timeout: float) -> tuple[int, bytes]:
        conn, target = self._connect(url, timeout)
        try:
            conn.request("POST", target, body=json.dumps(body).encode("utf-8"), headers=self._headers(False))
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def _stream(self, path: str, body: dict) -> tuple[str, str | None]:
        """流式请求，返回 (完整文本, finish_reason)。卡住、断开、没正常结束抛 _Retryable。"""
        started = time.monotonic()
        conn, target = self._connect(self.base_url + path, self.first_byte_timeout)
        sock = conn.sock  # 先拿住：收到第一个字后把读超时从 first_byte_timeout 换成 idle_timeout
        parts: list[str] = []
        finish_reason = None
        done = False
        waiting = f"{self.first_byte_timeout:g} 秒没有响应（模型可能还在加载）"
        try:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            conn.request("POST", target, body=payload, headers=self._headers(True))
            resp = conn.getresponse()
            if resp.status >= 400:
                detail = resp.read().decode("utf-8", "replace")[:300]
                raise LLMError(f"本地模型服务报错 {resp.status}：{detail}")
            first = True
            for raw in resp:
                if first:
                    sock.settimeout(self.idle_timeout)
                    waiting = f"{self.idle_timeout:g} 秒没有任何输出，像是卡住了"
                    first = False
                if time.monotonic() - started > self.total_timeout:
                    raise LLMError(f"本地模型 {self.total_timeout:g} 秒内没有完成，放弃这次请求")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as e:
                    raise LLMError(f"本地模型返回格式不对：{data[:200]}") from e
                if not isinstance(chunk, dict):
                    raise LLMError(f"本地模型返回格式不对：{data[:200]}")
                if chunk.get("error"):
                    err = chunk["error"]
                    raise LLMError(f"本地模型服务报错：{err.get('message', err) if isinstance(err, dict) else err}")
                choices = chunk.get("choices")
                if not choices:  # 有的服务会单独发一条只带 usage 的数据
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
        except TimeoutError as e:
            raise _Retryable(waiting) from e
        except (ConnectionError, http.client.IncompleteRead) as e:
            raise _Retryable(f"连接中途断开（{type(e).__name__}）") from e
        except (http.client.HTTPException, OSError) as e:
            raise _Retryable(f"连接出错（{type(e).__name__}：{e}）") from e
        finally:
            conn.close()
        if not done and finish_reason is None:
            raise _Retryable("输出没有正常结束就断开了")
        return "".join(parts), finish_reason


class _Retryable(Exception):
    """本地模型卡住、连接断开或流没正常结束，可以重试。"""

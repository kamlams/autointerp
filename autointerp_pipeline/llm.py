"""Minimal OpenAI-compatible chat client (works with RiVault, vLLM, any /v1/chat/completions).

Credentials resolution order:
  1. env RIVAULT_BASE_URL / RIVAULT_API_KEY
  2. ~/.config/rivault.env (shell-style `export KEY="value"` lines)
  3. ~/.config/opencode/opencode.json (opencode provider config)
Never hardcode keys in this repo.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

RIVAULT_ENV = os.path.expanduser("~/.config/rivault.env")
OPENCODE_CONFIG = os.path.expanduser("~/.config/opencode/opencode.json")

_ENV_LINE = re.compile(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*"?([^"\n]*)"?\s*$')


def _read_env_file(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = _ENV_LINE.match(line.strip())
                if m:
                    out[m.group(1)] = m.group(2)
    return out


def resolve_credentials() -> tuple[str, str]:
    base_url = os.environ.get("RIVAULT_BASE_URL")
    api_key = os.environ.get("RIVAULT_API_KEY")
    if base_url and api_key:
        return base_url.rstrip("/"), api_key
    filed = _read_env_file(RIVAULT_ENV)
    base_url = base_url or filed.get("RIVAULT_BASE_URL")
    api_key = api_key or filed.get("RIVAULT_API_KEY")
    if base_url and api_key:
        return base_url.rstrip("/"), api_key
    if os.path.exists(OPENCODE_CONFIG):
        with open(OPENCODE_CONFIG) as f:
            cfg = json.load(f)
        for prov in cfg.get("provider", {}).values():
            opts = prov.get("options", {})
            if opts.get("baseURL") and opts.get("apiKey"):
                return (base_url or opts["baseURL"]).rstrip("/"), (api_key or opts["apiKey"])
    raise RuntimeError(
        "No API credentials: set RIVAULT_BASE_URL and RIVAULT_API_KEY, "
        f"or put baseURL/apiKey in {OPENCODE_CONFIG}"
    )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    n_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: dict):
        with self._lock:
            self.n_calls += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens += details.get("reasoning_tokens", 0)

    def summary(self) -> str:
        return (
            f"{self.n_calls} calls, {self.prompt_tokens} prompt tok, "
            f"{self.completion_tokens} completion tok (of which {self.reasoning_tokens} reasoning)"
        )


class _LengthTruncated(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: int = 600, max_retries: int = 4, log_path: str | None = None):
        if base_url is None or api_key is None:
            rb, rk = resolve_credentials()
            base_url = base_url or rb
            api_key = api_key or rk
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self.log_path = log_path
        self._log_lock = threading.Lock()

    def chat(self, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.0,
             tag: str = "") -> str:
        # Streamed (SSE) on purpose: long reasoning generations (>~10 min) exceed the
        # RiVault gateway's HTTP timeout on non-streaming requests (nginx 504), while a
        # stream delivers bytes continuously and survives.
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if isinstance(last_err, _LengthTruncated):
                    # reasoning models can burn the whole budget thinking; give room
                    payload["max_tokens"] = min(payload["max_tokens"] * 2, 131072)
                content, reasoning, finish, usage = self._stream_once(payload)
                self.usage.add(usage)
                self._log(tag, messages,
                          {"content": content, "reasoning_content": reasoning}, usage)
                if not content.strip():
                    if finish == "length":
                        # reasoning model ran out of budget before emitting text
                        raise _LengthTruncated(f"empty content, max_tokens={payload['max_tokens']}")
                    raise RuntimeError(f"empty content (finish_reason={finish})")
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2.0 * (2 ** attempt)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def _stream_once(self, payload: dict) -> tuple[str, str, str | None, dict]:
        resp = requests.post(
            self.url,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            json=payload,
            stream=True,
            # (connect, read): the read timeout applies between chunks, not to the whole
            # generation, so slow-but-alive streams are fine and true stalls still fail
            timeout=(30, 120),
        )
        if resp.status_code != 200:
            body = resp.text[:200]
            resp.close()
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")
        content_parts, reasoning_parts = [], []
        finish, usage = None, {}
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        finally:
            resp.close()
        return "".join(content_parts), "".join(reasoning_parts), finish, usage

    def _log(self, tag: str, messages: list[dict], reply_msg: dict, usage: dict):
        if not self.log_path:
            return
        rec = {"tag": tag, "model": self.model, "messages": messages,
               "reply": reply_msg.get("content"),
               "reasoning": (reply_msg.get("reasoning_content") or "")[:2000],
               "usage": usage, "ts": time.time()}
        with self._log_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def map_concurrent(fn, items: list, workers: int = 4) -> list:
    """Apply fn over items with a thread pool, preserving order. Exceptions propagate."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))

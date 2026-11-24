"""OpenAI Chat Completions 兼容 provider（PyTRi0 / 私有化部署 / 任意兼容端点）。

为什么单独一层
--------------
赛题要求"驱动 Harness 的大模型须为 PyTRi0 的 Qwen3.8-27B 或选手在 PyTRi0
训练的模型"。PyTRi0 的推理服务暴露 OpenAI 兼容接口，因此只要把
`base_url/model/api_key` 三件套通过环境变量注入，本 provider 无需改代码即可切换
到官方模型；校赛阶段没有额度时默认关闭（`enabled=False`），`complete()` 立即抛
`ProviderUnavailable`（**不发起任何网络请求**），由 LLMDriver 回退确定性策略。

环境变量（按优先级，第一个非空生效）
------------------------------------
```
base_url : ZHIXUN_LLM_BASE_URL  OPENAI_BASE_URL  LLM_BASE_URL  PYTRI0_BASE_URL
api_key  : ZHIXUN_LLM_API_KEY   OPENAI_API_KEY   LLM_API_KEY   PYTRI0_API_KEY
model    : ZHIXUN_LLM_MODEL     OPENAI_MODEL     LLM_MODEL     PYTRI0_MODEL
timeout  : ZHIXUN_LLM_TIMEOUT   （秒，默认 20）
```
示例（PyTRi0 Qwen3.8-27B）：
```bash
export ZHIXUN_LLM_BASE_URL="https://api.pytri0.example/v1"   # 以官方下发为准
export ZHIXUN_LLM_API_KEY="..."
export ZHIXUN_LLM_MODEL="Qwen3.8-27B"
python -m src.cli run --config configs/llm_llm_driver.json
```

安全
----
- 密钥只从环境变量读取，绝不写入配置/日志/记录；`health()` 只报告是否已设置；
- 运行期除本 provider 的显式调用外无网络（`cfg.safety.allow_network` 仅作展示，
  真正的开关是"是否配置了 base_url + provider=openai_compat"）；
- 失败快速降级：连接超时/HTTP 错误/响应非法都抛 `ProviderError`，不重试（默认
  `max_retries=0`），由 Harness 层回退，避免闭环卡死。
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .base import (
    API_KEY_ENVS,
    BASE_URL_ENVS,
    MODEL_ENVS,
    ProviderError,
    ProviderResult,
    ProviderUnavailable,
    count_message_tokens,
    estimate_tokens,
    first_env,
    redact,
)


class OpenAICompatProvider:
    """OpenAI `/chat/completions` 兼容 provider（默认关闭、离线快速失败）。"""

    name = "openai_compat"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout_s: float = 20.0,
                 env: Optional[dict] = None, cfg: Optional[dict] = None,
                 max_retries: int = 0):
        env = dict(os.environ) if env is None else dict(env)
        self.base_url = (base_url or first_env(env, BASE_URL_ENVS)).strip()
        self.api_key = (api_key or first_env(env, API_KEY_ENVS)).strip()
        default_model = ""
        if cfg:
            default_model = str((cfg.get("harness") or {}).get("model", "") or "")
            if default_model in ("mock-trainer-v1", "mock"):
                default_model = ""
        self.model = (model or first_env(env, MODEL_ENVS) or default_model).strip()
        timeout_env = env.get("ZHIXUN_LLM_TIMEOUT")
        try:
            self.timeout_s = float(timeout_env) if timeout_env else float(timeout_s)
        except (TypeError, ValueError):
            self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.calls: list[dict] = []

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    @classmethod
    def from_env(cls, cfg: Optional[dict] = None, env: Optional[dict] = None,
                 **kwargs) -> "OpenAICompatProvider":
        return cls(cfg=cfg, env=env, **kwargs)

    @property
    def url(self) -> str:
        base = self.base_url.rstrip("/")
        if not base:
            return ""
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def health(self) -> dict:
        return {
            "provider": self.name,
            "enabled": self.enabled,
            "base_url": redact(self.base_url),
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "timeout_s": self.timeout_s,
            "network_required": self.enabled,
        }

    # ------------------------------------------------------------------
    def _payload(self, messages: list[dict], temperature: float,
                 max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

    def complete(self, messages: list[dict], temperature: float = 0.2,
                 max_tokens: int = 2048) -> ProviderResult:
        if not self.enabled:
            raise ProviderUnavailable(
                "openai_compat provider 未启用：请设置 "
                "ZHIXUN_LLM_BASE_URL / ZHIXUN_LLM_MODEL（以及需要时的 "
                "ZHIXUN_LLM_API_KEY），或改用 cfg.harness.provider=mock")
        prompt_tokens_est = count_message_tokens(messages)
        payload = self._payload(messages, temperature, max_tokens)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers,
                                             method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw_text = resp.read().decode("utf-8", errors="replace")
                wall = time.time() - t0
                result = self._parse_response(raw_text, wall,
                                              prompt_tokens_est)
                self.calls.append({"url": redact(self.url), "ok": True,
                                   "prompt_tokens": result.prompt_tokens,
                                   "completion_tokens": result.completion_tokens,
                                   "wall_s": wall})
                return result
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:  # noqa: BLE001
                    pass
                last_exc = ProviderError(
                    f"HTTP {exc.code}（{self.url}）：{redact(detail)}")
                # 4xx 配置/鉴权错误不重试；5xx 才允许重试
                if exc.code < 500:
                    break
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    ConnectionError, OSError) as exc:
                last_exc = ProviderUnavailable(
                    f"连接失败（{self.url}）：{type(exc).__name__}: {exc}；"
                    "请检查网络/端点，或改用 mock provider 离线运行")
                if attempt >= self.max_retries:
                    break
            except ProviderError as exc:  # 解析错误
                last_exc = exc
                break
            except Exception as exc:  # noqa: BLE001 - 防御
                last_exc = ProviderError(f"{type(exc).__name__}: {exc}")
                break
        self.calls.append({"url": redact(self.url), "ok": False,
                           "error": str(last_exc)})
        raise last_exc or ProviderError("未知 provider 错误")

    # ------------------------------------------------------------------
    def _parse_response(self, raw_text: str, wall_s: float,
                        prompt_tokens_est: int) -> ProviderResult:
        try:
            data = json.loads(raw_text)
        except ValueError as exc:
            raise ProviderError(f"响应不是合法 JSON：{redact(raw_text[:200])}") from exc
        if not isinstance(data, dict):
            raise ProviderError("响应 JSON 顶层不是对象")
        if "error" in data and data.get("error"):
            raise ProviderError(f"服务端错误：{redact(str(data['error'])[:300])}")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("响应缺少 choices")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return ProviderResult(
            text=text,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None
            else prompt_tokens_est,
            completion_tokens=int(completion_tokens) if completion_tokens is not None
            else estimate_tokens(text),
            wall_s=float(wall_s),
            model=str(data.get("model") or self.model),
            finish_reason=str(choices[0].get("finish_reason") or ""),
            raw={"id": data.get("id"), "usage": usage},
        )


__all__ = ["OpenAICompatProvider"]

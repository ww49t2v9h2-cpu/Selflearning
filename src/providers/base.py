"""LLM Provider 协议与离线 token 估算。

约束（docs/03 §5.3）
--------------------
- 系统运行期默认**零网络**；只有显式 `cfg.harness.provider=openai_compat`
  且环境变量配置齐全时才会发起 HTTP 请求，且失败快速降级；
- token 计量必须可复现：无真实 tokenizer 时按字符启发式估算（CJK 1 字≈1 token，
  其它字符 4 字≈1 token），误差方向保守（宁可高估）；
- `ProviderResult` 字段与契约一致：text, prompt_tokens, completion_tokens,
  wall_s, model，另附 raw（原始响应，便于审计）与 total_tokens 便利属性。
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# 环境变量优先级（从高到低）。PyTRi0 官方端点兼容 OpenAI Chat Completions。
BASE_URL_ENVS = ("ZHIXUN_LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL",
                 "PYTRI0_BASE_URL")
API_KEY_ENVS = ("ZHIXUN_LLM_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY",
                "PYTRI0_API_KEY")
MODEL_ENVS = ("ZHIXUN_LLM_MODEL", "OPENAI_MODEL", "LLM_MODEL", "PYTRI0_MODEL")

SNAPSHOT_PREFIX = "【状态快照】"
RESULT_PREFIX = "【工具结果】"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class ProviderError(RuntimeError):
    """provider 调用失败（解析/HTTP/协议错误）。"""


class ProviderUnavailable(ProviderError):
    """provider 未配置或网络不可达——调用方应回退确定性策略。"""


@dataclass
class ProviderResult:
    """一次 LLM 补全结果（契约字段 + 审计附加字段）。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    model: str = ""
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "wall_s": self.wall_s,
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


@runtime_checkable
class LLMProvider(Protocol):
    """provider 协议（docs/03 §5.3）。"""

    def complete(self, messages: list[dict], temperature: float = 0.2,
                 max_tokens: int = 2048) -> ProviderResult: ...


def estimate_tokens(text: str) -> int:
    """离线 token 估算：CJK≈1 token/字，其它≈1 token/4 字符。

    与 Qwen/LLaMA 类 BPE 的实测偏差通常在 ±15% 内；用于预算硬约束时
    向上取整，保证不会因为低估而超额。
    """
    if not text:
        return 0
    text = str(text)
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk + math.ceil(other / 4.0))


def count_message_tokens(messages: list[dict], overhead_per_message: int = 4) -> int:
    """估算一次请求的 prompt tokens（含 role/分隔符开销）。"""
    total = 0
    for msg in messages or []:
        total += overhead_per_message
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if isinstance(content, str):
            total += estimate_tokens(content)
        else:  # 多模态/工具参数等
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        total += estimate_tokens(str(msg.get("role", ""))) if isinstance(msg, dict) else 0
    return total


def redact(text: str) -> str:
    """日志脱敏：把疑似密钥（sk-... / 32+ 位十六进制）替换为 ***。"""
    if not text:
        return text
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-***", str(text))
    text = re.sub(r"\b[a-fA-F0-9]{32,}\b", "***", text)
    return text


def make_result(text: str, model: str, prompt_tokens: Optional[int] = None,
                completion_tokens: Optional[int] = None,
                wall_s: float = 0.0, finish_reason: str = "stop") -> ProviderResult:
    return ProviderResult(
        text=text,
        prompt_tokens=estimate_tokens(text) * 2 if prompt_tokens is None else int(prompt_tokens),
        completion_tokens=estimate_tokens(text) if completion_tokens is None
        else int(completion_tokens),
        wall_s=float(wall_s), model=model, finish_reason=finish_reason,
    )


def first_env(env: dict, names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = env.get(name)
        if value:
            return str(value)
    return default


__all__ = [
    "SNAPSHOT_PREFIX", "RESULT_PREFIX", "ProviderResult", "LLMProvider", "ProviderError", "ProviderUnavailable",
    "estimate_tokens", "count_message_tokens", "redact", "make_result",
    "first_env", "BASE_URL_ENVS", "API_KEY_ENVS", "MODEL_ENVS",
]

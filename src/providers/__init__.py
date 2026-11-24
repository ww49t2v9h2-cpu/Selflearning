"""Provider 层：LLM 补全的可插拔后端。

- `base.py`：ProviderResult / LLMProvider 协议 / 离线 token 估算（契约 §5.3）；
- `mock_llm.py`：确定性离线模拟（默认，无网络，按字符估算 token）；
- `openai_compat.py`：OpenAI Chat Completions 兼容（PyTRi0 Qwen3.8-27B 可直连），
  默认关闭、无配置时调用立即失败并回退。
"""

from .base import (  # noqa: F401
    ProviderError,
    ProviderResult,
    ProviderUnavailable,
    LLMProvider,
    count_message_tokens,
    estimate_tokens,
    redact,
)
from .mock_llm import MockLLMProvider  # noqa: F401
from .openai_compat import OpenAICompatProvider  # noqa: F401

__all__ = [
    "LLMProvider", "ProviderResult", "ProviderError", "ProviderUnavailable",
    "MockLLMProvider", "OpenAICompatProvider",
    "estimate_tokens", "count_message_tokens", "redact",
]

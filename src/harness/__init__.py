"""Harness 层：工具（12 个）+ 技能 + 确定性/LLM 驱动。

对外常用入口：
```python
from src.harness.tools import build_tool_registry, build_tool_context, ToolPolicy
from src.harness.agent import build_driver, DeterministicDriver, LLMDriver

ctx = build_tool_context(cfg, budget=budget, state=loop_state,
                         oracle=oracle, library=library, registry=registry,
                         safety=safety, tracker=tracker, planner=planner,
                         diagnoser=diagnoser)
tools = build_tool_registry(ctx, cfg=cfg)
driver = build_driver(cfg)
traj = driver.decide(snapshot, tools)
```
"""

from .agent import (  # noqa: F401
    DeterministicDriver,
    DeterministicRules,
    HarnessDriver,
    LLMDriver,
    build_driver,
    harness_turn,
    make_provider,
)
from .skills import (  # noqa: F401
    build_system_prompt,
    load_skills,
    skill_index,
    skill_titles,
    skills_by_ids,
)
from .tools import (  # noqa: F401
    DEFAULT_TOOLS,
    DefaultToolContext,
    ToolContext,
    ToolError,
    ToolPolicy,
    ToolRegistry,
    build_tool_context,
    build_tool_registry,
    to_plain,
    validate_args,
)

__all__ = [
    "ToolRegistry", "ToolPolicy", "ToolContext", "DefaultToolContext",
    "ToolError", "build_tool_registry", "build_tool_context", "DEFAULT_TOOLS",
    "validate_args", "to_plain",
    "HarnessDriver", "DeterministicDriver", "DeterministicRules", "LLMDriver",
    "build_driver", "make_provider", "harness_turn",
    "load_skills", "skill_titles", "skill_index", "skills_by_ids",
    "build_system_prompt",
]

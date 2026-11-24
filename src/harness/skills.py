"""技能加载与 system prompt 拼装（skills/*.md → LLM 驱动上下文）。

技能文件是**中文 Markdown**，既要给人看（评分材料），也要能被机器拼进
LLM 的 system prompt。本模块负责：

- `load_skills()`：加载 `skills/*.md`（按文件名排序，带 mtime 缓存）；
- `skill_titles()` / `skill_index()`：给文档与 CLI 用的索引（完整正文仍在磁盘）；
- `select_skill_ids(kind)`：按诊断类型挑选本轮相关技能（控制 token 成本）；
- `build_system_prompt()`：把"角色 + 输出协议 + 工具清单 + 技能正文 +
  安全红线"拼成系统提示词，供 `LLMDriver` 注入 provider。

token 策略
----------
完整技能库约 1.6 万字符（≈7k tokens），每步都发送成本高。LLMDriver 默认：
1) 只注入"总则/门禁/安全红线 + 与本轮诊断相关的 1~2 篇"；
2) 每篇注入前 `per_skill_max_chars` 字符（文首即结论/阈值，摘要足够决策，
   完整正文供人评审与确定性驱动使用）；
3) 工具清单用 `ToolRegistry.brief_specs()` 的紧凑格式而非完整 JSON Schema。
实测 system prompt ≈ 1.2 万字符（≈5k tokens），配合 `max_steps` 与
`budget.tokens_total` 双重约束，离线 mock 全程可复现。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_DIR = ROOT / "skills"

# 完整正文注入上限（人类阅读/确定性驱动参考用；LLM prompt 另有截断）
MAX_SKILL_CHARS = 24000
MAX_TOTAL_CHARS = 48000
# LLM system prompt 默认每篇技能注入字符数 / 总上限
PROMPT_PER_SKILL_CHARS = 1600
PROMPT_MAX_TOTAL_CHARS = 16000

# 任何场景都必须注入的技能（安全红线优先级最高）
ALWAYS_SKILL_IDS = ("00", "04", "06")

# 诊断类型 → 追加技能（前缀匹配文件名）
_KIND_SKILLS: dict[str, tuple[str, ...]] = {
    "benign": ("01",),
    "repair": ("05", "01"),
    "covariate_affine": ("01", "02"),
    "condition_switch": ("02", "03"),
    "concept": ("02", "03"),
    "novel_class": ("02",),
    "label_shift": ("02",),
    "noise_inflate": ("01", "02"),
}
_DEFAULT_KIND_SKILLS = ("01", "02")

_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.+)$", re.MULTILINE)

_CACHE: dict[tuple[str, float], dict[str, str]] = {}


def skills_dir(dirs: Optional[str | Path] = None) -> Path:
    if dirs is None:
        return DEFAULT_SKILLS_DIR
    path = Path(dirs)
    return path if path.is_absolute() else (ROOT / path)


def load_skills(dirs: Optional[str | Path] = None,
                max_chars_per_skill: int = MAX_SKILL_CHARS) -> dict[str, str]:
    """加载技能：`{文件名(不含.md): 正文}`，按文件名升序。"""
    base = skills_dir(dirs)
    if not base.exists():
        return {}
    files = sorted(p for p in base.glob("*.md") if p.is_file())
    mtime = max((p.stat().st_mtime for p in files), default=0.0)
    key = (str(base), mtime)
    cached = _CACHE.get(key)
    if cached is not None:
        return dict(cached)
    out: dict[str, str] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if len(text) > max_chars_per_skill:
            text = text[:max_chars_per_skill] + "\n\n…（技能正文超长，已截断）"
        out[path.stem] = text
    _CACHE.clear()
    _CACHE[key] = dict(out)
    return out


def skill_titles(dirs: Optional[str | Path] = None) -> dict[str, str]:
    """技能 id → 标题（从 `# ` 一级标题提取）。"""
    titles: dict[str, str] = {}
    for name, text in load_skills(dirs).items():
        match = _TITLE_RE.search(text)
        titles[name] = match.group(1).strip() if match else name
    return titles


def skill_index(dirs: Optional[str | Path] = None) -> list[dict]:
    """技能索引（供 CLI/docs 展示）。"""
    titles = skill_titles(dirs)
    out = []
    for name, text in load_skills(dirs).items():
        quote = _BLOCKQUOTE_RE.search(text)
        out.append({"id": name, "title": titles.get(name, name),
                    "summary": quote.group(1).strip() if quote else "",
                    "chars": len(text)})
    return out


def select_skill_ids(kind: str = "", dirs: Optional[str | Path] = None) -> list[str]:
    """按诊断类型选择本轮要注入的技能前缀（token 成本可控）。"""
    ids = list(ALWAYS_SKILL_IDS)
    extra = _KIND_SKILLS.get(str(kind or ""), _DEFAULT_KIND_SKILLS)
    for prefix in extra:
        if prefix not in ids:
            ids.append(prefix)
    available = set(load_skills(dirs))
    return [prefix for prefix in ids
            if any(name.startswith(prefix) for name in available)]


def skills_by_ids(ids: Optional[list[str]] = None,
                  dirs: Optional[str | Path] = None,
                  per_skill_max_chars: Optional[int] = None) -> dict[str, str]:
    """按 id（文件名前缀）选择技能，可逐篇截断。"""
    all_skills = load_skills(dirs)
    if not ids:
        return all_skills
    out: dict[str, str] = {}
    for wanted in ids:
        for name, text in all_skills.items():
            if name.startswith(str(wanted)) and name not in out:
                if per_skill_max_chars and len(text) > per_skill_max_chars:
                    text = text[:per_skill_max_chars] + \
                        "\n\n…（技能摘要注入，完整正文见 skills/" + name + ".md）"
                out[name] = text
                break
    return out


_PROTOCOL = """\
## 输出协议（必须严格遵守）
你每一步只能输出**一个 JSON 对象**，不要输出 Markdown 代码块或其它文字：
1) 调用工具：{"thought": "为什么调这个工具", "tool": "<工具名>", "args": {...}}
2) 给出最终决策：{"thought": "...", "final": {"chosen_playbook": "<候选 playbook 或 NOOP>",
   "action": "noop|run_experiment|deploy|rollback|shadow", "reasoning": "一句话依据",
   "deploy": false, "version": null, "confidence": 0.0~1.0, "outcome": {...}}}
规则：
- `chosen_playbook` 必须来自本轮 candidates 的 playbook，或 "NOOP"；否则决策非法；
- 只读工具可随意调用；写动作（run_experiment/deploy/rollback/request_labels）每步一个，
  预算/白名单/门禁由工具侧强制校验，被拒是正常反馈（读 error 后换策略或收手）；
- 你不可以自己批准 high-risk 动作，也不可以绕过门禁上线（上线必须通过 deploy 工具）；
- 工具返回 `ok=false` 时不要重复同一失败调用；最多重试一次不同参数。
"""


def build_system_prompt(tool_specs: Optional[list[dict]] = None,
                        skills: Optional[dict[str, str]] = None,
                        dirs: Optional[str | Path] = None,
                        extra_rules: str = "",
                        max_total_chars: int = PROMPT_MAX_TOTAL_CHARS,
                        tool_briefs: Optional[list[str]] = None,
                        per_skill_max_chars: int = PROMPT_PER_SKILL_CHARS) -> str:
    """拼装 LLM 驱动的 system prompt：角色 + 协议 + 工具清单 + 技能正文。"""
    if skills is not None:
        selected = dict(skills)
        if per_skill_max_chars:
            selected = {
                name: (text if len(text) <= per_skill_max_chars else
                       text[:per_skill_max_chars] +
                       f"\n…（摘要注入，完整正文见 skills/{name}.md）")
                for name, text in selected.items()}
    else:
        selected = skills_by_ids(select_skill_ids("", dirs), dirs,
                                 per_skill_max_chars=per_skill_max_chars)
    titles = skill_titles(dirs) if skills is None else {
        name: (_TITLE_RE.search(text).group(1).strip()
               if _TITLE_RE.search(text) else name)
        for name, text in selected.items()}
    header = (
        "你是「智训引擎」的工业模型自进化决策智能体（Harness 驱动层）。\n"
        "你的职责：在**零网络、预算硬约束、可审计**的前提下，根据漂移监测与诊断结果，\n"
        "从候选 playbook 中选择并执行一步实验，评估后决定上线/影子/收手，并保证所有\n"
        "决策有证据、有成本记录、可回滚。\n"
        "决策原则：安全 > 成本 > 增益 > 速度；无证据不动作；一次只改一个变量。\n"
    )
    parts = [header, _PROTOCOL]
    if extra_rules:
        parts.append("\n## 本轮附加规则\n" + extra_rules.strip() + "\n")
    if tool_briefs:
        parts.append("\n## 可用工具（按需调用；schema 由工具侧强制校验）\n")
        parts.extend(line + "\n" for line in tool_briefs)
    elif tool_specs:
        parts.append("\n## 可用工具（JSON Schema）\n```json\n" +
                     json.dumps(tool_specs, ensure_ascii=False, indent=1) +
                     "\n```\n")
    parts.append("\n## 技能库（必须遵守，编号 06 安全红线优先于其它技能）\n")
    total = len("".join(parts))
    for name, text in selected.items():
        chunk = f"\n### 技能 {name}：{titles.get(name, name)}\n{text}\n"
        if total + len(chunk) > max_total_chars:
            parts.append("\n…（其余技能因 token 上限未注入，按已注入规则执行）\n")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


__all__ = [
    "DEFAULT_SKILLS_DIR", "ALWAYS_SKILL_IDS", "load_skills", "skill_titles",
    "skill_index", "skills_by_ids", "select_skill_ids", "build_system_prompt",
]

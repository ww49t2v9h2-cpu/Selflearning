# 07 · 智能体 Harness 与工具技能

> 负责人：C（Harness/观测）。对应代码：`src/harness/tools.py`、`src/harness/agent.py`、
> `src/harness/skills.py`、`src/providers/**`、`skills/*.md`。
> 契约：`docs/03_接口契约与目录结构.md` §5.1–§5.3（本文不改变冻结接口，只补充实现细节）。

---

## 0. 一句话总结

Harness 是"工业模型自进化系统"的**决策执行层**：它把"看漂移 → 归因 → 选动作 → 做实验 →
过门禁 → 上线/回滚"的每一步，都变成**有 schema 校验、有权限与预算硬门、有耗时/成本记账、
可完整回放的工具调用**。默认由 `DeterministicDriver` 离线驱动（零网络、零 token、可复现），
也可一行配置切换到 **PyTRi0 Qwen3.8-27B** 等 OpenAI 兼容大模型驱动（`LLMDriver`），
后者在无额度/无网络/解析失败时自动降级到确定性策略——**评测现场没有 LLM 额度也能跑满闭环**。

实测（本机，Python 3.12 / numpy 2.3.5 / 无 swanlab）：

| 指标 | 结果 |
|---|---|
| 工具数 | 12（5 只读 + 1 只读评估 + 2 预算型 + 4 写） |
| 单元测试 | `tests.test_tracker` 28 条 + `tests.test_harness` 56 条，全部通过 |
| 确定性驱动单轮 | 5~6 次工具调用、0 token、~5ms（不含引擎训练） |
| LLM 驱动（mock）单轮 | 7 次补全 / 6 次工具调用，42,841 tokens，完整协议闭环 |
| LLM 失败降级 | 连续 2 次非法 JSON → 自动回退确定性，`driver="llm-fallback"` |
| 记录写入耗时 | 单条 `log_metrics` 0.26 ms（预算 <5 ms） |

---

## 1. 架构与数据流

```
                       ┌────────────────────────── src/loop.py（Lead）──────────────────────────┐
                       │ 构造 snapshot：t/phase/monitor_report/diagnosis/candidates/budget/...  │
                       └───────────────────────────────┬──────────────────────────────────────────┘
                                                       ▼
        ┌──────────────────────────── src/harness/agent.py ───────────────────────────┐
        │  HarnessDriver 协议：decide(snapshot, tools) -> Trajectory                  │
        │   ├─ DeterministicDriver（默认）  规则 = skills/*.md 的可执行子集             │
        │   └─ LLMDriver（可插拔）          skills → system prompt，JSON 工具调用协议    │
        └───────────────────────────────┬──────────────────────────────────────────────┘
                                        ▼
        ┌──────────────────────────── src/harness/tools.py ───────────────────────────┐
        │ ToolRegistry：schema 校验 → 权限/白名单/kill-switch → handler → 异常捕获      │
        │               → 计时 → Cost 记账 → ToolCall（可 JSON 序列化）                │
        │ 12 工具：inspect_drift / diagnose / budget_status / propose_candidates /    │
        │ run_experiment / evaluate_candidate / deploy / rollback / request_labels /  │
        │ list_registry / query_history / shadow_eval                                 │
        └───────┬──────────────────────────────┬──────────────────────────────┬───────┘
                ▼                              ▼                              ▼
        ToolContext（依赖注入）        src/engine/**（B）              src/observability
        oracle/diagnoser/planner/      monitor/diagnosis/playbooks/    tracker：全量记录
        library/registry/safety/       planner/gates/registry/safety    + SwanLab 兼容
        gate/tracker/state                                                
```

关键设计：**工具不直接 import 引擎**，只依赖 `ToolContext` 协议。好处：
1. 引擎未就绪时可用 fake 完整单测（本交付的 56 条 harness 测试就是这么做的）；
2. loop 可以把同一套工具绑定到"真引擎"或"评测模式"，无需改 Harness；
3. 所有工具调用天然被记录为 `ToolCall`，直接构成 `Trajectory.steps`，满足策略层自改进与审计。

---

## 2. 工具清单与权限表

| # | 工具 | 类型 | risk | side_effects | 预算消耗 | 主要参数 | 返回要点 | 拒绝条件 |
|---|---|---|---|---|---|---|---|---|
| 1 | `inspect_drift` | 只读 | low | none | 0 | include_events, max_events | events/state/level/max_severity/thresholds | 无监测报告、参数非法 |
| 2 | `diagnose` | 只读+预算 | low | budget | 少量 labels（诊断探针） | force, phase | Diagnosis（kind/severity/confidence/recommended）+ cost | 诊断器未绑定、上下文缺失、过预算 |
| 3 | `budget_status` | 只读 | low | none | 0 | phase | labels/compute/tokens/updates 余额 + 建议 | 未绑定 BudgetState |
| 4 | `propose_candidates` | 只读 | low | none | 0 | max_candidates, phase | 候选 playbook + est_cost + prior + risk + affordable | 无诊断 |
| 5 | `run_experiment` | 写 | medium | budget | compute/labels（labels 由 oracle 扣） | playbook, params, hypothesis | exp_id/run_id/version/metrics/cost/gate 前状态 | 白名单外、预算不足、phase 次数用满、kill-switch、安全否决 |
| 6 | `evaluate_candidate` | 评估 | medium | none | 0（可选探针） | exp_id, version | gate（criteria/passed/decision）+ probe_gain | 候选不存在、门禁无法计算 |
| 7 | `deploy` | 写 | high | model | 0 | version, reason, force | version/previous_version/gate/safety/escalation | 门禁未过、kill-switch、高危未审批/未记录 escalation |
| 8 | `rollback` | 写 | high | model | 0 | to_version, reason, emergency | from/to version + safety | 目标版本不存在、已是 champion；kill-switch 下仅 emergency 放行 |
| 9 | `request_labels` | 写 | low | budget | labels | n, ids, reason, phase | ids/y/n/cost/labels_left | 超预算、oracle 未绑定、ids 为空 |
| 10 | `list_registry` | 只读 | low | none | 0 | include_archived | champion/versions/metrics/artifact | registry 历史读取失败 |
| 11 | `query_history` | 只读 | low | none | 0 | filters, limit | 历史 run 摘要（可接 tracker/SQLite） | 过滤键非法 |
| 12 | `shadow_eval` | 只读评估 | low | none | 0（可选探针） | exp_id, version | agreement/entropy_delta/verdict | 候选不存在 |

约定：
- **只读工具**（side_effects=none）可在 `ToolPolicy(read_only=True)` 模式单独使用，用于分析/复盘；
- **预算型**（budget）在只读模式下被拒绝——只读分析不允许消耗标注预算；
- **写工具**（model）在 kill-switch 生效时全部拒绝（`rollback` 的 `emergency=true` 除外，
  因为回滚本身是安全恢复动作）；
- `risk=high` 的工具默认需要审批：校赛阶段按 `safety.escalation_mode=record_and_proceed`
  以 **escalation 记录**代替人工审批（写入 `ctx.state["escalations"]` 与 `Decision.safety`），
  也可配置为 `block` 强制拒绝；**LLM 永远不能自我批准**（批准只来自 `ctx.approvals`）。

### 2.1 JSON Schema 校验（轻量但足够）

每个工具带 `input_schema`，`ToolRegistry.call()` 在 handler 前做校验（子集实现）：
`type`（object/array/string/integer/number/boolean/null）、`required`、`properties`、
`enum`、`minimum/maximum`、`items`、`additionalProperties:false`、`default` 填充。
非法参数返回 `ToolCall(ok=False, result={"error_kind":"invalid_args","validation_errors":[...]})`，
LLM 驱动会把错误反馈给模型重试（最多 2 次），避免"带病执行"。

### 2.2 异常与记账

- handler 内**可预期失败**抛 `ToolError(code, message, result)`；其它任何异常也被捕获为
  `error_kind=exception`，**绝不打断闭环**；
- 每次调用记录 `latency_s`；工具把资源消耗写进 `result["cost"]`，
  `ToolCall.cost` 汇总后进入 `Decision.cost`；
- 预算记账防重复：**labels 由 `LabelOracle` 扣**、compute/tokens 由 Harness 补扣，
  通过调用前后 `labels_used` 差值只补扣未被 oracle 覆盖的部分；
- 每次实验/评估/上线/回滚都会向 tracker 写独立 `RunRecord`（含 parent_run 关联 loop run），
  实验失败也写 `status="failed"`，满足"全部实验完整记录"。

---

## 3. ToolContext：与 loop 的集成契约

`ToolContext`（`src/harness/tools.py`）是协议，`DefaultToolContext` 是默认实现：

```python
from src.harness.tools import build_tool_context, build_tool_registry, ToolPolicy

ctx = build_tool_context(
    cfg, budget=budget,             # BudgetState（必须与 oracle 共用同一对象！）
    state=loop_state,               # 实时字典：工具会写 last_experiment/gate_results 等
    oracle=oracle, diagnoser=diagnoser, planner=planner, library=library,
    registry=registry, safety=safety, tracker=tracker, gate=gate,
    evaluate_model=optional_callable,
)
tools = build_tool_registry(ctx, cfg=cfg)          # 12 工具
# 可选：只读/受限模式
tools = build_tool_registry(ctx, cfg=cfg, policy=ToolPolicy(read_only=True))
```

`loop_state` 约定（loop 每轮更新）：

```python
{
  "t": int, "phase": int,
  "chunk": StreamChunk, "model": MLP, "base_x", "base_y", "history", "rng",
  "n_classes": int, "monitor_report": MonitorReport, "diagnosis": Diagnosis,
  "candidates": [...], "registry_summary": {...}, "champion_metrics": {...},
  "champion_model": MLP, "champion_version": int,
  "run_id": "<loop 主 run_id>", "group": "main",
}
```

- 写作用域：loop 只读 `state` 中引擎对象；工具会写 `experiments/last_experiment/gate_results/
  escalations/shadow_results/denied_calls/candidates/diagnosis` 等**结果键**，loop 可读回；
- `ctx.snapshot()` 供 driver 使用；`ctx.current()` 返回实时对象字典；
- 若 loop 希望自己构造引擎的 `PlaybookContext`，设置
  `ctx.build_playbook_context = lambda name, params: PlaybookContext(...)` 即可覆盖默认构造；
- 工具注册表可随时替换 `policy`（例如评测阶段切 `read_only=True` 做纯分析）。

---

## 4. 安全与受控设计（对齐"安全可控"评分项）

| 机制 | 实现位置 | 说明 |
|---|---|---|
| 动作白名单 | `run_experiment` / `deploy` | 只允许 `cfg.safety.allowed_playbooks` |
| 工具白名单 | `ToolPolicy.allowed_tools` | loop 可按阶段只开放部分工具 |
| 只读模式 | `ToolPolicy.read_only` | 纯分析/评审阶段不可产生任何模型/预算副作用 |
| 预算硬门 | `run_experiment`/`request_labels` | 估算成本 > 余额直接拒绝，无"先跑后补" |
| phase 更新上限 | `budget_status` + run_experiment | `per_phase_updates=3` 用满即拒绝 |
| kill-switch | `SafetyGuard.should_kill()` 或 `STOP` 文件 | 写动作全拒；`rollback(emergency=True)` 例外 |
| 高危 escalation | `run_experiment`/`deploy` | 高危 playbook 记录 escalation（校赛以记录代替人工） |
| LLM 不可自批 | `ToolContext.approvals` | LLM 输出的 `force/approve` 不被信任 |
| LLM 不可绕过门禁 | `deploy` handler | 必须存在 `gate.passed=True`（或人工 force） |
| 异常隔离 | `ToolRegistry.call` | 工具异常转为失败 ToolCall，不打断闭环 |
| 审计 | tracker + `Trajectory.steps` + `Decision` | 每次调用参数/结果/成本/耗时全部留痕 |

补充：`force=true` 只对 loop/人工开放的 `deploy` 参数有效；LLM 驱动 prompt 明确要求
"不可以使用 force"，并且 `LLMDriver` 对最终决策只做记录、**真正上线必须由模型显式调用
`deploy` 工具**——若模型只在 final 里写 `deploy: true` 而没调工具，`Decision.outcome`
会标记 `deploy_not_executed=True` 且不产生任何上线副作用（见测试
`test_llm_cannot_deploy_without_tool`）。

---

## 5. 技能库（skills/*.md）

7 篇中文技能文档，既作为**评审材料**（策略可解释、可审计），又是 **LLM system prompt 的
规则来源**，其阈值与 `DeterministicDriver` 规则同源：

| 文件 | 技能 | 关键内容 |
|---|---|---|
| `00_技能索引与总则.md` | 总则 | 优先级（安全>成本>增益>速度）、机器可执行阈值 JSON、输出协议 |
| `01_漂移分诊.md` | 漂移分诊 | 无/弱/中/强四档阈值、必查前提（warmup/连续触发/回滚版本）、4 个反例、禁忌 |
| `02_Playbook选择矩阵.md` | 动作选择 | 8 种诊断 kind → playbook 映射、cheapest-first/先验优先规则、风险表、反例 |
| `03_实验设计与预算.md` | 实验设计与预算 | 实验四件套、220 labels/220 compute 分配、停止条件、单位增益公式、反例 |
| `04_上线门禁与影子评估.md` | 上线门禁 | 6 项门禁判据（0.005/0.12/40/0.03…）、promote/shadow/reject、反例 |
| `05_回滚SOP.md` | 回滚 SOP | 5 个触发条件、6 步标准动作、kill-switch 规则、反抖动约束、反例 |
| `06_安全红线与成本纪律.md` | 安全与成本 | 六条红线、成本纪律、工具权限矩阵、审计链路要求 |

注入策略（控制 token）：`LLMDriver` 每次只注入
`00 总则 + 04 门禁 + 06 安全红线 + 按诊断 kind 的相关 1~2 篇`，每篇截断到约 1600 字符
（文首即阈值/结论，完整正文供人评审）；工具清单用 `ToolRegistry.brief_specs()`
紧凑格式（比完整 JSON Schema 省约 70% token）。实测 system prompt ≈ 11k 字符 ≈ **5k tokens**。

---

## 6. DeterministicDriver：默认驱动

规则来自 skills 的可执行子集，**完全通过工具执行**并留下完整轨迹：

```
1. budget_status（只读）→ 预算/phase 次数
2. 若 kill-switch → NOOP（写 rationale）
3. 若 phase 更新次数用满 → NOOP
4. 若 candidates 为空 → NOOP（"无候选：无漂移或规划器为空"）
5. 若 diagnosis.confidence < 0.40 → NOOP（归因不明不动作）
6. 候选排序（可负担过滤）：
     benign  → NOOP 优先
     repair  → ROLLBACK 优先
     其它    → severity ≥ 0.55 时先验优先，否则 cheapest-first
7. 取第一个候选：
     NOOP      → 收手
     ROLLBACK  → rollback 工具（回滚 SOP）
     SHADOW_*  → shadow_eval
     常规 playbook → run_experiment
8. run_experiment 成功 → evaluate_candidate：
     门禁 passed → deploy；否则 shadow_eval + 不上线
9. 任一候选被拒 → 试下一个；全部失败 → NOOP（失败证据全部留痕）
10. 组装 Decision（chosen_playbook 必来自 candidates 或 NOOP，
    rationale/budget_before/after/cost/outcome/evidence_refs/safety/driver 全填）
```

保证：`Trajectory` 字段完整；不依赖 LLM；不消耗 token；同一 snapshot 输出稳定可复现；
任何内部异常都会转成 NOOP + 异常信息，**driver 永不把异常抛给 loop**。

---

## 7. LLMDriver：可插拔大模型驱动

### 7.1 协议

```
system : 角色 + 输出协议 + 工具 brief + 技能库（中文规则）
user   : 【状态快照】{t,phase,monitor_report,diagnosis,candidates,budget,registry_summary,...}
model  : 每一步只输出一个 JSON 对象，二选一
         A) 调工具 {"thought":"...","tool":"<name>","args":{...}}
         B) 收 尾 {"thought":"...","final":{"chosen_playbook":"...","action":"...",
                    "reasoning":"...","deploy":false,"version":null,"confidence":0.8}}
tool   : 【工具结果】{"tool":..., "ok":..., "result":..., "error":..., "error_kind":...}
```

### 7.2 严格性与降级

- JSON 解析：容忍 ``` 围栏与前后杂字，提取第一个**平衡** JSON 对象；
- 校验：顶层必须是对象；`tool` 必为已注册工具（未知工具算解析失败）；
  `args` 必须过工具 JSON Schema；`final.chosen_playbook` 必须 ∈ candidates ∪ {NOOP}；
- 重试：解析/校验失败把错误作为 user 消息反馈，连续 `max_parse_failures=2` 次失败 → 降级；
- 降级：`DeterministicDriver.decide()` 接管，合并两边 steps，`Trajectory.driver="llm-fallback"`，
  `outcome.fallback_reason` 记录具体原因（json_parse_failed / unknown_tool /
  invalid_chosen_playbook / provider_unavailable / max_steps_reached / single_call_token_cap…）；
- token：`Trajectory.tokens` 与 `Decision.cost.tokens` 同步；`safety.max_tokens_per_decision`
  作为**单次 LLM 调用**上限，整轮/整个 run 由 `budget.tokens_total` 硬约束；
  **记账责任在 loop**（`src/loop.py` 每轮统一 `budget.charge(Cost(tokens=...))`，
  避免 driver 与 loop 重复计数）；独立使用 Harness（不经 loop）时可传
  `LLMDriver(..., charge_budget_tokens=True)` 让 driver 自行记账。

### 7.3 实测：mock provider 的完整一轮（测试输出节选）

```
LLM prompt: 11054 chars ≈ 4967 tokens（skills 选择: 00/04/06/01/02）
llm steps: inspect_drift✔ → diagnose✔ → budget_status✔ → run_experiment✔
           → evaluate_candidate✔ → deploy✔（7 次补全，6 次工具调用）
trajectory.driver = "llm"
trajectory.tokens = 42841
final_decision.chosen_playbook = "RECALIBRATE_AFFINE"
final_decision.outcome = {"action":"deploy","deployed":true,"version":1,...}
budget.tokens_used = 42841
```

> 说明：mock 的 prompt/completion token 按字符估算（CJK 1 字≈1 token，其它 4 字符≈1 token），
> 与真实 Qwen BPE 偏差通常 ±15%；真实 provider 优先使用服务端 `usage` 字段。

---

## 8. 接线 PyTRi0 Qwen3.8-27B（官方评测环境）

赛题要求"驱动 Harness 的大模型须为 PyTRi0 的 Qwen3.8-27B 或选手在 PyTRi0 训练的模型"。
PyTRi0 推理服务兼容 OpenAI Chat Completions，因此**只换环境变量、不改代码**：

```bash
# 1) 环境变量（按优先级取第一个非空；密钥只从环境读，绝不写日志/记录）
export ZHIXUN_LLM_BASE_URL="https://<pytri0-endpoint>/v1"     # 以官方下发为准
export ZHIXUN_LLM_API_KEY="<your-key>"
export ZHIXUN_LLM_MODEL="Qwen3.8-27B"
# 可选：export ZHIXUN_LLM_TIMEOUT=60

# 2) 配置（configs/llm.json，差异覆盖）
{
  "harness": {"driver": "llm", "provider": "openai_compat",
              "model": "Qwen3.8-27B", "max_steps": 8, "temperature": 0.2}
}

# 3) 运行闭环
$PY -m src.cli run --config configs/llm.json

# 4) 校验接线是否生效（不跑闭环，只看 provider 状态）
$PY - <<'EOF'
from src.core.config import load_config
from src.harness.agent import make_provider
provider, meta = make_provider(load_config(), "openai_compat")
print(meta)   # enabled=True, model=Qwen3.8-27B, api_key_set=True（不含密钥）
EOF
```

实现细节（`src/providers/openai_compat.py`）：
- `enabled = bool(base_url and model)`；**未配置时 `complete()` 立即抛
  `ProviderUnavailable`（0 网络请求，<1ms）**，LLMDriver 当场降级确定性；
- 请求体 `{model, messages, temperature, max_tokens, stream:false}`；
  URL 可写 `.../v1` 或 `.../v1/chat/completions`；响应兼容 `choices[0].message.content`
  与 `usage.prompt_tokens/completion_tokens`；
- 失败快速失败（默认 `max_retries=0`，4xx 不重试、5xx 才允许重试），不阻塞闭环；
- `health()` 只报告 base_url（脱敏）/model/api_key_set/timeout，**绝不输出密钥**。

### 8.1 token 成本估算（给预算规划用）

单轮决策 token ≈ `system prompt (~5k) + snapshot (~1k) + 历史工具结果` × 步数。
以实测 mock 的 7 步为例约 **43k tokens/决策**；若 40 次更新全部用 LLM，约
`40 × 43k ≈ 172 万 tokens`，明显超过 `budget.tokens_total=120000`。
因此建议（评测现场按额度调整 `configs/llm.json`）：

| 场景 | 建议配置 |
|---|---|
| 无额度/无网络（校赛默认） | `driver=deterministic`（0 token） |
| 少量额度演示 | `driver=llm` + `safety.max_tokens_per_decision=40000` + `budget.tokens_total=500000`，`max_steps=6` |
| 全量 LLM 跑 | 提高 `tokens_total` 到 ~200 万，或只在"诊断+决策"关键窗口启用（其余确定性） |
| 混合模式（推荐） | 确定性驱动跑全量闭环；LLM 只做 `auto` 模式下的关键决策解释/影子复核 |

> 注意：token 预算是**硬约束**，用满后 LLMDriver 自动降级确定性策略（不会超支），
> 这正是"安全可控"的体现。

---

## 9. 离线降级链（评测现场可靠性）

```
driver=llm, provider=openai_compat
   ├─ 未配置环境变量 / 无网络 / 4xx/5xx / 超时     → ProviderUnavailable/Error
   ├─ 响应非法 JSON / 未知工具 / 非法 chosen        → 连续 2 次失败
   ├─ 达到 max_steps 或单次调用 token 超上限        → fallback
   └──────────────────────┬────────────────────────
                          ▼
              DeterministicDriver（离线、零 token、规则=skills）
                          │
                          ▼
              Trajectory.driver="llm-fallback"
              outcome.fallback_reason / Decision.safety 记录原因
```

`build_driver(cfg, tools=None) -> HarnessDriver` 是 loop 唯一入口：
- `driver=deterministic`（默认）→ `DeterministicDriver`；
- `driver=llm` + `provider=mock` → `LLMDriver(MockLLMProvider)`（离线也能验证完整 LLM 协议）；
- `driver=llm` + `provider=openai_compat` → 配置齐全则真 LLM，否则降级确定性；
- `driver=auto` → 只有真 LLM 可用才走 LLM，否则确定性（避免"以为在用大模型、其实在跑 mock"）。

返回对象带 `.provider_info`（driver/provider/model/network/enabled/fallback_reason）与
`.skill_ids`，loop 可直接打印进运行日志。

---

## 10. 测试与证据

```bash
PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
$PY -m unittest tests.test_harness -v     # 56 tests OK
$PY -m unittest tests.test_tracker -v     # 28 tests OK
$PY -m unittest discover -s tests -v      # 全套 OK
$PY scripts/smoke.py                      # [OK] 冒烟验证通过
```

覆盖点（对应评分要求）：

| 测试类 | 覆盖 |
|---|---|
| `TestToolRegistry` | 12 工具注册、schema 校验、未知工具、只读/白名单/次数/高风险策略 |
| `TestReadTools` | inspect/diagnose（缓存+force+预算）/budget/propose（planner+skills 兜底）/registry/history |
| `TestWriteTools` | 预算拒绝、白名单、kill-switch、safety veto、高危 escalation/审批、失败记录、evaluate/deploy/reject、request_labels、shadow_eval、异常捕获 |
| `TestDeterministicDriver` | 完整轨迹/Decision 字段、无候选 NOOP、预算耗尽 NOOP、benign/repair 规则、回滚、kill-switch、坏上下文不崩、可复现 |
| `TestLLMDriver` | mock 完整协议、非法 JSON/未知工具/非法 chosen 降级、provider 失败降级、max_steps、token 计量、不可绕过 deploy 工具 |
| `TestProviders` | token 估算、mock 确定性、openai_compat 未配置快速失败、不可达端点快速失败、响应解析 |
| `TestSkillsAndFactories` | 技能库≥6、阈值存在、prompt 拼装、build_driver 三种模式、env 接线、harness_turn |

---

## 10.1 与 loop 的真实集成验证（证据）

在 Lead 的 `src/loop.py` 闭环上实跑（`python -m src.cli run --config configs/quick.json --group quick --seed 0`）：

- Harness 路径生效：11 个触发窗口产生 11 条 `Decision`，`safety.tool_steps` 完整记录
  `budget_status → inspect_drift → … → run_experiment → evaluate_candidate → shadow_eval` 的
  每次调用（ok/耗时/成本/错误）；
- 真实记录落库：experiment/deploy 等 RunRecord 均由工具写入 tracker，`parent_run` 指向 loop 主 run；
- 真实上线验证：在 `debugmain`/`enginetest` 等含部署的 run 中，`deploy` run 的 summary
  包含引擎 Gate 的完整 criteria（`probe_gain/calibration/cost/…`）与 escalation 记录，
  证明"工具 → 引擎门禁 → 模型仓库"链路真实跑通；
- 安全闭环：quick run 中未过门禁的候选被正确转为 shadow（不上线），
  repair/label_shift 在预算不足时正确 NOOP——与 skills/02、04、06 的规则一致。

> 注：quick 配置预算较小（labels 120/compute 120），多数标签型 playbook 被预算硬门拦下，
> 因此部署数偏少是**策略+预算的预期结果**；正式评分以 `configs/main.json`（更大预算）全量 run 为准。

---

## 11. 复现命令

```bash
PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
# 离线确定性驱动（默认）
$PY -m src.cli run --config configs/default.json
# LLM 驱动（mock provider，离线）
$PY -m src.cli run --config configs/llm_mock.json
# LLM 驱动（PyTRi0 Qwen3.8-27B）
$PY -m src.cli run --config configs/llm.json
# 单测 + 冒烟
$PY -m unittest discover -s tests -v && $PY scripts/smoke.py
```

---

## 12. 已知限制与后续

1. **LLM token 预算与质量权衡**：默认 20k/次调用 + 120k 总量对全量 LLM 跑偏紧，
   评测时按额度调整配置（docs/12 §配置项有说明）；
2. **SwanLab 基础版本无稳定 artifact API**：模型文件以本地 `experiments/artifacts/{run_id}/`
   为权威，SwanLab 记录指标/config/summary 与 run 映射；文件名/哈希写入记录可追溯；
3. **工具结果压缩**：LLM 看到的工具结果做过 `_shrink`（列表截断 8 项、字符串 400 字符），
   极大数组不会撑爆上下文，但极端场景模型可能看不到完整事件列表（可让模型再查一次）；
4. **多轮工具协议**：当前一次决策最多 `max_steps` 次补全，不支持模型自主并行调用多工具
   （串行更利于审计与预算控制，符合赛题要求）。

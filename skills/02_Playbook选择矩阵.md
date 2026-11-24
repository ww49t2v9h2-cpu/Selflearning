# 02 · Playbook 选择矩阵（Playbook Matrix）

> 输入：`diagnose` 输出的 `kind`、`severity`、`confidence`、`affected_*`、`recommended`；
> 输出：本轮要跑的 **一个** playbook（必须来自 `candidates`）。
> 工具：`propose_candidates` → `run_experiment`。

## 1. 诊断类型 → 首选动作

| `kind` | 首选 playbook | 备选 | 判定依据 / 为什么 |
|---|---|---|---|
| `benign` | `NOOP` | — | 分布变化不影响决策，干预只会引入风险 |
| `repair` | `ROLLBACK` | `NOOP` | 工况修好、分布回基线：应回到基线版本，继续训练会遗忘 |
| `covariate_affine` | `RECALIBRATE_AFFINE` | `LAST_LAYER_FT` | 传感器增益/缩放漂移，无标签仿射可逆；labels=0 |
| `condition_switch` | `LAST_LAYER_FT` | `FULL_FINETUNE_REPLAY` | 工况切换是特征关系重排，少量标签校准头部即可 |
| `concept` | `FULL_FINETUNE_REPLAY` | `RETRAIN_MIXED` | 决策边界变了，需要带标签+回放防止遗忘 |
| `novel_class` | `NOVEL_CLASS_EXTEND` | `SHADOW_ONLY` | 必须扩输出层；用仿射/头校准会系统性误判为新旧类的混合 |
| `label_shift` | `PRIOR_CORRECTION` | `RECALIBRATE_AFFINE` | 类别先验变了但类条件分布没变，重加权即可；labels=0 |
| `noise_inflate` | `RECALIBRATE_AFFINE` | `NOOP` | 噪声膨胀先校准统计量；若校准后门禁不过则收手 |
| 低置信度（<0.40） | `NOOP` | `SHADOW_ONLY` | 归因不明时不做不可逆动作 |

## 2. 选择顺序（确定性规则，`DeterministicDriver` 同源）

1. **过滤**：只保留 `cfg.safety.allowed_playbooks` 内、且 `est_cost` 在预算内的候选；
2. **排序**：默认 cheapest-first（`Cost.scalar` 升序）；当
   `severity ≥ severity_high(0.55)` 时改为"先验优先"（`prior` 降序，再成本升序）；
3. **并列**：先验相同 → 选风险更低者；风险相同 → 选 `applicable_kinds` 命中者；
4. **上限**：一次决策最多执行 1 个候选实验（`max_candidates=3` 只影响展示）。

## 3. 风险与审批

| playbook | 层 | 风险 | 需要标签 | 上线前要求 |
|---|---|---|---|---|
| `NOOP` | model | low | 0 | — |
| `RECALIBRATE_AFFINE` | model | low | 0 | 门禁（probe 不降即可） |
| `PRIOR_CORRECTION` | model | low | 0 | 门禁 |
| `LAST_LAYER_FT` | model | low | 少 | 门禁 |
| `FULL_FINETUNE_REPLAY` | model | medium | 中 | 门禁 + 回滚版本存在 |
| `RETRAIN_MIXED` | model | medium | 多 | 门禁 + 回滚版本存在 |
| `NOVEL_CLASS_EXTEND` | model | **high** | 多 | 门禁 + 类别扩展验证 + escalation 记录 |
| `SHADOW_ONLY` | strategy | low | 0 | 不上线，仅记录 |
| `ROLLBACK` | strategy | medium | 0 | 直接执行（恢复动作） |

> `high_risk_playbooks = ["NOVEL_CLASS_EXTEND", "RETRAIN_MIXED"]`（`DEFAULT_CONFIG`）。
> 高危动作必须走门禁并在 `Decision.safety` 里记录 escalation；校赛阶段以
> "escalation 记录代替人工审批"，但仍受门禁与 kill-switch 约束。

## 4. 反例

- ❌ **novel_class 用 RECALIBRATE_AFFINE**：只会把新类样本更自信地判错，probe F1 反而下降；
  正确：`NOVEL_CLASS_EXTEND`，或先 `SHADOW_ONLY` 收集证据。
- ❌ **benign 也跑 FULL_FINETUNE_REPLAY**：无增益还消耗 compute/labels，且引入遗忘风险；
  门禁会拒绝，但预算已浪费——先分诊再动作。
- ❌ **高严重度仍 cheapest-first 到 NOOP**：候选若含 `NOVEL_CLASS_EXTEND` 且先验 0.8，
  应给高风险动作一次机会，但必须先确认标签预算足够（≥ 该 playbook 估算的 2 倍余量）。
- ❌ **同一窗口连跑 2 个 playbook**：违反"一次只改一个变量"，结果无法归因。

## 5. 输出格式（`Decision.alternatives` 记录被放弃的候选及原因）

```json
{"chosen": "RECALIBRATE_AFFINE",
 "alternatives": [{"playbook": "LAST_LAYER_FT", "why_not": "需要 40 labels，预算余量仅 18"}],
 "rule": "cheapest-first: cost 0.0 < 0.21，且 affine_recoverability=0.72>0.6"}
```

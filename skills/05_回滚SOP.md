# 05 · 回滚 SOP（Rollback Standard Operating Procedure）

> 目标：上线后变差时，**在 1~3 个窗口内**恢复到上一稳定版本，并把过程完整留痕。
> 工具：`list_registry` → `rollback`（写动作）；触发源：`Watchdog` + `inspect_drift`。
> 依据：`gates.rollback_window=3`、`gates.rollback_drop_points=0.03`、
> `safety.kill_switch_file="STOP"`。

## 1. 触发条件（满足任一）

1. **看门狗触发**：上线后连续 `rollback_window(3)` 个窗口，相对上线时参考性能
   下降 > `rollback_drop_points(0.03)`；
2. **漂移分诊转 repair**：流分布回归基线（`repair`），当前 champion 是在漂移期
   训练的版本 → 回滚到基线/parent 版本通常优于继续保持；
3. **门禁后被证伪**：上线 1~2 窗口内 probe 掉点 > `min_gain_points` 且显著
   （`class_recall_min` 同步下滑）；
4. **人工/安全事件**：kill-switch 文件被触发、异常连续失败 ≥3 次；
5. **依赖失效**：候选 artifact 不可加载/不可复现（运行期检测到）。

## 2. 标准动作（顺序执行，逐步记录）

1. **冻结**：本轮不再启动新的 `run_experiment`（`should_kill` 时全部写动作停）；
2. **确认目标版本**：`list_registry` 找上一稳定版本（通常为 champion 的
   `parent`，或最近一个 `status=champion` 的历史版本）；没有则报
   `rollback_unavailable`，转 NOOP 并升级；
3. **执行**：`rollback(to_version, reason, emergency=...)`，返回 `ModelVersion`；
4. **校验**：回滚后用同一探针窗口做一次评估，确认恢复 ≥ 回滚前；
5. **记录**：写 `RunRecord(kind="rollback", stage="rollback")`，summary 含
   `from_version/to_version/drop/reason/trigger`；
6. **复盘**：`Decision.outcome.rollback` 记录触发证据；在下一轮策略先验中降低
   触发 playbook 的权重（策略层自改进）。

## 3. Kill-Switch 特别规则

- 文件 `STOP`（`cfg.safety.kill_switch_file`）存在时：`run_experiment` / `deploy` /
  `request_labels` 全部拒绝；`rollback` **允许**（恢复动作本身是安全路径），
  但只允许 `emergency=true` 的显式回滚；
- kill-switch 解除后不要立刻恢复重型动作：先 NOOP 观察 1 个窗口，确认指标稳定。

## 4. 反例

- ❌ **连续回滚抖动**：版本 A→B→A→B 反复横跳。原因通常是"回滚后又立刻自动上新"；
  规则：同一 phase 内回滚次数 ≤ 1，回滚后至少 2 窗口 `SHADOW_ONLY`/NOOP；
- ❌ **回滚到不存在的版本**：先 `list_registry` 确认；回滚目标必须是历史
  `champion/archived` 且 artifact 可加载；
- ❌ **只回滚模型不回滚策略**：若问题是数据/工况导致，回滚模型不解决根因；
  必须同时检查 `inspect_drift`，必要时诊断归因；
- ❌ **回滚不记录成本/原因**：回滚同样消耗决策与记录成本，必须写 summary；
  审计链缺失会导致"可追溯"评分项失分。

## 5. 记录格式

```json
{"trigger": "watchdog:3窗口掉点0.041>0.03",
 "from_version": 7, "to_version": 5,
 "before": {"probe_macro_f1": 0.712}, "after": {"probe_macro_f1": 0.768},
 "steps": [{"tool": "list_registry", "ok": true},
           {"tool": "rollback", "ok": true}],
 "note": "禁用触发候选的 playbook 先验，等待 repair 稳定后重新实验"}
```

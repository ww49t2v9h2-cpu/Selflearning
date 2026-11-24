"""引擎层配置兜底（ENGINE_DEFAULTS）与配置解析。

背景
----
`src/core/config.py::DEFAULT_CONFIG` 归 Lead 维护。引擎需要的全部阈值/系数在这里
给出**可直接运行的默认值**，并在 `resolve_config()` 中按下面的优先级合并：

    1. cfg[section]（Lead 已写入 DEFAULT_CONFIG 的正式配置，优先级最高）
    2. cfg["engine"][section]（Lead 按本文件提案放入 engine 命名空间的形式）
    3. ENGINE_DEFAULTS[section]（本文件兜底）

这样即使 Lead 尚未合并新增配置项，引擎也能独立运行；一旦合并，配置覆盖立即生效。
所有引擎模块的 `__init__` 都必须先 `self.cfg = resolve_config(cfg)`，代码里不得出现
魔法数字——阈值一律从解析后的配置读取（任务约束：阈值/系数写进配置）。
"""

from __future__ import annotations

import copy
from typing import Any, Optional

# 引擎默认配置。语义见 docs/04-06；与 DEFAULT_CONFIG 重名的键以 DEFAULT_CONFIG 为准。
ENGINE_DEFAULTS: dict = {
    "monitor": {
        # ---- 参考窗口 ----
        "warmup_chunks": 4,          # 用前 N 个 chunk 建立参考窗口（若无显式 reset_reference）
        "window": 5,                 # 滚动窗口最大 chunk 数（参考/稳定判定共用）
        "ref_max_chunks": 5,         # 参考窗口最多保留的 chunk 数
        "ref_max_samples": 800,      # 参考窗口最多保留的样本数（MMD/KS 的子采样上限）
        "ref_patience": 3,           # 连续 N 个窗口与滚动窗口相似 → 认为进入新稳定域，重设 active 参考
        "event_cooldown": 3,         # 同一轮漂移只触发一次：事件后 N 窗口内只更新统计量不重复发事件
        "repair_patience": 1,        # repair 信号连续窗口数
        "score_ema": 0.5,            # （见 watchdog）探针分数平滑系数
        "stability_ratio": 0.5,      # MMD(chunk, rolling) < ratio*mmd_threshold 判为“稳定新域”
        "repair_ratio": 0.5,         # MMD(chunk, baseline) < ratio*mmd_threshold 判为“回到基线”
        # ---- 无标签检测器 ----
        "mmd_threshold": 0.35,       # DEFAULT_CONFIG 中的占位阈值（保留兼容）
        "mmd_threshold_calibrated": 0.02,  # 实测校准阈值（MMD^2 量级见 docs/04 §3）
        "mmd_subsample": 400,        # MMD 子采样上限（控制 O(n^2) 成本）
        "ks_alpha": 0.001,           # 逐特征 KS 显著性水平
        "ks_topk": 12,               # 记录 top-k 显著特征的个数
        "ks_count_threshold": 6,     # 至少 N 个特征显著才发 ks_topk 事件
        "ks_stat_threshold": 0.10,   # 同时要求 top-k 平均 KS 统计量 ≥ 该值（防大样本微小差异）
        "entropy_z_threshold": 3.0,  # 预测熵 z 分数阈值
        "margin_z_threshold": 3.0,   # 预测边际下降 z 分数阈值
        "pred_hist_tv_threshold": 0.15,  # 预测类别直方图 TV 距离阈值（标签先验漂移代理）
        "novelty_k": 8,              # kNN novelty 的 k
        "novelty_threshold": 0.6,    # novelty 统计量阈值（见 docs/04 定义）
        "novelty_subsample": 300,    # novelty 参考子采样上限
        "ph_alpha": 0.02,            # Page-Hinkley 均值 EMA 遗忘因子
        "ph_delta": 0.002,           # Page-Hinkley 容忍漂移
        "ph_lambda": 0.06,           # Page-Hinkley 告警阈值
        "ph_threshold": 6.0,         # 归一化 PH 统计量事件阈值
        # ---- 标注探针（预算内） ----
        "probe_every": 2,            # 每 N 个窗口做一次准确率探针
        "probe_labels": 6,           # 单次探针标签数
        "probe_min_labels": 3,       # 单次探针至少需要多少标签（控制探针 4 条）
        "adapt_reserve": 80,         # 为"适应（playbook）"预留的标签绝对下限
        "adapt_reserve_frac": 0.60,   # 预留 = max(绝对下限, frac*labels_total)
        "probe_control_every": 3,    # 无异常时每 N 窗口做一次"控制探针"（≤4 条）
        "probe_max_labels": 4,       # 单次探针标签上限（预算优先留给 playbook）
        "anchor_labels": 4,          # 参考域锚定所需的最小探针标签数（双门不变量）
        "min_gap": 1,                # 事件触发探针的最小窗口间隔
        # ---- 其它 ----
        "severity_scale": 1.0,
        "rolling_samples_per_chunk": 200,  # 滚动窗口每 chunk 保留样本数
    },
    "diagnosis": {
        "topk_features": 12,             # 与 DEFAULT_CONFIG 对齐
        "severity_high": 0.55,
        "affine_recoverability": 0.6,    # 仿射校准可恢复比阈值（DEFAULT_CONFIG 已有）
        "probe_labels": 20,              # 诊断用标签上限（复用 monitor 已购标签后追加）
        "probe_spend_min_drop": 0.03,    # 探针准确率下降到该值才买诊断标签
        "corr_change_thresh": 0.10,      # 相关系数结构变化阈值（工况旋转实测校准）
        "mean_shift_thresh": 0.15,       # 逐特征均值漂移比阈值
        "affine_residual_tol": 0.08,     # 仿射校准后残差容忍度
        "min_probe_labels": 10,          # 少于此数退化为无标签诊断
        "label_reserve": 40,             # 给 playbook 预留的标签，低于该校验不再买诊断标签
        "affine_min_gain": 0.03,         # 仿射增益下限（绝对准确率点）
        "affine_min_drop": 0.015,        # 静态准确率至少下降这么多才谈“可恢复”
        "novelty_ratio_thresh": 0.35,    # novelty 事件阈值的放松版（证据用）
        "novel_label_frac": 0.02,        # 探针中新标签占比达到该值 → novel 证据
        "error_conc_thresh": 0.5,        # novelty 误差集中度阈值
        "label_shift_tv": 0.15,          # 估计先验 TV 距离阈值
        "noise_std_ratio": 1.25,         # 噪声膨胀特征 std 比阈值
        "noise_ks_frac": 0.5,            # 噪声膨胀时显著特征占比要求
        "repair_drop_ratio": 0.5,        # 回到基线的 MMD 比例阈值
        "repair_acc_recover": 0.03,      # repair 时探针准确率相对近期最低值的恢复量
        "recommended_top_k": 4,          # recommended 最多给几个 playbook
        # 诊断类型 → playbook 先验（可被 bandit 策略表修正）。核心排序原则：
        # 先试"零标签、可跨窗口泛化的 TTA 归一化"，再考虑带标签动作；
        # 只有当 TTA 无法解释漂移（novel/label_shift）才直接上专用动作。
        "priors": {
            "benign": {"NOOP": 0.70, "RECALIBRATE_AFFINE": 0.20, "SHADOW_ONLY": 0.10},
            "covariate_affine": {"RECALIBRATE_AFFINE": 0.70, "LAST_LAYER_FT": 0.12,
                                 "FULL_FINETUNE_REPLAY": 0.08, "NOOP": 0.10},
            "condition_switch": {"RECALIBRATE_AFFINE": 0.50, "LAST_LAYER_FT": 0.22,
                                 "FULL_FINETUNE_REPLAY": 0.16, "RETRAIN_MIXED": 0.12},
            "concept": {"RECALIBRATE_AFFINE": 0.40, "LAST_LAYER_FT": 0.24,
                        "FULL_FINETUNE_REPLAY": 0.22, "NOVEL_CLASS_EXTEND": 0.14},
            "novel_class": {"NOVEL_CLASS_EXTEND": 0.62, "RECALIBRATE_AFFINE": 0.18,
                            "FULL_FINETUNE_REPLAY": 0.12, "NOOP": 0.08},
            "label_shift": {"PRIOR_CORRECTION": 0.48, "NOOP": 0.32,
                            "RECALIBRATE_AFFINE": 0.20},
            "noise_inflate": {"RECALIBRATE_AFFINE": 0.66, "NOOP": 0.24,
                              "FULL_FINETUNE_REPLAY": 0.10},
            "repair": {"NOOP": 0.48, "ROLLBACK": 0.30, "RECALIBRATE_AFFINE": 0.22},
        },
    },
    "playbooks": {
        # labels=训练标签预算（会按剩余预算裁剪），labels_eval=候选评估留出标签（不与训练重叠）
        "reserve_labels": 8,             # 为后续窗口预留（低到可以让廉价动作在末尾仍可行）
        "eval_frac": 0.25,               # eval 标签占总标签比例（下限 labels_eval）
        "RECALIBRATE_AFFINE": {"labels_eval": 16, "blend": 1.0, "pool_chunks": 1},
        "PRIOR_CORRECTION": {"labels_eval": 20, "method": "em", "strength": 1.0,
                             "posterior_temp": 1.0},
        "LAST_LAYER_FT": {"labels": 72, "labels_eval": 20, "min_train": 32,
                          "max_labels": 110, "epochs": 80, "lr": 0.01, "batch": 64,
                          "class_weight_power": 1.0},
        "FULL_FINETUNE_REPLAY": {"labels": 110, "labels_eval": 24, "min_train": 48,
                                 "max_labels": 150, "epochs": 25, "lr": 0.002,
                                 "batch": 128, "replay_n": 900, "refit_scaler": False,
                                 "class_weight_power": 0.5, "replay_new_ratio": 0.0},
        "RETRAIN_MIXED": {"labels": 120, "labels_eval": 24, "min_train": 56,
                          "max_labels": 150, "epochs": 35, "lr": 0.0035, "batch": 128,
                          "replay_n": 1600, "class_weight_power": 0.5,
                          "refit_scaler": True},
        "NOVEL_CLASS_EXTEND": {"labels": 96, "labels_eval": 24, "min_train": 56,
                               "max_labels": 150, "epochs": 30, "lr": 0.0025,
                               "batch": 128, "replay_n": 1000, "novel_weight": 2.0,
                               "class_weight_power": 0.5, "refit_scaler": True},
        "SHADOW_ONLY": {"sub_playbook": "FULL_FINETUNE_REPLAY"},
        "NOOP": {},
        "ROLLBACK": {"labels_eval": 16, "epochs": 20, "lr": 0.004, "batch": 128},
        # 安全/规划辅助
        "high_risk": ["NOVEL_CLASS_EXTEND", "RETRAIN_MIXED"],
    },
    "gates": {
        "min_gain_points": 0.01,
        "min_probe_gain": 0.005,
        "max_recall_drop": 0.12,
        "max_cost_per_update": 40.0,
        "calibration_slack": 0.04,
        "min_class_recall_floor": 0.0,
        "min_class_recall_drop_abs": 0.0,
        "high_risk_gain_multiplier": 1.5,
        "shadow_required": False,
        "shadow_windows": 2,
        "repro_tol": 1e-6,
        "min_labels_for_verdict": 12,
        "rollback_window": 3,
        "rollback_drop_points": 0.03,
    },
    "safety": {
        "allowed_playbooks": ["NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION",
                              "LAST_LAYER_FT", "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED",
                              "NOVEL_CLASS_EXTEND", "SHADOW_ONLY", "ROLLBACK"],
        "high_risk_playbooks": ["NOVEL_CLASS_EXTEND", "RETRAIN_MIXED"],
        "human_gate": True,
        "kill_switch_file": "STOP",
        "kill_switch_files": ["STOP", "experiments/STOP"],
        "max_consecutive_failures": 3,
        "max_tokens_per_decision": 20000,
        "max_decisions": 80,
        "max_labels_per_decision": 160,
        "max_updates_per_window": 1,
        "allow_network": False,
        "audit_dir": "experiments/audit",
    },
    "planner": {
        "mode": "bandit",
        "max_candidates": 3,
        "exploration": 0.25,
        "cheapest_first": True,
        "strategy_learning": True,
        "policy_path": "experiments/strategy_policy.json",
        "ucb_c": 0.35,
        "prior_weight": 0.5,
        "cost_penalty": 0.06,
        "reward_ema": 0.6,
        "prior_strength": 2.0,
        "always_keep_cheap": True,
    },
    "engine": {
        "registry": {"dir": "experiments/registry", "artifact_subdir": "versions",
                     "keep_last": 40},
        "watchdog": {"rollback_window": 3, "rollback_drop_points": 0.03,
                     "recover_margin": 0.005, "ref_ema": 0.8, "score_ema": 0.5,
                     "min_drop_abs": 0.12},
    },
}

_SECTIONS = ("monitor", "diagnosis", "playbooks", "gates", "safety", "planner",
             "registry", "watchdog")


def _deep_merge(base: dict, override: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(cfg: Optional[dict]) -> dict:
    """把 Lead 的 cfg + ENGINE_DEFAULTS 解析成引擎内部使用的完整配置。

    返回的是**新 dict**，不修改入参；保留 cfg 中其它 section（budget/model/
    benchmark/cost_model/...）的原始内容。
    """
    cfg = cfg or {}
    engine_cfg = cfg.get("engine") if isinstance(cfg.get("engine"), dict) else {}
    out: dict = {}
    for sec in _SECTIONS:
        if sec in ("registry", "watchdog"):
            base = ENGINE_DEFAULTS["engine"].get(sec, {})
        else:
            base = ENGINE_DEFAULTS.get(sec, {})
        merged = copy.deepcopy(base)
        for src in (cfg.get(sec), engine_cfg.get(sec)):
            merged = _deep_merge(merged, src if isinstance(src, dict) else None)
        out[sec] = merged
    for k, v in cfg.items():
        if k == "engine" or k in _SECTIONS:
            continue
        out[k] = copy.deepcopy(v)
    # 兼容：engine 命名空间仍然可读
    out["engine"] = {"registry": out["registry"], "watchdog": out["watchdog"]}
    # watchdog 与 gates 中既有的滚动判定参数对齐（DEFAULT_CONFIG 把它们放在 gates 下）
    gates = out.get("gates", {})
    for key in ("rollback_window", "rollback_drop_points"):
        if key in gates:
            out["watchdog"][key] = gates[key]
    # 若 Lead 未提供 budget/cost_model/model，给出最小兜底（正常来自 DEFAULT_CONFIG）
    out.setdefault("budget", {})
    out.setdefault("cost_model", {"base_compute_unit": 4000.0,
                                  "compute_weight": 1.0, "label_weight": 0.02,
                                  "token_weight": 1e-5})
    out.setdefault("seed", 0)
    return out

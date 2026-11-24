"""配置加载与运行时上下文。"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"


DEFAULT_CONFIG: dict = {
    "seed": 7,
    "benchmark": {
        "name": "synth_bearing",
        "adapter": "synth",              # synth|official|ai4i
        "official_root": "data/official",
        "ai4i_root": "data/external",
        "scenario": "S7_full_mix",
        "chunk_size": 200,
        "hidden_ratio": 0.34,
        "base_train_n": 4000,
        "base_val_n": 1000,
        "n_classes": 4,
        "d_features": 48,
        "phase_n": 1800,
        "noise_scale": 3.4,
        "rot_alpha": 0.45,
        "offset_scale": 0.15,
    },
    "model": {
        "hidden": [64, 32],
        "activation": "relu",
        "epochs": 40,
        "lr": 0.004,
        "batch": 128,
        "l2": 1e-4,
        "seed": 0,
    },
    "budget": {
        "labels_total": 220,
        "compute_total": 220.0,
        "tokens_total": 120000,
        "updates_total": 40,
        "per_phase_updates": 3,
        "probe_labels": 6,
    },
    "monitor": {
        "window": 5,
        "mmd_threshold": 0.35,
        "ks_alpha": 0.001,
        "proxy_drop_threshold": 0.055,
        "probe_every": 2,
        "min_gap": 1,
        "novelty_threshold": 0.6,
        "warmup_chunks": 4,
    },
    "diagnosis": {
        "topk_features": 12,
        "severity_high": 0.55,
        "affine_recoverability": 0.6,
    },
    "gates": {
        "min_gain_points": 0.01,
        "min_probe_gain": 0.005,
        "max_recall_drop": 0.12,
        "max_cost_per_update": 40.0,
        "shadow_required": False,
        "rollback_window": 3,
        "rollback_drop_points": 0.03,
    },
    "safety": {
        "allowed_playbooks": [
            "NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION", "LAST_LAYER_FT",
            "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED", "NOVEL_CLASS_EXTEND",
            "SHADOW_ONLY", "ROLLBACK",
        ],
        "high_risk_playbooks": ["NOVEL_CLASS_EXTEND", "RETRAIN_MIXED"],
        "human_gate": True,
        "auto_approve_high_risk": True,
        "kill_switch_file": "STOP",
        "max_consecutive_failures": 3,
        "max_tokens_per_decision": 20000,
        "allow_network": False,
        "audit_dir": "experiments/audit",
    },
    "planner": {
        "mode": "bandit",          # bandit|grid|random
        "max_candidates": 3,
        "exploration": 0.25,
        "cheapest_first": True,
        "strategy_learning": True,
    },
    "harness": {
        "driver": "deterministic",  # deterministic|llm|auto
        "provider": "mock",         # mock|openai_compat
        "model": "mock-trainer-v1",
        "max_steps": 8,
        "temperature": 0.2,
        "record_trajectories": True,
    },
    "tracker": {
        "backend": "local",         # local|swanlab|both|none
        "swanlab_mode": "offline",  # online|local|offline|disabled
        "project": "zhixun-engine",
        "dir": "experiments/runs",
    },
    "cost_model": {
        "compute_weight": 1.0,
        "label_weight": 0.02,
        "token_weight": 1e-5,
        "base_compute_unit": 4000.0,
    },
    "run": {
        "name": "main",
        "n_seeds": 3,
        "save_artifacts": True,
        "deterministic": True,
    },
}



# ---------------------------------------------------------------------------
# 引擎（src/engine）细粒度配置：由 engine 队友提供，Lead 合并进 DEFAULT_CONFIG。
# 组件内部用 .get 兜底，但统一在此声明便于 configs/*.json 覆盖。
# ---------------------------------------------------------------------------
DEFAULT_CONFIG.update({
    "ablation": {
        "trigger_mode": "events",   # events|always|never
        "diagnosis": True,
        "planner": "bandit",        # bandit|random
        "gate": True,
        "strategy_learning": True,
        "watchdog": True,
    },
    "playbooks": {
        "RECALIBRATE_AFFINE": {"labels_eval": 16, "blend": 1.0, "pool_chunks": 1},
        "PRIOR_CORRECTION": {"labels_eval": 20, "method": "em", "strength": 1.0,
                             "posterior_temp": 1.0},
        "LAST_LAYER_FT": {"labels": 72, "labels_eval": 20, "min_train": 32,
                          "max_labels": 110, "epochs": 80, "lr": 0.01, "batch": 64,
                          "class_weight_power": 1.0, "oversample": 1.0},
        "FULL_FINETUNE_REPLAY": {"labels": 110, "labels_eval": 24, "min_train": 48,
                                 "max_labels": 150, "epochs": 25, "lr": 0.002,
                                 "batch": 128, "replay_n": 900, "refit_scaler": False,
                                 "class_weight_power": 0.5},
        "RETRAIN_MIXED": {"labels": 120, "labels_eval": 24, "min_train": 56,
                          "max_labels": 150, "epochs": 35, "lr": 0.0035,
                          "batch": 128, "replay_n": 1600, "class_weight_power": 0.5},
        "NOVEL_CLASS_EXTEND": {"labels": 140, "labels_eval": 24, "min_train": 64,
                               "max_labels": 150, "epochs": 30, "lr": 0.0025,
                               "batch": 128, "replay_n": 1000, "novel_weight": 2.0,
                               "class_weight_power": 0.5},
        "SHADOW_ONLY": {"sub_playbook": "RECALIBRATE_AFFINE"},
        "NOOP": {}, "ROLLBACK": {},
    },
    "data": {
        "sanitize": True,          # 抗脏数据：NaN/Inf 中位数填充 + 分位数截断
        "clip_quantile": 0.9995,
        "fill": "median",
    },
    "engine": {
        "monitor": {"ref_max_chunks": 5, "ref_max_samples": 800, "ref_patience": 3,
                    "ks_topk": 12, "ks_count_threshold": 6, "entropy_z_threshold": 3.0,
                    "margin_z_threshold": 3.0, "novelty_k": 8, "novelty_threshold": 0.6,
                    "ph_alpha": 0.01, "ph_lambda": 0.1, "ph_threshold": 8.0,
                    "probe_every": 2, "probe_labels": 6, "probe_min_labels": 6},
        "diagnosis": {"probe_labels": 32, "affine_min_gain": 0.03,
                      "affine_recoverability": 0.6, "novelty_ratio_thresh": 0.35,
                      "label_shift_tv": 0.15, "noise_std_ratio": 1.25,
                      "error_concentration": 0.5, "repair_drop_ratio": 0.5,
                      "recommended_top_k": 4},
        "gates": {"calibration_slack": 0.04, "min_class_recall_drop_abs": 0.0,
                  "high_risk_gain_multiplier": 1.5, "shadow_windows": 2,
                  "repro_tol": 1e-6, "min_labels_for_verdict": 12},
        "safety": {"max_labels_per_decision": 160, "max_decisions": 80,
                   "max_updates_per_window": 1,
                   "kill_switch_files": ["STOP", "experiments/STOP"]},
        "planner": {"policy_path": "experiments/strategy_policy.json",
                    "policy_scope": "run", "ucb_c": 0.35,
                    "prior_weight": 0.5, "cost_penalty": 0.02, "reward_ema": 0.6,
                    "prior_strength": 2.0},
        "registry": {"dir": "experiments/registry", "artifact_subdir": "versions",
                     "keep_last": 40},
        "watchdog": {"rollback_window": 3, "rollback_drop_points": 0.03,
                     "recover_margin": 0.005, "ref_ema": 0.8},
    },
})


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None,
                overrides: dict | None = None) -> dict:
    """加载默认配置 + 可选 JSON 文件 + 命令行式覆盖。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        with open(p, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, json.load(f))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


@dataclass
class RunContext:
    """一次完整闭环 run 的运行时上下文。"""

    cfg: dict
    run_id: str
    seed: int = 0
    outdir: Path = field(default_factory=lambda: EXPERIMENTS_DIR / "runs")
    artifacts_dir: Path = field(default_factory=lambda: EXPERIMENTS_DIR / "artifacts")

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        d = self.outdir / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

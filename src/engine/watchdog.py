"""上线看门狗：champion 上线后的在线性能监控与自动回滚判定。

对应 docs/03 §4.8 契约：
    Watchdog(cfg).observe(t, phase, score, champion_version) -> dict
        {"action": "none"|"rollback"|"recover", "reason": str, "drop": float}
    Watchdog(cfg).reset()

判定规则（见 docs/06 §4）
------------------------
- 以 champion 上线后前若干窗口的**平滑分数**为参考（ref）；分数越大越好
  （loop 传探针准确率，无探针时传 mean_margin）；
- 当平滑分数相对 ref 的下降持续 `rollback_window` 个窗口超过
  `rollback_drop_points` → 返回 action="rollback"；
- champion 版本变化/部署成功/回滚后会调用 reset() 重新建参考；
- 分数回升到参考附近时返回 action="recover"（用于记录"自愈/恢复"事件）。
平滑是为了抑制 6 条探针标签带来的离散噪声（0.167 一档）导致误回滚。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .defaults import resolve_config


class Watchdog:
    def __init__(self, cfg: dict):
        self.cfg = resolve_config(cfg)
        w = self.cfg["watchdog"]
        self.rollback_window = int(w.get("rollback_window", 3))
        self.drop_points = float(w.get("rollback_drop_points", 0.03))
        self.recover_margin = float(w.get("recover_margin", 0.005))
        self.ref_ema = float(np.clip(w.get("ref_ema", 0.8), 0.0, 0.999))
        self.score_ema = float(np.clip(w.get("score_ema", 0.5), 0.01, 1.0))
        # 探针只有 4-6 条标签时准确率是 1/n 的离散台阶（±0.25），原始 0.03 阈值会被
        # 噪声击穿；用正下限提高鲁棒性（实测校准见 docs/06 §4）。
        self.min_drop_abs = float(w.get("min_drop_abs", 0.12))
        self._score_hist: list[float] = []
        self.last_champion_version: Optional[int] = None
        self.reset()

    def reset(self) -> None:
        self.ref: Optional[float] = None
        self.smooth: Optional[float] = None
        self.bad = 0
        self.prev_bad = 0
        self.rolled = False
        self.n_obs = 0
        self.last_action = "none"

    def observe(self, t: int, phase: int, score: float,
                champion_version: int) -> dict:
        try:
            score = float(score)
        except (TypeError, ValueError):
            return {"action": "none", "reason": "score 不可用", "drop": 0.0}
        if not np.isfinite(score):
            return {"action": "none", "reason": "score 非有限值", "drop": 0.0}
        self.n_obs += 1
        # champion 更替：重新建立参考，不做判定（避免拿旧模型基线冤枉新模型）
        if self.last_champion_version is None:
            self.last_champion_version = int(champion_version)
        elif int(champion_version) != int(self.last_champion_version):
            self.last_champion_version = int(champion_version)
            self.ref = None
            self.smooth = None
            self.bad = 0
            return {"action": "none", "reason": "champion 更替，重建参考", "drop": 0.0}
        if self.ref is None:
            self.ref = score
            self.smooth = score
            return {"action": "none", "reason": "建立参考性能", "drop": 0.0}
        self.smooth = (self.score_ema * score
                       + (1.0 - self.score_ema) * float(self.smooth))
        drop = float(self.ref - self.smooth)
        eff_drop = max(self.drop_points, self.min_drop_abs)
        if drop > eff_drop:
            self.bad += 1
            self.prev_bad = self.bad
            if self.bad >= self.rollback_window:
                reason = (f"连续 {self.bad} 窗口平滑性能下降 {drop:.3f} > "
                          f"{eff_drop:.3f}（ref={self.ref:.3f}→{self.smooth:.3f}）")
                self.bad = 0
                self.rolled = True
                self.ref = None    # 回滚后由 reset()/下一次 observe 重建参考
                self.smooth = None
                self.last_action = "rollback"
                return {"action": "rollback", "reason": reason, "drop": drop}
            self.last_action = "none"
            return {"action": "none",
                    "reason": f"性能下降告警 {self.bad}/{self.rollback_window}",
                    "drop": drop}
        recovered = self.prev_bad > 0
        self.bad = 0
        self.prev_bad = 0
        # 健康窗口缓慢更新参考（只向上/缓慢，避免把缓慢劣化当新常态）
        self.ref = float(self.ref_ema * self.ref
                         + (1.0 - self.ref_ema) * self.smooth)
        if recovered:
            self.last_action = "recover"
            return {"action": "recover",
                    "reason": f"性能恢复至参考附近 drop={drop:.3f}", "drop": drop}
        self.last_action = "none"
        return {"action": "none", "reason": "正常", "drop": drop}

    def state(self) -> dict:
        return {"ref": self.ref, "smooth": self.smooth, "bad": self.bad,
                "n_obs": self.n_obs, "last_action": self.last_action,
                "rollback_window": self.rollback_window,
                "rollback_drop_points": self.drop_points}

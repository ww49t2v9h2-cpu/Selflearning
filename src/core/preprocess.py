"""特征清洗（抗脏数据）：缺失值填充、异常值截断、不可用样本标记。

对应官方速览视频披露的决赛数据包-B：「更难、更复杂、可能混脏数据」，
以及企业赛"隐藏测试流 + 统一离线评测"下系统必须**不因脏数据崩溃/失真**。

设计原则
--------
1. 统计量只从**干净的初始训练集**（base_x）估计，避免目标域污染；
2. 对非有限值（NaN/Inf）用训练集中位数填充；
3. 对极端异常值按训练集分位数截断（默认 0.1%/99.9%）；
4. 统计每次 transform 的修复量，写入运行记录（可审计），不改变输入 shape；
5. 纯 numpy，无额外依赖；训练/推理一致（loop 与 evaluator 共用同一 sanitizer）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SanitizerStats:
    n_samples: int = 0
    n_nonfinite: int = 0
    n_clipped: int = 0
    n_features_touched: int = 0

    def to_dict(self) -> dict:
        return {"n_samples": self.n_samples, "n_nonfinite": self.n_nonfinite,
                "n_clipped": self.n_clipped, "n_features_touched": self.n_features_touched}


class RobustSanitizer:
    """中位数填充 + 分位数截断的特征清洗器。"""

    def __init__(self, clip_quantile: float = 0.9995, fill: str = "median"):
        self.clip_quantile = float(np.clip(clip_quantile, 0.5, 1.0))
        self.fill = str(fill)
        self.median: np.ndarray | None = None
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None
        self.stats = SanitizerStats()

    # ------------------------------------------------------------------
    def fit(self, x: np.ndarray) -> "RobustSanitizer":
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        # 用中位数预填后再算分位数（避免 NaN 影响）
        xf = np.where(np.isfinite(x), x, med[None, :])
        q_hi = self.clip_quantile
        q_lo = 1.0 - q_hi
        self.median = med
        self.lo = np.quantile(xf, q_lo, axis=0)
        self.hi = np.quantile(xf, q_hi, axis=0)
        # 退化列（lo==hi）放宽为 ±1
        same = np.abs(self.hi - self.lo) < 1e-12
        self.lo = np.where(same, self.median - 1.0, self.lo)
        self.hi = np.where(same, self.median + 1.0, self.hi)
        return self

    # ------------------------------------------------------------------
    def transform(self, x: np.ndarray, update_stats: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.median is None:
            self.fit(x)
        x = x.copy()
        finite = np.isfinite(x)
        n_bad = int((~finite).sum())
        if n_bad:
            x = np.where(finite, x, self.median[None, :])
        below = x < self.lo[None, :]
        above = x > self.hi[None, :]
        n_clip = int(below.sum() + above.sum())
        if n_clip:
            x = np.where(below, self.lo[None, :], x)
            x = np.where(above, self.hi[None, :], x)
        if update_stats:
            self.stats.n_samples += int(x.shape[0])
            self.stats.n_nonfinite += n_bad
            self.stats.n_clipped += n_clip
            touched = np.any(below | above, axis=0)
            self.stats.n_features_touched = int(np.sum(touched))
        return x

    # ------------------------------------------------------------------
    def reset_stats(self) -> None:
        self.stats = SanitizerStats()

    def state(self) -> dict:
        return {"clip_quantile": self.clip_quantile, "fill": self.fill,
                "median": self.median, "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_state(cls, st: dict) -> "RobustSanitizer":
        obj = cls(clip_quantile=st.get("clip_quantile", 0.9995), fill=st.get("fill", "median"))
        obj.median = np.asarray(st["median"], dtype=float)
        obj.lo = np.asarray(st["lo"], dtype=float)
        obj.hi = np.asarray(st["hi"], dtype=float)
        return obj

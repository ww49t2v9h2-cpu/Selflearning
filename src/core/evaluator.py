"""隐藏测试评测器（平台侧）。

系统（agent）永远不能访问隐藏标签：
- `Benchmark.hidden_by_chunk` 只在本模块被读取；
- `LabelOracle` 只暴露可见流的标签；
- `HiddenEvaluator` 只把**分数**（macro-F1/准确率/类召回）交给闭环用于报告，
  且闭环不得把隐藏分数用于任何决策（决策只能用无标签信号 + 探针标签）。

这是对官方"隐藏测试流不公开、平台自动算分"约束的仿真。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import evaluate
from .spec import HiddenScore


@dataclass
class StreamEvaluation:
    """一次 run 的隐藏测试评估结果集合。"""

    scores: list          # list[HiddenScore]
    phase_of_chunk: list
    phase_names: list
    n_classes: int

    # ---- 聚合 ----
    def curve(self, key: str = "macro_f1") -> list[float]:
        return [float(getattr(s, key)) for s in self.scores]

    def per_phase(self, key: str = "macro_f1") -> dict:
        out: dict[int, list[float]] = {}
        for s in self.scores:
            out.setdefault(s.phase, []).append(float(getattr(s, key)))
        return {p: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for p, v in sorted(out.items())}

    def phase_curve(self, key: str = "macro_f1") -> list[float]:
        pp = self.per_phase(key)
        return [pp[p]["mean"] for p in sorted(pp)]

    def area(self, key: str = "macro_f1") -> float:
        """流式性能面积（按窗口平均，等价于 trapezoid 的窗口均值）。"""
        c = np.asarray(self.curve(key), dtype=float)
        return float(np.mean(c)) if len(c) else 0.0

    def summary(self) -> dict:
        hf1 = self.curve("macro_f1")
        acc = self.curve("accuracy")
        return {
            "stream_macro_f1": float(np.mean(hf1)) if hf1 else 0.0,
            "stream_accuracy": float(np.mean(acc)) if acc else 0.0,
            "per_phase_macro_f1": {str(k): v["mean"] for k, v in self.per_phase().items()},
            "final_phase_macro_f1": float(self.phase_curve()[-1]) if self.scores else 0.0,
            "n_chunks": len(self.scores),
        }


class HiddenEvaluator:
    def __init__(self, benchmark, sanitizer=None):
        self.benchmark = benchmark
        self.n_classes = benchmark.n_classes
        # 抗脏数据：隐藏测试流同样先清洗（与系统看到的数据流一致）
        self.sanitizer = sanitizer

    def score_chunk(self, model, t: int) -> HiddenScore:
        hx, hy = self.benchmark.hidden_by_chunk[t]
        if self.sanitizer is not None and len(hx) > 0:
            hx = self.sanitizer.transform(hx, update_stats=False)
        if len(hy) == 0:
            return HiddenScore(t=t, phase=self.benchmark.chunks[t].phase,
                               macro_f1=float("nan"), accuracy=float("nan"),
                               per_class_recall={}, n=0)
        n_classes = max(self.n_classes, int(np.max(hy)) + 1)
        pred = model.predict(hx)
        m = evaluate(hy, pred, n_classes)
        return HiddenScore(
            t=t,
            phase=self.benchmark.chunks[t].phase,
            macro_f1=m["macro_f1"],
            accuracy=m["accuracy"],
            per_class_recall=m["per_class_recall"],
            n=m["n"],
        )

    def score_stream(self, model) -> StreamEvaluation:
        scores = [self.score_chunk(model, t) for t in range(self.benchmark.n_chunks)]
        return StreamEvaluation(scores=scores,
                                phase_of_chunk=list(self.benchmark.phase_of_chunk),
                                phase_names=list(self.benchmark.phase_names),
                                n_classes=self.n_classes)

    def score_stream_dynamic(self, models_by_t: dict) -> StreamEvaluation:
        """逐窗口模型可不同的评估（自进化系统的真实评测方式）。"""
        scores = []
        for t in range(self.benchmark.n_chunks):
            model = models_by_t.get(t)
            if model is None:
                raise KeyError(f"缺少窗口 {t} 的模型")
            scores.append(self.score_chunk(model, t))
        return StreamEvaluation(scores=scores,
                                phase_of_chunk=list(self.benchmark.phase_of_chunk),
                                phase_names=list(self.benchmark.phase_names),
                                n_classes=self.n_classes)


def compare_curves(method: StreamEvaluation, static: StreamEvaluation) -> dict:
    """自进化方法 vs 静态基线的逐窗口/逐阶段对比。"""
    m = np.asarray(method.curve(), dtype=float)
    s = np.asarray(static.curve(), dtype=float)
    d = m - s
    per_phase = {}
    for p in sorted(set(method.phase_of_chunk)):
        mask = np.array([pp == p for pp in method.phase_of_chunk])
        per_phase[str(p)] = float(np.mean(d[mask]))
    return {
        "stream_gain_points": float(np.mean(d)) * 100.0,
        "best_gain_points": float(np.max(d)) * 100.0,
        "worst_gain_points": float(np.min(d)) * 100.0,
        "per_phase_gain_points": {k: v * 100.0 for k, v in per_phase.items()},
        "win_windows": int(np.sum(d > 1e-9)),
        "lose_windows": int(np.sum(d < -1e-9)),
        "n_windows": int(len(d)),
    }

"""指标与统计量（纯 numpy 实现，无 scipy 依赖）。"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------
# 分类指标
# --------------------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out = {}
    for c in range(n_classes):
        mask = y_true == c
        out[int(c)] = float(np.mean(y_pred[mask] == c)) if mask.any() else float("nan")
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append(2 * tp / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "per_class_recall": {str(k): v for k, v in per_class_recall(y_true, y_pred, n_classes).items()},
        "n": int(len(y_true)),
    }


# --------------------------------------------------------------------------
# 置信度 / 校准 / 无标签代理指标
# --------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def marginal_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def mean_entropy(proba: np.ndarray) -> float:
    return float(np.mean(marginal_entropy(proba)))


def margin(proba: np.ndarray) -> np.ndarray:
    s = np.sort(proba, axis=1)
    return s[:, -1] - s[:, -2]


def mean_margin(proba: np.ndarray) -> float:
    return float(np.mean(margin(proba)))


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == np.asarray(y_true, dtype=np.int64)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def prediction_histogram(proba: np.ndarray, n_classes: int) -> np.ndarray:
    pred = np.argmax(proba, axis=1)
    return np.bincount(pred, minlength=n_classes).astype(float) / max(1, len(pred))


# --------------------------------------------------------------------------
# 两样本检验（纯 numpy）
# --------------------------------------------------------------------------
def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """双侧 Kolmogorov-Smirnov 统计量 + 渐近 p 值。"""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    if lam < 1e-8:
        return d, 1.0
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 400, seed: int = 0) -> float:
    """RBF MMD^2（无偏估计的简单版本，对样本量做子采样）。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    z = np.vstack([x, y])
    # 中位数启发式带宽
    sub = z[rng.choice(len(z), min(len(z), 300), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)
    kxx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    kyy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    kxy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1))
                 + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2.0 * kxy.mean())


def label_prior_em(proba: np.ndarray, n_classes: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """黑盒标签漂移估计（BBSE/EM）：估计目标域类别先验。

    返回 (prior_hat, p_hat_after_reweight)。p_hat 为把 predict_proba 按估计先验
    重新加权后的预测分布（可用于先验校正）。
    """
    proba = np.asarray(proba, dtype=float)
    n = len(proba)
    prior = np.ones(n_classes) / n_classes          # 源域先验（假定均衡）
    for _ in range(n_iter):
        # E 步：w_ic ∝ prior_t(c) * p_s(c|x_i) / prior_s(c)
        w = proba * prior[None, :]
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        # M 步
        new_prior = w.mean(axis=0)
        new_prior = new_prior / new_prior.sum()
        if np.max(np.abs(new_prior - prior)) < 1e-9:
            prior = new_prior
            break
        prior = new_prior
    # 先验重加权：p_t(c|x) ∝ p_s(c|x) * prior_t(c)/prior_s(c)
    reweighted = proba * (prior[None, :] * n_classes)
    reweighted = reweighted / np.clip(reweighted.sum(axis=1, keepdims=True), 1e-12, None)
    return prior, reweighted


# --------------------------------------------------------------------------
# 漂移严重度 / 恢复时间
# --------------------------------------------------------------------------
def recovery_time(curve: list[float], reference: float, window: int = 3,
                  tol: float = 0.02) -> int:
    """性能曲线跌到 reference 之下后，回到 reference-tol 所需窗口数。

    curve 为按时间顺序的 macro-F1 序列；reference 为部署前（相对静态）参考值。
    返回 -1 表示未跌或未恢复。
    """
    t_drop = -1
    for i, v in enumerate(curve):
        if v < reference - tol:
            t_drop = i
            break
    if t_drop < 0:
        return 0
    for j in range(t_drop, len(curve) - window + 1):
        if all(v >= reference - tol for v in curve[j:j + window]):
            return j - t_drop
    return -1


def area_under(curve: list[float]) -> float:
    """梯形积分（用于归一化性能保持率）。"""
    if not curve:
        return 0.0
    if len(curve) == 1:
        return float(curve[0])
    return float(np.trapezoid(np.asarray(curve, dtype=float)) / (len(curve) - 1))

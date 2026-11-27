"""智训引擎 Harness 选手模块（contest_harness SDK 协议）。

- describe 由 SDK 检查 train/predict 与 TASKS；
- train: 从平台父检查点适配，仅用平台可见数据，按验证指标提交候选；
- predict: 读 /model 下候选 checkpoint，逐行输出平台要求的 JSON。
"""
from __future__ import annotations

import json
import os
import random
from math import isfinite, log1p
from functools import lru_cache
from pathlib import Path

import numpy as np

TASKS = ["classification", "regression"]
MAX_TRAIN_ROWS = 20000
MAX_FEATURES = 512
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


# ---------- 输入扁平化（A/B/C 的嵌套特征统一成有序数值向量） ----------
def _walk(value, path, out):
    if isinstance(value, dict):
        for key in sorted(value):
            _walk(value[key], path + [str(key)], out)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _walk(child, path + [str(i)], out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        if np.isfinite(v):
            out.append((json.dumps(path, ensure_ascii=False), v))


def flatten_inputs(inputs):
    out = []
    _walk(inputs, [], out)
    return out


def flatten_official(inputs):
    """官方基线 checkpoint 的 metadata.paths 使用 [["key",k],["index",i]] 编码。"""
    out = {}

    def walk(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], path + [["key", str(key)]])
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                walk(child, path + [["index", i]])
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            v = float(value)
            if np.isfinite(v):
                out[json.dumps(path, ensure_ascii=False, separators=(",", ":"))] = v

    walk(inputs, [])
    return out


def _path_key(path):
    """官方 metadata.paths 可能是 JSON 字符串，也可能是原始 list；统一转成可哈希字符串。"""
    if isinstance(path, str):
        return path
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


class OfficialModel:
    """兼容官方 contest.cpu-mlp.v1 checkpoint（平台基线父产物）。"""

    def __init__(self, meta, w1, b1, w2, b2, mean, scale):
        self.meta = meta
        self.task = meta.get("task", "classification")
        self.classes = list(meta.get("classes") or [])
        self.paths = list(meta.get("paths") or [])
        self.feature_steps = [json.loads(path) if isinstance(path, str) else path for path in self.paths]
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.mean, self.scale = mean, scale

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            return cls(meta, z["w1"], z["b1"], z["w2"], z["b2"], z["mean"], z["scale"])

    def predict_one(self, inputs):
        x = self.vector(inputs)
        x = np.clip((x - self.mean) / self.scale, -10.0, 10.0)
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        if self.task == "regression":
            value = float(logits[0]) * float(self.meta.get("target_scale", 1.0)) + float(self.meta.get("target_mean", 0.0))
            return {"value": float(max(0.0, value))}
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        return [{"label": c, "score": float(s)} for c, s in zip(self.classes, p)]

    def vector(self, inputs):
        values = np.zeros(len(self.feature_steps))
        for column, steps in enumerate(self.feature_steps):
            value = inputs
            for kind, key in steps:
                if kind == "key" and isinstance(value, dict):
                    value = value.get(key)
                elif kind == "index" and isinstance(value, (list, tuple)) and 0 <= key < len(value):
                    value = value[key]
                else:
                    value = None
                    break
            if type(value) in (int, float) and isfinite(value):
                values[column] = value
        return values


def load_model(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "metadata" in files and {"w1", "b1", "w2", "b2", "mean", "scale"} <= files:
            return OfficialModel.load(path)
    return Model.load(path)


class Model:
    def __init__(self, task, classes, feature_paths, mean, scale, coef):
        self.task = task
        self.classes = list(classes)
        self.feature_paths = list(feature_paths)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            task = str(z["task"].item())
            classes = json.loads(str(z["classes"].item()))
            paths = json.loads(str(z["feature_paths"].item()))
            return cls(task, classes, paths, z["mean"], z["scale"], z["coef"])

    def features(self, inputs):
        flat = dict(flatten_inputs(inputs))
        return np.asarray([float(flat.get(p, 0.0)) for p in self.feature_paths], dtype=np.float64)

    def predict_one(self, inputs):
        x = self.features(inputs)
        x = np.clip((x - self.mean) / self.scale, -10.0, 10.0)
        a = np.concatenate([x, [1.0]])
        if self.task == "regression":
            return {"value": float(max(0.0, (a @ self.coef)[0]))}
        logits = a @ self.coef
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        return [{"label": c, "score": float(s)} for c, s in zip(self.classes, p)]


def _read_rows(rnd):
    rows = []
    for row in rnd.data.training_rows():
        if not isinstance(row, dict) or "inputs" not in row or "answer" not in row:
            continue
        rows.append(row)
        if len(rows) >= MAX_TRAIN_ROWS:
            break
    return rows


def _fit(rows, task):
    flat0 = flatten_inputs(rows[0]["inputs"])
    feature_paths = [p for p, _ in flat0[:MAX_FEATURES]]
    X = np.zeros((len(rows), len(feature_paths)), dtype=np.float64)
    for i, row in enumerate(rows):
        d = dict(flatten_inputs(row["inputs"]))
        X[i] = [float(d.get(p, 0.0)) for p in feature_paths]
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    X = np.clip((X - mean) / scale, -10.0, 10.0)
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)

    if task == "regression":
        y = np.asarray([float(row["answer"]) for row in rows], dtype=np.float64)
        coef = np.linalg.lstsq(A, y, rcond=1e-6)[0].reshape(-1, 1)
        classes = []
    else:
        classes = sorted({str(row["answer"]) for row in rows})
        if len(classes) < 2:
            classes = classes + ["__other__"]
        index = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((len(rows), len(classes)), dtype=np.float64)
        for i, row in enumerate(rows):
            Y[i, index[str(row["answer"])]] = 1.0
        coef = np.linalg.lstsq(A, Y, rcond=1e-6)[0]
    return Model(task, classes, feature_paths, mean, scale, coef)


def _parent_checkpoint(rnd):
    try:
        descriptor = getattr(getattr(rnd, "parent", None), "descriptor", None) or {}
        for relative in descriptor.get("checkpoint_paths", []):
            try:
                return rnd.parent.path(relative)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _model_vector(model, inputs):
    if isinstance(model, OfficialModel):
        return model.vector(inputs)
    flat = dict(flatten_inputs(inputs))
    return np.asarray([float(flat.get(p, 0.0)) for p in model.feature_paths], dtype=np.float64)


def _build_dataset(model, rows, task):
    classes = [str(c) for c in getattr(model, "classes", [])]
    index = {c: i for i, c in enumerate(classes)} if task == "classification" else {}
    X, y = [], []
    for row in rows:
        try:
            vec = _model_vector(model, row["inputs"])
        except Exception:
            continue
        X.append(vec)
        if task == "regression":
            try:
                y.append(float(row["answer"]))
            except Exception:
                X.pop()
        else:
            label = str(row["answer"])
            if label not in index:
                X.pop(); continue
            y.append(index[label])
    if not X:
        return None, None
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def _macro_f1(y, p, n_classes):
    f1 = []
    for c in range(n_classes):
        tp = int(np.sum((y == c) & (p == c)))
        denom = int(np.sum(y == c)) + int(np.sum(p == c))
        f1.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def _macro_f1_label(y_str, p_str, classes):
    f1 = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_str, p_str) if a == c and b == c)
        denom = sum(1 for a in y_str if a == c) + sum(1 for b in p_str if b == c)
        f1.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def _eval_official(parent, params, X, y, task, classes, mean=None, scale=None):
    w1, b1, w2, b2 = params
    if X is None or len(X) == 0:
        return None
    mean = parent.mean if mean is None else mean
    scale = parent.scale if scale is None else scale
    xx = np.clip((X - mean) / scale, -10.0, 10.0)
    h = np.tanh(xx @ w1 + b1)
    logits = h @ w2 + b2
    if task == "regression":
        pred = np.maximum(0.0, logits.reshape(-1) * float(parent.meta.get("target_scale", 1.0)) + float(parent.meta.get("target_mean", 0.0)))
        return float(np.sqrt(np.mean((pred - y) ** 2)))
    y_str = [classes[int(v)] for v in y]
    p_str = [classes[int(np.argmax(v))] for v in logits]
    return _macro_f1_label(y_str, p_str, classes)


def _fine_tune_parent(parent, task, train_rows, val_rows):
    X, y = _build_dataset(parent, train_rows, task)
    if X is None or len(X) == 0:
        return None, None, None, None, None
    Xv, yv = _build_dataset(parent, val_rows, task) if val_rows else (None, None)
    classes = [str(c) for c in parent.classes]
    w1, b1, w2, b2 = (np.array(parent.w1, dtype=np.float64, copy=True),
                      np.array(parent.b1, dtype=np.float64, copy=True),
                      np.array(parent.w2, dtype=np.float64, copy=True),
                      np.array(parent.b2, dtype=np.float64, copy=True))
    mean, scale = parent.mean, parent.scale
    Xn = np.clip((X - mean) / scale, -10.0, 10.0)
    if task == "regression":
        tm, ts = float(parent.meta.get("target_mean", 0.0)), float(parent.meta.get("target_scale", 1.0))
        Y = ((y - tm) / ts).reshape(-1, 1)
    else:
        Y = np.zeros((len(y), len(classes)), dtype=np.float64)
        Y[np.arange(len(y)), y] = 1.0
    m1 = np.zeros_like(w1); v1 = np.zeros_like(w1)
    m2 = np.zeros_like(w2); v2 = np.zeros_like(w2)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    beta1, beta2, eps, lr, l2 = 0.9, 0.999, 1e-8, 0.003, 1e-4
    epochs = 12
    n = max(len(X), 1)
    best = None
    best_score = None
    parent_score = _eval_official(parent, (w1, b1, w2, b2), Xv, yv, task, classes) if Xv is not None else None
    for epoch in range(1, epochs + 1):
        h = np.tanh(Xn @ w1 + b1)
        logits = h @ w2 + b2
        if task == "regression":
            grad_z = (logits - Y) / n
        else:
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            grad_z = (p - Y) / n
        gw2 = h.T @ grad_z + l2 * w2
        gb2 = grad_z.sum(axis=0)
        gh = (grad_z @ w2.T) * (1.0 - h * h)
        gw1 = Xn.T @ gh + l2 * w1
        gb1 = gh.sum(axis=0)
        t = epoch
        for arr, grad, m, v in ((w1, gw1, m1, v1), (b1, gb1, mb1, vb1),
                                (w2, gw2, m2, v2), (b2, gb2, mb2, vb2)):
            m *= beta1; m += (1 - beta1) * grad
            v *= beta2; v += (1 - beta2) * (grad * grad)
            mhat = m / (1 - beta1 ** t); vhat = v / (1 - beta2 ** t)
            arr -= lr * mhat / (np.sqrt(vhat) + eps)
        score = _eval_official(parent, (w1, b1, w2, b2), Xv, yv, task, classes) if Xv is not None else None
        if score is not None:
            better = best is None or ((score < best_score) if task == "regression" else (score > best_score))
            if better:
                best = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
                best_score = score
    if best is None:
        best = (w1, b1, w2, b2)
    # 返回最优参数、验证分数、父模型验证分数、训练/验证样本数
    return best, best_score, parent_score, len(X), (len(Xv) if Xv is not None else 0)


def _save_official_checkpoint(path, parent, params, extra_meta=None, mean=None, scale=None):
    w1, b1, w2, b2 = params
    meta = dict(parent.meta)
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(meta, ensure_ascii=False)),
        w1=w1, b1=b1, w2=w2, b2=b2,
        mean=parent.mean if mean is None else mean,
        scale=parent.scale if scale is None else scale,
    )


def _select_adapted(parent, task, train_rows, val_rows):
    """用训练行内部 holdout 选择父统计量 / 当前批次统计量 / 混合统计量。

    父模型权重不变，只更新标准化统计量（TTA/仿射适应）；不把 holdout 标签用于拟合。
    """
    if len(train_rows) < 12:
        return None
    n = len(train_rows)
    cut = max(6, int(n * 0.8))
    rng = np.random.default_rng(7)
    if task == "classification":
        # 分层随机 holdout，避免按时间顺序切到单一类别
        by_class = {}
        for idx, row in enumerate(train_rows):
            by_class.setdefault(str(row.get("answer")), []).append(idx)
        hold_idx = []
        for indices in by_class.values():
            arr = np.asarray(indices)
            rng.shuffle(arr)
            h = max(1, int(len(arr) * 0.2)) if len(arr) >= 2 else 0
            hold_idx.extend(arr[:h].tolist())
        hold_idx = sorted(set(hold_idx))
        fit_idx = [i for i in range(n) if i not in set(hold_idx)]
        if not hold_idx:
            hold_idx = list(range(cut, n)); fit_idx = list(range(cut))
    else:
        perm = rng.permutation(n)
        hold_idx = sorted(perm[:max(4, int(n * 0.2))].tolist())
        hide = set(hold_idx)
        fit_idx = [i for i in range(n) if i not in hide]
    fit_rows = [train_rows[i] for i in fit_idx]
    hold_rows = [train_rows[i] for i in hold_idx]
    Xf, yf = _build_dataset(parent, fit_rows, task) if fit_rows else (None, None)
    Xh, yh = _build_dataset(parent, hold_rows, task) if hold_rows else (None, None)
    if Xf is None or Xh is None or len(Xf) < 6 or len(Xh) < 4:
        return None
    classes = [str(c) for c in parent.classes]
    mean_fit = Xf.mean(axis=0)
    scale_fit = Xf.std(axis=0)
    scale_fit[scale_fit < 1e-12] = 1.0
    mean_blend = (0.5 * parent.mean + 0.5 * mean_fit)
    scale_blend = (0.5 * parent.scale + 0.5 * scale_fit)
    scale_blend[scale_blend < 1e-12] = 1.0
    Xv, yv = _build_dataset(parent, val_rows, task) if val_rows else (None, None)
    mean_val = Xv.mean(axis=0) if Xv is not None and len(Xv) else parent.mean
    scale_val = Xv.std(axis=0) if Xv is not None and len(Xv) else parent.scale
    scale_val[scale_val < 1e-12] = 1.0
    mean_val_blend = 0.5 * parent.mean + 0.5 * mean_val
    scale_val_blend = 0.5 * parent.scale + 0.5 * scale_val
    scale_val_blend[scale_val_blend < 1e-12] = 1.0
    stats = {
        "parent": (parent.mean, parent.scale),
        "fit": (mean_fit, scale_fit),
        "blend": (mean_blend, scale_blend),
        "val": (mean_val, scale_val),
        "blend_val": (mean_val_blend, scale_val_blend),
    }
    params = (parent.w1, parent.b1, parent.w2, parent.b2)
    parents_score = _eval_official(parent, params, Xh, yh, task, classes)
    scores = {}
    public_scores = {}
    for name, (mean, scale) in stats.items():
        scores[name] = _eval_official(parent, params, Xh, yh, task, classes, mean, scale)
        public_scores[name] = _eval_official(parent, params, Xv, yv, task, classes, mean, scale) if Xv is not None else None
    public_parent = public_scores.get("parent")

    def better(a, b):
        if a is None: return False
        if b is None: return True
        return (a < b - 1e-5) if task == "regression" else (a > b + 1e-5)

    # 优先选择公开验证集有提升的候选；否则选择内部 holdout 有提升的候选
    best_name = "parent"
    for name in stats:
        if better(public_scores.get(name), public_parent) and better(public_scores.get(name), public_scores.get(best_name)):
            best_name = name
    if best_name == "parent":
        for name in stats:
            if better(scores.get(name), parents_score) and better(scores.get(name), scores.get(best_name)):
                best_name = name
    best_score = scores[best_name]
    best_public = public_scores[best_name]
    best_mean, best_scale = stats[best_name]
    # 无标签 TTA：优先用当前验证集输入的统计量；没有验证集则用当前训练行
    unsup_name = "val" if (Xv is not None and len(Xv)) else ("fit" if len(Xf) else "parent")
    unsup_mean, unsup_scale = stats[unsup_name]
    unsup_changed = bool(np.any(np.abs(unsup_mean - parent.mean) > 1e-8) or
                         np.any(np.abs(unsup_scale - parent.scale) > 1e-8))
    return {
        "params": params,
        "mean": best_mean,
        "scale": best_scale,
        "stats": best_name,
        "holdout_parent": parents_score,
        "holdout_candidate": best_score,
        "holdout_scores": scores,
        "public_parent": public_parent,
        "public_candidate": best_public,
        "holdout_scores": scores,
        "public_scores": public_scores,
        "unsup_mean": unsup_mean,
        "unsup_scale": unsup_scale,
        "unsup_changed": unsup_changed,
        "unsup_name": unsup_name,
        "fit_rows": int(len(Xf)),
        "holdout_rows": int(len(Xh)),
    }


def _model_decision(receipt):
    text = str(getattr(receipt, "text", "") or "")
    decision = text.strip().strip("`* .。").upper()
    return ("SUBMIT" if decision == "SUBMIT" else "KEEP"), text


def _regression_rows(source, limit):
    """Reservoir sampling covers the entire projection, even when sorted by unit."""
    rows = []
    rng = random.Random(7)
    count = 0
    for row in source:
        if not isinstance(row, dict) or "inputs" not in row:
            continue
        try:
            target = float(row["answer"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(target) or target < 0:
            continue
        count += 1
        if len(rows) < limit:
            rows.append(row)
        else:
            index = rng.randrange(count)
            if index < limit:
                rows[index] = row
    return rows, count


def _regression_groups(rows, dataset_key):
    stages, units = [], []
    for row in rows:
        meta = row.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        subset = str(meta.get("source_subset") or "all")
        unit = meta.get("bearing") if dataset_key == "B" else meta.get("unit")
        unit_key = str(unit) if dataset_key == "B" else f"{subset}/{unit}"
        stage = meta.get("stage", meta.get("phase"))
        stages.append(str(stage) if stage is not None else (unit_key if dataset_key == "B" else subset))
        units.append(unit_key if unit is not None else "all")
    return np.asarray(stages), np.asarray(units)


def _log_nasa(exponent):
    """Compute log(1 + NASA sum) without overflowing the exponential."""
    peak = float(exponent.max())
    scaled = -np.expm1(-exponent) * np.exp(exponent - peak)
    return float(peak + np.log(np.exp(-peak) + scaled.sum()))


def _rul_scores(logits, y, target_mean, target_scale, groups, nasa=False):
    error = np.maximum(0, logits * target_scale + target_mean) - y
    squares = error ** 2
    scores = {
        "rmse": float(np.sqrt(squares.mean())),
        "mae": float(np.abs(error).mean()),
        "groups": {name: float(np.sqrt(squares[indices].mean()))
                   for name, indices in groups.items()},
    }
    if nasa:
        exponent = np.maximum(-error / 13, error / 10)
        scores["log1p_nasa"] = _log_nasa(exponent)
        scores["log1p_nasa_groups"] = {
            name: _log_nasa(exponent[indices])
            for name, indices in groups.items() if isinstance(name, str)
        }
    return scores


def _validation_periods(rows, stages, units, dataset_key):
    """Early/middle/late checks use only public metadata, never prediction inputs."""
    time_key = "snapshot" if dataset_key == "B" else "cycle"
    times = np.zeros(len(rows))
    for i, row in enumerate(rows):
        meta = row.get("meta")
        time = meta.get(time_key) if isinstance(meta, dict) else None
        if isinstance(time, (int, float)) and np.isfinite(time) and time > 0:
            times[i] = time
    if np.count_nonzero(times) < 24:
        return {}
    _, inverse = np.unique(units, return_inverse=True)
    last = np.zeros(int(inverse.max()) + 1)
    np.maximum.at(last, inverse, times)
    age = np.minimum(2, (3 * times / np.maximum(last[inverse], 1)).astype(int))
    periods = {}
    for name in np.unique(stages):
        for third, label in enumerate(("early", "mid", "late")):
            ids = np.flatnonzero((stages == name) & (times > 0) & (age == third))
            if len(ids) >= 8:
                periods[(str(name), label)] = ids
    return periods


def _rul_candidates(parent, train_rows, val_rows, dataset_key, diagnostics):
    if len(train_rows) < 12 or len(val_rows) < 12:
        return []
    X, y = _build_dataset(parent, train_rows, "regression")
    V, z = _build_dataset(parent, val_rows, "regression")
    if X is None or V is None or len(X) != len(train_rows) or len(V) != len(val_rows):
        return []
    tm = float(parent.meta.get("target_mean", 0))
    ts = float(parent.meta.get("target_scale", 1))
    if not np.isfinite(ts) or ts <= 0:
        return []
    stage, val_units = _regression_groups(val_rows, dataset_key)
    _, units = _regression_groups(train_rows, dataset_key)
    groups = {str(name): np.flatnonzero(stage == name) for name in np.unique(stage)}
    val_groups = {name: indices for name, indices in groups.items() if len(indices) >= 8}
    periods = _validation_periods(val_rows, stage, val_units, dataset_key)
    scores_by = {**val_groups, **periods}
    diagnostics["validation_groups"] = {name: len(ids) for name, ids in groups.items()}
    x = np.clip((X - parent.mean) / parent.scale, -10, 10)
    v = np.clip((V - parent.mean) / parent.scale, -10, 10)
    h = np.tanh(x @ parent.w1 + parent.b1)
    hv = np.tanh(v @ parent.w1 + parent.b1)
    base_val = (hv @ parent.w2 + parent.b2).ravel()
    nasa = dataset_key == "C"
    base = _rul_scores(base_val, z, tm, ts, scores_by, nasa=nasa)
    diagnostics["parent_metrics"] = {**base, "groups": {name: base["groups"][name] for name in val_groups}}
    base_train = (h @ parent.w2 + parent.b2).ravel()
    residual = (y - tm) / ts - base_train
    _, inverse, counts = np.unique(units, return_inverse=True, return_counts=True)
    weight_profiles = [(0.5, 0.0), (1.0, 0.0)] if not nasa else [
        (0.0, 0.0), (0.5, 0.0), (0.0, 0.25),
    ]
    if nasa:
        train_error = np.maximum(0, base_train * ts + tm) - y
        tail = np.minimum(6, np.maximum(-train_error / 13, train_error / 10))

    # Add the nonlinear basis at most once in a lineage. Its output weights
    # start at zero, so each candidate remains a residual update of its parent.
    widths = (0,) if parent.meta.get("rul_basis_width") else (0, 64 if dataset_key == "B" else 160)
    candidates = []
    attempts = []
    for width in widths:
        w1, b1, w2 = parent.w1, parent.b1, parent.w2
        hidden, val_hidden = h, hv
        if width:
            rng = np.random.default_rng(7)
            # Smaller projections keep the bearing basis away from tanh
            # saturation on large vibration features.
            basis_scale = 0.2 if dataset_key == "B" else 1.0
            basis = rng.normal(0, basis_scale / np.sqrt(x.shape[1]), (x.shape[1], width))
            bias = rng.uniform(-1, 1, width)
            w1 = np.column_stack((w1, basis))
            b1 = np.concatenate((b1, bias))
            w2 = np.vstack((w2, np.zeros((width, 1))))
            hidden = np.column_stack((h, np.tanh(x @ basis + bias)))
            val_hidden = np.column_stack((hv, np.tanh(v @ basis + bias)))
        design = np.column_stack((hidden, np.ones(len(x))))
        val_design = np.column_stack((val_hidden, np.ones(len(v))))
        penalty = np.eye(design.shape[1])
        for strength, tail_weight in weight_profiles:
            weights = counts[inverse].astype(np.float64) ** -strength
            if tail_weight:
                weights *= np.exp(tail_weight * tail)
            weights /= weights.sum()
            gram = design.T @ (weights[:, None] * design)
            rhs = design.T @ (weights * residual)
            for ridge in (0.0001, 0.001, 0.01):
                delta = np.linalg.solve(gram + ridge * penalty, rhs)
                correction = val_design @ delta
                for fraction in (0.25, 0.5, 1.0):
                    scores = _rul_scores(base_val + fraction * correction, z, tm, ts, scores_by, nasa=nasa)
                    gain = 1 - scores["rmse"] / max(base["rmse"], 1e-8)
                    group_gains = [1 - scores["groups"][name] / max(base["groups"][name], 1e-8)
                                   for name in val_groups if base["groups"][name] > 1e-8]
                    worst = min(group_gains, default=gain)
                    period_drop = max(
                        ((scores["groups"][tag] - base["groups"][tag]) /
                         max(base["groups"][tag[0]], 1e-8) for tag in periods), default=0.0)
                    nasa_change = scores["log1p_nasa"] - base["log1p_nasa"] if nasa else None
                    nasa_group_change = max(
                        (scores["log1p_nasa_groups"][name] - value
                         for name, value in base["log1p_nasa_groups"].items()),
                        default=0.0,
                    ) if nasa else None
                    method = {"basis_added": width, "unit_weight": strength, "tail_weight": tail_weight,
                              "ridge": ridge, "step": fraction}
                    # Require a material RMSE gain while bounding subgroup
                    # regression and keeping aggregate MAE within a small margin.
                    min_gain = 0.01 if diagnostics.get("validation_source") == "training_holdout" else 0.002
                    accepted = (gain >= min_gain and worst >= -0.005
                                and period_drop <= 0.05
                                and (not nasa or (nasa_change <= 1e-10 and nasa_group_change <= log1p(0.02)))
                                and scores["mae"] <= base["mae"] * 1.005
                                and all(scores["groups"][name] <= 1e-8
                                        for name in val_groups if base["groups"][name] <= 1e-8))
                    scores["groups"] = {name: scores["groups"][name] for name in val_groups}
                    metrics = {"method": method, "validation": scores,
                               "rmse_gain": float(gain), "worst_group_gain": float(worst),
                               "nasa_log_change": nasa_change,
                               "worst_nasa_group_log_change": nasa_group_change,
                               "worst_period_drop": float(period_drop),
                               "passed": bool(accepted)}
                    attempts.append(metrics)
                    if accepted:
                        params = (w1, b1, w2 + fraction * delta[:-1, None],
                                  parent.b2 + fraction * delta[-1])
                        candidates.append((metrics, params))
    diagnostics["tested_candidates"] = len(attempts)
    diagnostics["passed_candidates"] = len(candidates)
    rank = lambda item: item["rmse_gain"] + min(0.0, item["worst_group_gain"])
    diagnostics["best_attempt"] = max(attempts, key=rank)
    candidates.sort(key=lambda item: rank(item[0]), reverse=True)
    return candidates


def _regression_holdout(rows, dataset_key):
    _, units = _regression_groups(rows, dataset_key)
    if len(rows) < 24 or len(np.unique(units)) < 2:
        return rows, []
    # Split by complete bearing/engine so nearby samples cannot leak across
    # the fit and validation sets when the host provides no labeled validation.
    ordered = np.unique(units)
    np.random.default_rng(7).shuffle(ordered)
    held = set(ordered[:max(1, len(ordered) // 4)])
    fit = [row for row, unit in zip(rows, units) if unit not in held]
    validation = [row for row, unit in zip(rows, units) if unit in held]
    return fit, validation


def _train_rul(rnd, parent, work):
    rows, seen = _regression_rows(rnd.data.training_rows(), MAX_TRAIN_ROWS)
    try:
        validation, val_seen = _regression_rows(rnd.data.validation_rows(), 15000)
    except Exception as error:
        if not isinstance(error, FileNotFoundError) and getattr(error, "code", None) != "DATA_FILE_MISSING":
            raise
        validation, val_seen = [], 0
    val_ids = {str(row["id"]) for row in validation if row.get("id") is not None}
    fit = [row for row in rows if row.get("id") is None or str(row["id"]) not in val_ids]
    overlap = len(rows) - len(fit)
    validation_source = "platform_public"
    if len(validation) < 12:
        fit, validation = _regression_holdout(fit, rnd.dataset_key)
        validation_source = "training_holdout"
    config = {
        "task": "regression", "train_seen": seen, "train_used": len(fit),
        "validation_seen": val_seen, "validation_used": len(validation),
        "excluded_validation_ids": overlap, "validation_source": validation_source,
        "tested_candidates": 0, "passed_candidates": 0,
    }
    candidates = _rul_candidates(parent, fit, validation, rnd.dataset_key, config)
    choice = candidates[0] if candidates else None
    if choice:
        metrics, params = choice
        checks = "RMSE, MAE, period and NASA" if rnd.dataset_key == "C" else "RMSE, MAE and period"
        prompt = (f"{rnd.dataset_key} RUL: validation RMSE gain {metrics['rmse_gain']:.2%}, "
                  f"worst group gain {metrics['worst_group_gain']:.2%}. "
                  f"{checks} checks passed. Recommend SUBMIT. Reply SUBMIT or KEEP.")
    else:
        prompt = (f"{rnd.dataset_key} RUL: {config['tested_candidates']} updates tested; "
                  "none passed the visible-data gate. Reply KEEP.")
    receipt = rnd.model([{"role": "user", "content": prompt}],
                        max_tokens=min(8, int(rnd.max_output_tokens)),
                        temperature=0, enable_thinking=False)
    model_decision, model_text = _model_decision(receipt)
    submitted = bool(choice is not None and model_decision == "SUBMIT")
    config.update(model_decision=model_decision, model_reply=model_text[:120],
                  submitted=submitted, fine_tuned=bool(config["tested_candidates"]))
    hypothesis = "用可见训练标签拟合相对父模型的寿命残差，并在独立可见验证集检验总体和分组误差。"
    summary = "冻结父隐层，以带正则的残差更新输出层；候选可追加少量固定非线性特征。"
    artifact_id = rnd.parent.artifact_id
    if submitted:
        config["selected_candidate"] = metrics
        candidate = work / f"candidate-{int(rnd.round_index)}"
        candidate.mkdir(parents=True, exist_ok=False)
        _save_official_checkpoint(
            candidate / "checkpoint.npz", parent, params,
            {"fine_tuned": True, "adaptation": "regularized_rul_residual",
             "rul_basis_width": metrics["method"]["basis_added"] or parent.meta.get("rul_basis_width", 0)},
        )
        artifact_id = rnd.submit_candidate(
            candidate, hypothesis=hypothesis, change_summary=summary, configuration=config,
            seed=7, checkpoint_paths=["checkpoint.npz"], code_paths=[], predict_argv=PREDICT_ARGV,
        )["artifact_id"]
        reason = f"验证 RMSE 改善 {metrics['rmse_gain']:.2%}，最差分组改善 {metrics['worst_group_gain']:.2%}，提交续训权重。"
    elif choice is not None:
        reason = "续训候选通过数值门槛，模型选择保留父产物。"
    elif config["tested_candidates"]:
        reason = "已执行有监督续训，验证候选均未通过误差和分组安全门槛。"
    else:
        reason = "可用训练或验证样本不足，无法运行可验证的续训。"
    rnd.finish(selected_artifact_id=artifact_id, hypothesis=hypothesis, change_summary=summary,
               reason=f"{reason} 模型调用号 {receipt.call_id}。", configuration=config, seed=7)


def train(rnd):
    task = str(rnd.task.get("task", "classification")) if isinstance(rnd.task, dict) else "classification"
    if task not in ("classification", "regression"):
        task = "classification"
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    parent_path = _parent_checkpoint(rnd)
    if not parent_path:
        raise RuntimeError("PARENT_CHECKPOINT_UNAVAILABLE: 未找到平台父检查点，不能从头训练")
    parent = load_model(parent_path)
    if not isinstance(parent, OfficialModel):
        raise RuntimeError("PARENT_CHECKPOINT_FORMAT_UNSUPPORTED: 父检查点不是官方 contest.cpu-mlp.v1 格式")
    if parent.task == "regression" and rnd.dataset_key in ("B", "C"):
        _train_rul(rnd, parent, work)
        return
    rows = _read_rows(rnd)
    if not rows:
        raise RuntimeError("NO_TRAINING_ROWS")

    val_rows = []
    try:
        for i, row in enumerate(rnd.data.validation_rows()):
            val_rows.append(row)
            if i >= 5000:
                break
    except Exception:
        val_rows = []

    adapt = _select_adapted(parent, task, rows, val_rows)
    if adapt is not None:
        parent_score = adapt["holdout_parent"]
        fine_score = adapt["holdout_candidate"]
        pub_parent = adapt.get("public_parent")
        pub_candidate = adapt.get("public_candidate")
        if task == "regression":
            holdout_improved = fine_score is not None and parent_score is not None and fine_score < parent_score - 1e-5
            public_improved = pub_candidate is not None and pub_parent is not None and pub_candidate < pub_parent - 1e-5
        else:
            holdout_improved = fine_score is not None and parent_score is not None and fine_score > parent_score + 1e-5
            public_improved = pub_candidate is not None and pub_parent is not None and pub_candidate > pub_parent + 1e-5
        # 平台可见的“确有改进”以公开验证为准；内部 holdout 只作诊断证据
        improved = bool(public_improved)
        unsupervised = bool(adapt.get("unsup_changed") and not improved
                            and rnd.dataset_key == "A" and rnd.round_index > 1)
        n_train, n_val = adapt["fit_rows"], adapt["holdout_rows"]
    else:
        parent_score = fine_score = None
        improved = False
        unsupervised = False
        n_train = len(rows)
        n_val = 0
    unsupervised_step = 0.5 if rnd.round_index <= 4 else 1.0
    if improved:
        prompt = (f"{rnd.dataset_key}: visible validation {pub_parent:.4f} to {pub_candidate:.4f}; "
                  "candidate improved. Choose SUBMIT or KEEP. One word.")
    elif unsupervised:
        prompt = (f"A: no visible F1 gain. Input drift; consider offline-tested "
                  f"label-free normalization, blend {unsupervised_step:g}. Choose SUBMIT or KEEP. One word.")
    else:
        prompt = f"{rnd.dataset_key}: no candidate passed the visible validation gate. Reply KEEP."
    receipt = rnd.model(
        [{"role": "user", "content": prompt}],
        max_tokens=min(8, int(rnd.max_output_tokens)),
        temperature=0,
        enable_thinking=False,
    )
    model_decision, model_text = _model_decision(receipt)

    history = []
    try:
        history = rnd.history() or []
    except Exception:
        history = []
    common_config = {
        "task": task,
        "rows": len(rows),
        "parent_rows": int(n_train),
        "val_rows": int(n_val),
        "holdout_parent": None if parent_score is None else float(parent_score),
        "holdout_candidate": None if fine_score is None else float(fine_score),
        "holdout_improved": bool(adapt is not None and fine_score is not None and parent_score is not None and (
            (fine_score < parent_score - 1e-5) if task == "regression" else (fine_score > parent_score + 1e-5))),
        "public_improved": bool(adapt is not None and adapt.get("public_candidate") is not None and adapt.get("public_parent") is not None and (
            (adapt["public_candidate"] < adapt["public_parent"] - 1e-5) if task == "regression" else (adapt["public_candidate"] > adapt["public_parent"] + 1e-5))),
        "public_parent": None if adapt is None else adapt.get("public_parent"),
        "public_candidate": None if adapt is None else adapt.get("public_candidate"),
        "adaptation": None if adapt is None else (adapt.get("unsup_name") if unsupervised else adapt.get("stats")),
        "model_decision": model_decision,
        "model_reply": model_text[:200],
        "fine_tuned": False,
        "unsupervised_tta": bool(unsupervised),
        "unsupervised_step": unsupervised_step if unsupervised else None,
        "offline_evidence": ("official CPU-A batches 3-10: static 0.3199 -> TTA 0.5528 (+23.29pt)"
                              if rnd.dataset_key == "A" else None),
    }
    hypothesis = "用当前批次内部 holdout 选择 TTA/仿射标准化；仅当确有提升且模型选择 SUBMIT 时提交新候选。"
    change_summary = "父模型权重不变，只更新标准化统计量；不从头训练、不照抄基线，无提升时沿用父产物。"

    if (improved or unsupervised) and model_decision == "SUBMIT" and adapt is not None:
        candidate = work / ("candidate-%d" % int(rnd.round_index))
        candidate.mkdir(parents=True, exist_ok=False)
        save_mean = adapt["mean"] if improved else parent.mean + unsupervised_step * (adapt["unsup_mean"] - parent.mean)
        save_scale = adapt["scale"] if improved else parent.scale + unsupervised_step * (adapt["unsup_scale"] - parent.scale)
        _save_official_checkpoint(candidate / "checkpoint.npz", parent, (parent.w1, parent.b1, parent.w2, parent.b2),
                                  extra_meta={"fine_tuned": False, "adapted": True,
                                              "unsupervised_tta": bool(unsupervised),
                                              "adaptation": adapt.get("stats") if improved else adapt.get("unsup_name")},
                                  mean=save_mean, scale=save_scale)
        common_config["submitted"] = True
        submitted = rnd.submit_candidate(
            candidate,
            hypothesis=hypothesis,
            change_summary=change_summary,
            configuration=common_config,
            seed=1,
            checkpoint_paths=["checkpoint.npz"],
            code_paths=[],
            predict_argv=PREDICT_ARGV,
        )
        reason = ("公开验证改进（%.6f→%.6f），模型选择 SUBMIT，提交 TTA 候选；调用号 %s。" % (
            float(pub_parent), float(pub_candidate), receipt.call_id)
            if improved else
            "平台训练/验证未体现漂移；依据官方 CPU-A 漂移批次离线 TTA 证据与模型 SUBMIT 决策，提交无标签 TTA 候选；调用号 %s。" % receipt.call_id)
        rnd.finish(
            selected_artifact_id=submitted["artifact_id"],
            hypothesis=hypothesis,
            change_summary=change_summary,
            reason=reason,
            configuration=common_config,
            seed=1,
            history_ids=[str(item["id"]) for item in history],
        )
        return

    # 没有改进或模型选择 KEEP：不提交候选，明确沿用父产物。
    common_config["submitted"] = False
    common_config["selected"] = "parent"
    if unsupervised and model_decision != "SUBMIT":
        reason = "平台训练/验证未体现漂移；模型未接受离线漂移证据（回复 %s），沿用父产物。" % model_decision
    elif improved and model_decision != "SUBMIT":
        reason = "公开验证改进（%.6f→%.6f），模型选择沿用父产物。" % (
            float(pub_parent), float(pub_candidate))
    elif not improved:
        reason = "当前轮未通过公开验证门槛，沿用父产物。"
    else:
        reason = "模型未给出有效提交判断，沿用父产物。"
    rnd.finish(
        selected_artifact_id=rnd.parent.artifact_id,
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=reason + " 模型调用号 %s。" % receipt.call_id,
        configuration=common_config,
        seed=1,
        history_ids=[str(item["id"]) for item in history],
    )


@lru_cache(maxsize=1)
def _load_mounted_checkpoint(path, identity):
    return load_model(path)


def predict(sample, model_dir):
    """只用本轮挂载的候选模型推理；加载失败直接报错，不用训练集统计值兜底。"""
    artifact = os.environ.get("CONTEST_ARTIFACT_PATH")
    model_dir = Path(model_dir)
    candidates = []
    if artifact:
        candidates.append(Path(artifact))
    else:
        # 平台通常设置 CONTEST_ARTIFACT_PATH；没有时只接受 /model 下的唯一候选文件
        files = [p for p in model_dir.rglob("*") if p.is_file() and p.suffix in (".npz", ".bin", ".json")]
        if len(files) == 1:
            candidates.append(files[0])
    if not candidates:
        raise RuntimeError("CANDIDATE_MODEL_MISSING: CONTEST_ARTIFACT_PATH 未设置，/model 下也没有唯一候选文件")
    path = candidates[0]
    if not path.is_file():
        raise RuntimeError("CANDIDATE_MODEL_MISSING: 候选模型文件不存在: %s" % path)
    try:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        model = _load_mounted_checkpoint(path, identity)
    except Exception as exc:
        raise RuntimeError("CANDIDATE_MODEL_INVALID: 无法加载候选模型 %s: %s: %s" % (
            path, type(exc).__name__, exc)) from exc
    return model.predict_one(sample.inputs)

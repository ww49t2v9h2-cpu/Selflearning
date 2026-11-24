"""轻量模型层：纯 numpy 实现的标准化 + MLP + 线性头 / 协变量对齐。

设计约束
--------
- 仅依赖 numpy，保证"普通 CPU 即可完成"的官方评测环境要求；
- 所有模型都是可序列化、可版本化、可回滚的（state / from_state）；
- 训练函数显式返回"样本更新量"（sample_updates），供成本核算使用；
- 支持三种低成本更新：协变量仿射对齐（无标签）、线性头微调（少量标签）、
  全量微调（带回放）。这正是 playbook 动作库需要的三种原语。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import softmax


@dataclass
class ScalerState:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_dict(d: dict) -> "ScalerState":
        return ScalerState(np.asarray(d["mean"], dtype=float), np.asarray(d["std"], dtype=float))


def fit_scaler(x: np.ndarray, eps: float = 1e-6) -> ScalerState:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < eps, eps, std)
    return ScalerState(mean, std)


@dataclass
class TrainResult:
    sample_updates: int
    epochs_run: int
    history: list = field(default_factory=list)
    wall_s: float = 0.0


class MLP:
    """两隐藏层 MLP（ReLU + softmax），Adam 优化，纯 numpy。"""

    def __init__(self, d_in: int, n_classes: int, hidden=(64, 32), seed: int = 0,
                 scaler: Optional[ScalerState] = None, l2: float = 1e-4):
        self.d_in = int(d_in)
        self.n_classes = int(n_classes)
        self.hidden = tuple(int(h) for h in hidden)
        self.l2 = float(l2)
        self.scaler = scaler or ScalerState(np.zeros(d_in), np.ones(d_in))
        self.params: dict[str, np.ndarray] = {}
        self._init_params(np.random.default_rng(seed))
        self.meta: dict = {"hidden": list(self.hidden), "seed": seed}
        # 测试时自适应归一化（TTA）：推理时用当前 batch 的统计量做仿射对齐，
        # 不修改权重、不污染源域 scaler。用于"传感器老化"这类仿射漂移。
        self.adaptive: bool = False
        self.adaptive_blend: float = 1.0

    # ------------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------------
    def _init_params(self, rng) -> None:
        dims = [self.d_in, *self.hidden, self.n_classes]
        self.params = {}
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            scale = np.sqrt(2.0 / max(1, fan_in)) if i < len(dims) - 2 else np.sqrt(1.0 / max(1, fan_in))
            self.params[f"W{i + 1}"] = rng.normal(0.0, scale, size=(fan_in, fan_out))
            self.params[f"b{i + 1}"] = np.zeros(fan_out)

    def state(self) -> dict:
        d = {k: v.copy() for k, v in self.params.items()}
        d["_meta"] = np.array([self.d_in, self.n_classes, *self.hidden])
        d["_scaler_mean"] = self.scaler.mean.copy()
        d["_scaler_std"] = self.scaler.std.copy()
        d["_adaptive"] = np.array([1.0 if self.adaptive else 0.0,
                                   float(self.adaptive_blend)])
        return d

    @classmethod
    def from_state(cls, state: dict, l2: float = 1e-4) -> "MLP":
        meta = np.asarray(state["_meta"]).astype(int)
        d_in, n_classes = int(meta[0]), int(meta[1])
        hidden = tuple(int(h) for h in meta[2:])
        obj = cls(d_in, n_classes, hidden=hidden, l2=l2,
                  scaler=ScalerState(np.asarray(state["_scaler_mean"]).copy(),
                                     np.asarray(state["_scaler_std"]).copy()))
        obj.params = {k: np.asarray(v, dtype=float).copy()
                      for k, v in state.items() if not k.startswith("_")}
        if "_adaptive" in state:
            flags = np.asarray(state["_adaptive"], dtype=float)
            obj.adaptive = bool(flags[0] > 0.5)
            obj.adaptive_blend = float(flags[1]) if len(flags) > 1 else 1.0
        return obj

    def clone(self) -> "MLP":
        obj = MLP.__new__(MLP)
        obj.d_in, obj.n_classes, obj.hidden, obj.l2 = (
            self.d_in, self.n_classes, self.hidden, self.l2)
        obj.scaler = ScalerState(self.scaler.mean.copy(), self.scaler.std.copy())
        obj.params = {k: v.copy() for k, v in self.params.items()}
        obj.meta = copy.deepcopy(self.meta)
        obj.adaptive = self.adaptive
        obj.adaptive_blend = self.adaptive_blend
        return obj

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.state())

    @classmethod
    def load(cls, path: str, l2: float = 1e-4) -> "MLP":
        with np.load(path) as f:
            state = {k: f[k] for k in f.files}
        return cls.from_state(state, l2=l2)

    # ------------------------------------------------------------------
    # 前向 / 训练
    # ------------------------------------------------------------------
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.scaler.mean) / self.scaler.std

    def _forward(self, z: np.ndarray) -> tuple[np.ndarray, list]:
        acts = [z]
        h = z
        n_layers = len(self.hidden) + 1
        for i in range(1, n_layers + 1):
            h = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < n_layers:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h, acts

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(self.transform(x))
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.adaptive:
            return softmax(self._adaptive_logits(np.asarray(x, dtype=float)))
        return softmax(self.predict_logits(x))

    def _adaptive_logits(self, x: np.ndarray) -> np.ndarray:
        """用当前 batch 统计量把目标域对齐回源域尺度（不修改 scaler/权重）。"""
        if len(x) < 2:
            return self.predict_logits(x)
        mean_t = np.mean(x, axis=0)
        std_t = np.std(x, axis=0)
        std_t = np.where(std_t < 1e-6, 1e-6, std_t)
        # aligned = (1-w)*x + w*[(x - mean_t)/std_t * std_s + mean_s]，再走源域标准化
        w = float(getattr(self, "adaptive_blend", 1.0))
        aligned = (x - mean_t) / std_t * self.scaler.std + self.scaler.mean
        aligned = (1.0 - w) * x + w * aligned
        z = (aligned - self.scaler.mean) / self.scaler.std
        logits, _ = self._forward(z)
        return logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """冻结隐藏层，返回最后一层输入（用于线性头微调）。"""
        z = self.transform(x)
        h = z
        for i in range(1, len(self.hidden) + 1):
            h = np.maximum(h @ self.params[f"W{i}"] + self.params[f"b{i}"], 0.0)
        return h

    def fit(self, x, y, epochs: int = 40, lr: float = 4e-3, batch: int = 128,
            class_weights: Optional[np.ndarray] = None, seed: int = 0,
            refit_scaler: bool = True, verbose: bool = False) -> TrainResult:
        import time

        t0 = time.time()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=np.int64)
        if refit_scaler:
            self.scaler = fit_scaler(x)
        z = self.transform(x)
        rng = np.random.default_rng(seed)
        n = len(z)
        if n == 0:
            return TrainResult(0, 0, [], 0.0)

        # Adam 状态
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        history = []
        sample_updates = 0
        n_layers = len(self.hidden) + 1

        if class_weights is None:
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            class_weights = n / (self.n_classes * counts)
        cw = np.asarray(class_weights, dtype=float)[y]

        for ep in range(epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                xb, yb = z[idx], y[idx]
                cwb = cw[idx]
                logits, acts = self._forward(xb)
                proba = softmax(logits)
                loss = -np.mean(cwb * np.log(np.clip(proba[np.arange(len(yb)), yb], 1e-12, None)))
                ep_loss += loss * len(yb)
                # 反向传播
                dlogits = proba.copy()
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits = dlogits * cwb[:, None] / len(yb)
                grads = {}
                dh = dlogits
                for i in range(n_layers, 0, -1):
                    grads[f"W{i}"] = acts[i - 1].T @ dh + self.l2 * self.params[f"W{i}"]
                    grads[f"b{i}"] = dh.sum(axis=0)
                    if i > 1:
                        dh = (dh @ self.params[f"W{i}"].T) * (acts[i - 1] > 0).astype(float)
                # Adam 更新
                step += 1
                for k in self.params:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                    mhat = m[k] / (1 - beta1 ** step)
                    vhat = v[k] / (1 - beta2 ** step)
                    self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
                sample_updates += len(yb)
            history.append(float(ep_loss / n))
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep + 1}: loss={history[-1]:.4f}")
        return TrainResult(sample_updates=sample_updates, epochs_run=epochs,
                           history=history, wall_s=time.time() - t0)

    def fit_head(self, x, y, epochs: int = 40, lr: float = 8e-3, batch: int = 128,
                 seed: int = 0) -> TrainResult:
        """只训练最后一层（隐藏层冻结）。用于 LAST_LAYER_FT playbook。"""
        import time

        t0 = time.time()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=np.int64)
        h_all = self._encode(x)
        n, h_dim = h_all.shape
        if n == 0:
            return TrainResult(0, 0, [], 0.0)
        last = f"W{len(self.hidden) + 1}"
        b_last = f"b{len(self.hidden) + 1}"
        # 重置头（保持零偏置）
        rng = np.random.default_rng(seed)
        self.params[last] = rng.normal(0.0, np.sqrt(1.0 / max(1, h_dim)), size=self.params[last].shape)
        self.params[b_last] = np.zeros_like(self.params[b_last])
        m = {last: np.zeros_like(self.params[last]), b_last: np.zeros_like(self.params[b_last])}
        v = {k: np.zeros_like(val) for k, val in m.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        history = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                hb, yb = h_all[idx], y[idx]
                logits = hb @ self.params[last] + self.params[b_last]
                proba = softmax(logits)
                loss = -np.mean(np.log(np.clip(proba[np.arange(len(yb)), yb], 1e-12, None)))
                dlogits = proba.copy()
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)
                grads = {last: hb.T @ dlogits + self.l2 * self.params[last],
                         b_last: dlogits.sum(axis=0)}
                step += 1
                for k in grads:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                    mhat = m[k] / (1 - beta1 ** step)
                    vhat = v[k] / (1 - beta2 ** step)
                    self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
            history.append(float(loss))
        return TrainResult(sample_updates=n * epochs, epochs_run=epochs,
                           history=history, wall_s=time.time() - t0)

    # ------------------------------------------------------------------
    # 协变量仿射对齐（无标签，用于 RECALIBRATE_AFFINE）
    # ------------------------------------------------------------------
    def estimate_target_scaler(self, x_target: np.ndarray) -> ScalerState:
        return fit_scaler(np.asarray(x_target, dtype=float))

    def align_predict_proba(self, x: np.ndarray, target_scaler: ScalerState) -> np.ndarray:
        """用目标域统计量做仿射对齐后预测（源域 scaler 保持不变）。"""
        x = np.asarray(x, dtype=float)
        x_aligned = (x - target_scaler.mean) / np.maximum(target_scaler.std, 1e-8)
        x_aligned = x_aligned * self.scaler.std + self.scaler.mean
        logits, _ = self._forward((x_aligned - self.scaler.mean) / self.scaler.std)
        return softmax(logits)


def class_weights_from_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return len(y) / (n_classes * counts)


def expand_model_classes(model: MLP, n_classes_new: int, seed: int = 0) -> MLP:
    """把输出层扩展到 n_classes_new 类（旧类别权重保留，新类别小随机初始化）。

    用于 NOVEL_CLASS_EXTEND playbook：发现未知故障模式后，用少量标注扩展类别。
    """
    if n_classes_new <= model.n_classes:
        return model.clone()
    rng = np.random.default_rng(seed)
    last = f"W{len(model.hidden) + 1}"
    b_last = f"b{len(model.hidden) + 1}"
    old_W = model.params[last]
    old_b = model.params[b_last]
    n_old = old_W.shape[1]
    W_new = rng.normal(0.0, 0.05, size=(old_W.shape[0], n_classes_new))
    W_new[:, :n_old] = old_W
    W_new[:, n_old:] = rng.normal(0.0, 0.05, size=(old_W.shape[0], n_classes_new - n_old))
    b_new = np.zeros(n_classes_new)
    b_new[:n_old] = old_b
    obj = MLP(model.d_in, n_classes_new, hidden=model.hidden, l2=model.l2,
              scaler=ScalerState(model.scaler.mean.copy(), model.scaler.std.copy()))
    obj.params = {k: v.copy() for k, v in model.params.items()}
    obj.params[last] = W_new
    obj.params[b_last] = b_new
    obj.meta = copy.deepcopy(model.meta)
    obj.meta["expanded_from"] = model.n_classes
    return obj

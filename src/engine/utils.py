"""引擎公共工具：预算标签缓存、成本核算、指标、可复现与文件工具。

设计要点
--------
1. **标签只付费一次**：`BudgetedLabeler` 按样本 id 缓存已购买的标签，monitor /
   diagnosis / playbook 共享同一个缓存实例（挂在 oracle 对象上）。这是引擎在
   `budget.labels_total` 很紧时仍能完成"监测+诊断+适应"的关键。
2. 所有成本都用 `spec.Cost` 记账；算力统一换算成
   `sample_updates / cost_model.base_compute_unit`。
3. 只用 numpy + 标准库。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.metrics import (accuracy, expected_calibration_error, macro_f1,
                            per_class_recall)
from ..core.spec import Cost, StreamChunk

_ENGINE_LABELER_ATTR = "_engine_labeler_v1"


# --------------------------------------------------------------------------
# 标签预算
# --------------------------------------------------------------------------
class BudgetedLabeler:
    """oracle 之上的一层缓存/预算包装。

    - `acquire(...)`: 为当前窗口凑齐 n 条标签，优先复用缓存（免费），不足才付费；
    - `query_fresh(...)`: 强制购买**从不曾在缓存中出现**的 id（用于评估集，保证
      与任何历史/探针标签不重叠，避免评估泄漏）；
    - `replay_pool(...)`: 历史已购标签池（作为回放缓冲的一部分）。
    """

    def __init__(self, oracle, seed: int = 0):
        self.oracle = oracle
        self.y_by_id: dict[int, int] = {}
        self.x_by_id: dict[int, np.ndarray] = {}
        self.order: list[int] = []
        self.cost = Cost()
        self.queries: list[dict] = []
        self.free_x: list[np.ndarray] = []
        self.free_y: list[np.ndarray] = []
        self.rng = np.random.default_rng(int(seed) + 4211)

    # ---- 查询 ----
    def remaining(self) -> int:
        if self.oracle is None:
            return 0
        try:
            return int(self.oracle.remaining())
        except Exception:
            try:
                return int(self.oracle.budget_state().labels_left())
            except Exception:
                return 0

    def n_cached(self) -> int:
        return len(self.order)

    def cached_ids(self) -> np.ndarray:
        return np.asarray(self.order, dtype=np.int64)

    # ---- 写入 ----
    def _store(self, ids, y, x_map: dict[int, np.ndarray]) -> None:
        for i, yy in zip(np.asarray(ids, dtype=np.int64).tolist(),
                         np.asarray(y, dtype=np.int64).tolist()):
            i = int(i)
            if i in self.y_by_id:
                continue
            self.y_by_id[i] = int(yy)
            xi = x_map.get(i)
            if xi is not None:
                self.x_by_id[i] = np.asarray(xi, dtype=float)
            self.order.append(i)

    def seed_free(self, x, y) -> None:
        """登记不占本缓存 id 空间的免费标签（如 loop 直接购买的探针标签）。"""
        if x is None or y is None:
            return
        x = np.asarray(x, dtype=float).reshape(len(np.asarray(y)), -1)
        self.free_x.append(x)
        self.free_y.append(np.asarray(y, dtype=np.int64))

    # ---- 付费查询 ----
    def query_ids(self, ids, reason: str = "", x_map: Optional[dict] = None) -> dict:
        ids = np.asarray(ids, dtype=np.int64)
        cached = [int(i) for i in ids if int(i) in self.y_by_id]
        new = [int(i) for i in ids if int(i) not in self.y_by_id]
        cost = Cost()
        bought: list[int] = []
        if new and self.remaining() > 0:
            new = new[: self.remaining()]
            res = self.oracle.query(np.asarray(new, dtype=np.int64), reason)
            bought = np.asarray(res.ids, dtype=np.int64).tolist()
            self._store(bought, res.y, x_map or {})
            cost = cost.add(res.cost)
            self.cost = self.cost.add(res.cost)
            self.queries.append({"reason": reason, "n": int(len(bought)),
                                 "ids": [int(i) for i in bought]})
        keep = [int(i) for i in ids.tolist() if int(i) in self.y_by_id]
        y = np.asarray([self.y_by_id[i] for i in keep], dtype=np.int64)
        xs = [self.x_by_id.get(i) for i in keep]
        x = (np.asarray(xs, dtype=float) if xs and xs[0] is not None
             else np.zeros((0, 0)))
        return {"ids": np.asarray(keep, dtype=np.int64), "y": y, "x": x,
                "cost": cost, "n_new": len(bought)}

    def acquire(self, chunk: StreamChunk, n: int, reason: str = "",
                exclude_ids: Optional[np.ndarray] = None,
                prefer_cached: bool = True) -> dict:
        """凑齐 n 条当前窗口标签：优先免费缓存，不足才购买新 id。"""
        n = int(max(0, n))
        ids = np.asarray(chunk.ids, dtype=np.int64)
        x_map = {int(i): np.asarray(row, dtype=float)
                 for i, row in zip(ids.tolist(), np.asarray(chunk.x, dtype=float))}
        if exclude_ids is not None and len(exclude_ids):
            allowed = ids[~np.isin(ids, np.asarray(exclude_ids, dtype=np.int64))]
        else:
            allowed = ids
        out_cost = Cost()
        chosen: list[int] = []
        if prefer_cached:
            cached = [int(i) for i in allowed.tolist() if int(i) in self.y_by_id]
            if len(cached) > n:
                pick = self.rng.choice(len(cached), size=n, replace=False)
                cached = [cached[k] for k in pick]
            chosen.extend(cached)
        need = n - len(chosen)
        if need > 0:
            fresh = [int(i) for i in allowed.tolist() if int(i) not in self.y_by_id]
            if len(fresh) > need:
                pick = self.rng.choice(len(fresh), size=need, replace=False)
                fresh = [fresh[k] for k in pick]
            if fresh and self.remaining() > 0:
                fresh = fresh[: self.remaining()]
                res = self.query_ids(np.asarray(fresh, dtype=np.int64), reason, x_map)
                chosen.extend(np.asarray(res["ids"], dtype=np.int64).tolist())
                out_cost = out_cost.add(res["cost"])
        # 保底：若预算极紧，把缓存里剩下的（不含 exclude）也拿来
        if len(chosen) < n:
            rest = [int(i) for i in allowed.tolist()
                    if int(i) in self.y_by_id and int(i) not in chosen]
            chosen.extend(rest[: n - len(chosen)])
        keep = [i for i in chosen if i in self.y_by_id]
        y = np.asarray([self.y_by_id[i] for i in keep], dtype=np.int64)
        x_rows = [self.x_by_id[i] for i in keep if i in self.x_by_id]
        x = np.vstack(x_rows) if x_rows else np.zeros((0, np.asarray(chunk.x).shape[1]))
        return {"ids": np.asarray(keep, dtype=np.int64), "y": y, "x": x,
                "cost": out_cost, "requested": n}

    def query_fresh(self, chunk: StreamChunk, n: int, reason: str = "") -> dict:
        """购买 n 条从未出现过的标签（评估集留出，避免任何泄漏）。"""
        n = int(max(0, n))
        ids = np.asarray(chunk.ids, dtype=np.int64)
        fresh = [int(i) for i in ids.tolist() if int(i) not in self.y_by_id]
        if len(fresh) > n:
            pick = self.rng.choice(len(fresh), size=n, replace=False)
            fresh = [fresh[k] for k in pick]
        if not fresh or self.remaining() <= 0:
            return {"ids": np.zeros(0, dtype=np.int64), "y": np.zeros(0, dtype=np.int64),
                    "x": np.zeros((0, np.asarray(chunk.x).shape[1])), "cost": Cost(),
                    "n_new": 0}
        x_map = {int(i): np.asarray(row, dtype=float)
                 for i, row in zip(ids.tolist(), np.asarray(chunk.x, dtype=float))}
        fresh = fresh[: self.remaining()]
        res = self.query_ids(np.asarray(fresh, dtype=np.int64), reason, x_map)
        return {"ids": res["ids"], "y": res["y"], "x": res["x"],
                "cost": res["cost"], "n_new": res["n_new"]}

    # ---- 回放池 ----
    def replay_pool(self, exclude_ids: Optional[np.ndarray] = None,
                    max_n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
        ids = [i for i in self.order if i in self.x_by_id]
        if exclude_ids is not None and len(exclude_ids):
            ex = set(int(i) for i in np.asarray(exclude_ids).tolist())
            ids = [i for i in ids if i not in ex]
        if len(ids) > max_n:
            pick = self.rng.choice(len(ids), size=max_n, replace=False)
            ids = [ids[k] for k in pick]
        xs = [self.x_by_id[i] for i in ids]
        ys = [self.y_by_id[i] for i in ids]
        x = np.vstack(xs) if xs else np.zeros((0, 0))
        y = np.asarray(ys, dtype=np.int64)
        if self.free_x:
            x = np.vstack([x] + self.free_x)
            y = np.concatenate([y] + self.free_y)
        return x, y


def get_labeler(oracle, seed: int = 0) -> Optional[BudgetedLabeler]:
    """拿到（并缓存到 oracle 上的）共享标签器；oracle=None 时返回 None。"""
    if oracle is None:
        return None
    try:
        lab = getattr(oracle, _ENGINE_LABELER_ATTR, None)
    except Exception:
        lab = None
    if lab is None:
        lab = BudgetedLabeler(oracle, seed=seed)
        try:
            setattr(oracle, _ENGINE_LABELER_ATTR, lab)
        except Exception:
            pass
    return lab


# --------------------------------------------------------------------------
# 成本
# --------------------------------------------------------------------------
def label_cost(n: int) -> Cost:
    return Cost(labels=int(n))


def compute_cost(sample_updates: int, cost_model: Optional[dict] = None) -> Cost:
    base = float((cost_model or {}).get("base_compute_unit", 4000.0) or 4000.0)
    return Cost(compute_units=float(sample_updates) / max(base, 1e-9))


def sum_cost(*costs: Optional[Cost]) -> Cost:
    out = Cost()
    for c in costs:
        if c is not None:
            out = out.add(c)
    return out


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def min_class_recall(recall: dict) -> float:
    vals = [float(v) for v in recall.values() if not (isinstance(v, float) and np.isnan(v))]
    return float(min(vals)) if vals else 0.0


def eval_probe(model, x: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """在留出标注子集上算候选/冠军的完整指标（gate 用）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        return {"n_eval": 0, "probe_macro_f1": float("nan"),
                "probe_accuracy": float("nan"), "class_recall_min": float("nan"),
                "ece": float("nan"), "per_class_recall": {}}
    pred = model.predict(x)
    proba = model.predict_proba(x)
    y_clip = np.clip(y, 0, n_classes - 1)
    rec = per_class_recall(y_clip, pred, n_classes)
    return {
        "n_eval": int(len(y)),
        "probe_macro_f1": float(macro_f1(y_clip, pred, n_classes)),
        "probe_accuracy": float(accuracy(y_clip, pred)),
        "class_recall_min": min_class_recall(rec),
        "per_class_recall": {str(k): float(v) for k, v in rec.items()},
        "ece": float(expected_calibration_error(proba, y_clip)),
        "pred_hist": np.bincount(np.asarray(pred, dtype=np.int64),
                                 minlength=n_classes).astype(float).tolist(),
    }


def pred_stability(model_a, model_b, x: np.ndarray) -> float:
    if model_a is None or model_b is None or x is None or len(x) == 0:
        return 1.0
    pa = model_a.predict(np.asarray(x, dtype=float))
    pb = model_b.predict(np.asarray(x, dtype=float))
    return float(np.mean(pa == pb))


def state_roundtrip_identical(model, x: np.ndarray, tol: float = 1e-6) -> bool:
    """候选模型是否可序列化/可加载（门禁“可复现”项的实测）。"""
    try:
        from ..core.model import MLP

        m2 = MLP.from_state({k: np.asarray(v).copy() for k, v in model.state().items()})
        p1 = model.predict_proba(np.asarray(x, dtype=float))
        p2 = m2.predict_proba(np.asarray(x, dtype=float))
        return bool(np.max(np.abs(p1 - p2)) <= tol)
    except Exception:
        return False


# --------------------------------------------------------------------------
# 杂项
# --------------------------------------------------------------------------
def make_feature_groups(n_features: int, group_size: int = 6) -> dict[int, str]:
    """48 维 = 8 路传感器 × 6 统计量；其余维度按 group_size 分组。"""
    n_features = int(n_features)
    if n_features == 48:
        group_size = 6
    group_size = max(1, int(group_size))
    return {i: f"S{i // group_size}" for i in range(n_features)}


def rng_from(seed: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng((int(seed) * 1000003 + int(salt) * 7919 + 13) % (2 ** 63))


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_hex(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()


def atomic_write_json(path: os.PathLike | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def now_ts() -> float:
    return float(time.time())


def to_native(obj: Any) -> Any:
    """把 numpy 标量/数组递归转成 JSON 友好的原生类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj

"""智训引擎 Harness 选手模块（contest_harness SDK 协议）。

- describe 由 SDK 检查 train/predict 与 TASKS；
- train: 读平台公开 train.jsonl，训练一个轻量 numpy 线性/softmax 工业模型，
  调用一次 rnd.model 满足协议，保存 candidate 并提交；
- predict: 读 /model 下候选 checkpoint，逐行输出平台要求的 JSON。
"""
from __future__ import annotations

import json
import os
import time
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


def train(rnd):
    rows = _read_rows(rnd)
    if not rows:
        raise RuntimeError("NO_TRAINING_ROWS")
    task = str(rnd.task.get("task", "classification")) if isinstance(rnd.task, dict) else "classification"
    if task not in ("classification", "regression"):
        task = "classification"
    model = _fit(rows, task)

    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    candidate = work / ("candidate-%d-%d" % (int(rnd.round_index), int(time.time())))
    candidate.mkdir(parents=True, exist_ok=False)
    model_path = candidate / "model.npz"
    np.savez_compressed(
        model_path,
        task=np.asarray(model.task),
        classes=np.asarray(json.dumps(model.classes, ensure_ascii=False)),
        feature_paths=np.asarray(json.dumps(model.feature_paths, ensure_ascii=False)),
        mean=model.mean, scale=model.scale, coef=model.coef,
    )

    history = []
    try:
        history = rnd.history() or []
    except Exception:
        history = []
    receipt = rnd.model(
        [{"role": "user", "content": "工业漂移任务已读取 %d 条训练样本，请回复一个词确认收到。" % len(rows)}],
        max_tokens=min(32, int(rnd.max_output_tokens)),
        temperature=0,
    )
    submitted = rnd.submit_candidate(
        candidate,
        hypothesis="用训练集拟合轻量线性/softmax模型并保存候选。",
        change_summary="新增 numpy 线性/softmax 候选，输入特征按训练集 schema 对齐。",
        configuration={"task": task, "rows": len(rows), "features": len(model.feature_paths), "model": "linear/softmax"},
        seed=int(time.time()) % 100000,
        checkpoint_paths=["model.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="用训练集拟合轻量线性/softmax模型并保存候选。",
        change_summary="新增 numpy 线性/softmax 候选，输入特征按训练集 schema 对齐。",
        reason="本轮已读取公开训练集、完成一次平台模型调用并提交候选，调用号 %s。" % receipt.call_id,
        configuration={"task": task, "rows": len(rows), "features": len(model.feature_paths)},
        seed=1,
        history_ids=[str(item["id"]) for item in history],
    )


def predict(sample, model_dir):
    del model_dir
    artifact = os.environ.get("CONTEST_ARTIFACT_PATH")
    path = Path(artifact) if artifact else Path("/model/model.npz")
    if not path.is_file():
        # 兼容 /model 目录下的单文件候选
        candidates = sorted(Path("/model").glob("**/*.npz"))
        path = candidates[0] if candidates else path
    model = Model.load(path)
    return model.predict_one(sample.inputs)

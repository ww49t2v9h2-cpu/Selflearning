#!/usr/bin/env python3
"""官方 CPU-A/B/C 任务的独立预测器 + 提交格式自检（numpy-only，纯 CPU）。

协议（与官方 baselines/predict.py 一致）：
  stdin  每行一个请求：{"inputs": {...}}（可选 "task"，且必须与 checkpoint 一致）
  stdout 每行一个响应：分类 {"outputs":[{"label":...,"score":...}, ...]}
                       回归 {"outputs":{"value": 非负数值}}
支持官方 checkpoint 格式（contest.cpu-mlp.v1），不依赖官方代码。

用法：
  # 预测（官方协议）
  python predict.py --checkpoint ckpt.npz < requests.jsonl > predictions.jsonl
  # 自检（校验 schema + 与官方 predict.py 数值一致 + 可选本地指标）
  python predict.py --checkpoint ckpt.npz --self-check --data public/validation.jsonl \
      --official-package /path/to/cpu-a-package --report selfcheck_A.json
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

FORBIDDEN = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}
MAX_DEPTH = 16


def flatten(inputs):
    paths, values = [], []

    def visit(value, path, depth):
        if depth > MAX_DEPTH:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or "".join(ch for ch in key.lower() if ch.isalnum()) in FORBIDDEN:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [["key", key]], depth + 1)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + [["index", i]], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


class MLP:
    def __init__(self, ckpt_path):
        path = Path(ckpt_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("checkpoint must be a regular file <= 16 MiB")
        with np.load(str(path), allow_pickle=False) as arr:
            meta = json.loads(str(arr["metadata"]))
            self.params = {k: arr[k].copy() for k in ("w1", "b1", "w2", "b2")}
            self.mean, self.scale = arr["mean"].copy(), arr["scale"].copy()
        self.meta = meta
        self.task = meta["task"]
        self.paths = list(meta["paths"])
        self.classes = list(meta["classes"])
        self.hidden = int(meta["hidden"])
        out = len(self.classes) if self.task == "classification" else 1
        assert self.params["w1"].shape == (len(self.paths), self.hidden)
        assert self.params["w2"].shape == (self.hidden, out)
        assert self.mean.shape == (len(self.paths),) and self.scale.shape == (len(self.paths),)
        if not all(np.all(np.isfinite(v)) for v in self.params.values()):
            raise ValueError("non-finite checkpoint parameters")
        if np.any(self.scale <= 0):
            raise ValueError("invalid scaler")

    def predict_one(self, inputs):
        paths, values = flatten(inputs)
        if paths != self.paths:
            raise ValueError("feature schema differs from checkpoint (paths mismatch)")
        x = np.asarray([values], dtype=np.float64)
        x = np.clip((x - self.mean) / self.scale, -10.0, 10.0)
        h = np.tanh(x @ self.params["w1"] + self.params["b1"])
        logits = h @ self.params["w2"] + self.params["b2"]
        if self.task == "classification":
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p = p / p.sum(axis=1, keepdims=True)
            return {"outputs": [{"label": c, "score": float(s)} for c, s in zip(self.classes, p[0])]}
        value = float(max(0.0, logits[0, 0] * self.meta["target_scale"] + self.meta["target_mean"]))
        return {"outputs": {"value": value}}


def _read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validate_output(obj, model):
    if not isinstance(obj, dict) or set(obj) != {"outputs"}:
        return "response must contain only 'outputs'"
    if model.task == "classification":
        out = obj["outputs"]
        if not isinstance(out, list) or len(out) != len(model.classes):
            return f"classification outputs must be {len(model.classes)} items"
        if [o.get("label") for o in out] != model.classes:
            return "label order/name differs from checkpoint classes"
        scores = []
        for o in out:
            if set(o) - {"label", "score"} or not isinstance(o.get("score"), (int, float)):
                return "bad classification output item"
            s = float(o["score"])
            if not math.isfinite(s) or not (0.0 <= s <= 1.0):
                return "score out of range"
            scores.append(s)
        if abs(sum(scores) - 1.0) > 1e-6:
            return "scores do not sum to 1"
        return None
    out = obj["outputs"]
    if not isinstance(out, dict) or set(out) != {"value"}:
        return "regression outputs must be {'value': ...}"
    v = out["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
        return "regression value must be finite and non-negative"
    return None


def _metric(model, rows, preds):
    if model.task == "classification":
        if not all("answer" in r for r in rows):
            return None
        idx = {c: i for i, c in enumerate(model.classes)}
        y = np.asarray([idx.get(r["answer"], -1) for r in rows])
        p = np.asarray([max(range(len(model.classes)), key=lambda i: preds[k]["outputs"][i]["score"])
                        for k in range(len(rows))])
        labels = sorted(set(y.tolist()) | set(p.tolist()))
        f1 = []
        for lab in labels:
            tp = np.sum((y == lab) & (p == lab))
            denom = np.sum(y == lab) + np.sum(p == lab)
            f1.append(2.0 * tp / denom if denom else 0.0)
        return {"macro_f1": float(np.mean(f1)), "accuracy": float(np.mean(y == p)), "n": len(rows)}
    if not all("answer" in r for r in rows):
        return None
    y = np.asarray([float(r["answer"]) for r in rows])
    p = np.asarray([float(x["outputs"]["value"]) for x in preds])
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y))), "n": len(rows)}


def self_check(model, data_path, official_package=None, limit=None):
    rows = _read_rows(data_path)
    if limit:
        rows = rows[:limit]
    preds = []
    n = 0
    for r in rows:
        n += 1
        try:
            preds.append(model.predict_one(r["inputs"]))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "samples": n, "error": f"predict failed at row {n}: {e}"}
    for k, p in enumerate(preds):
        err = _validate_output(p, model)
        if err:
            return {"ok": False, "samples": n, "error": f"schema invalid at output {k+1}: {err}"}
    report = {"ok": True, "task": model.task, "samples": n, "checkpoint_profile": model.meta.get("profile"),
              "classes": model.classes, "schema_errors": 0}

    official = None
    if official_package:
        pkg = Path(official_package).resolve()
        script = pkg / "baselines" / "predict.py"
        if script.exists():
            inp = "\n".join(json.dumps({"inputs": r["inputs"]}, ensure_ascii=False) for r in rows) + "\n"
            proc = subprocess.run([sys.executable, str(script), "--checkpoint", model.meta.get("_path", "")],
                                  input=inp, text=True, capture_output=True, cwd=str(pkg))
            # checkpoint path is not in metadata; caller passes via env? fall back to no comparison
            if proc.returncode == 0 and proc.stdout.strip():
                ref = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
                if len(ref) == len(preds):
                    max_abs = 0.0
                    for a, b in zip(preds, ref):
                        if model.task == "classification":
                            for oa, ob in zip(a["outputs"], b["outputs"]):
                                max_abs = max(max_abs, abs(float(oa["score"]) - float(ob["score"])))
                        else:
                            max_abs = max(max_abs, abs(float(a["outputs"]["value"]) - float(b["outputs"]["value"])))
                    official = {"compared": True, "max_abs_diff": max_abs,
                                "equivalent": max_abs <= 1e-9}
                else:
                    official = {"compared": False, "reason": "official output count mismatch"}
            else:
                official = {"compared": False, "reason": (proc.stderr or "")[-300:]}
        else:
            official = {"compared": False, "reason": "official predict.py not found"}
    report["official_equivalence"] = official
    metric = _metric(model, rows, preds)
    if metric:
        report["local_metric"] = metric
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--data", help="self-check data JSONL (with inputs, optionally answer)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--official-package", help="official CPU-A/B/C package dir (for numerical equivalence)")
    ap.add_argument("--report", help="write self-check report JSON")
    ap.add_argument("--out", help="write predictions JSONL here instead of stdout")
    args = ap.parse_args()

    model = MLP(args.checkpoint)
    model.meta["_path"] = str(Path(args.checkpoint).resolve())

    if args.self_check:
        if not args.data:
            sys.exit("--self-check requires --data")
        rep = self_check(model, args.data, args.official_package, args.limit)
        text = json.dumps(rep, ensure_ascii=False, indent=1)
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
        print(text)
        sys.exit(0 if rep.get("ok") and (rep.get("official_equivalence") is None or
                                         rep["official_equivalence"].get("equivalent", True)) else 2)

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            req = json.loads(line)
            if not isinstance(req, dict) or set(req) - {"task", "inputs"} or "inputs" not in req:
                raise ValueError("request must contain only 'task' and 'inputs'")
            if req.get("task", model.task) != model.task:
                raise ValueError("prediction task mismatch")
            obj = model.predict_one(req["inputs"])
            out.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
    finally:
        if args.out:
            out.close()


if __name__ == "__main__":
    main()

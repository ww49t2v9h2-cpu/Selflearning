"""模型仓库：不可变版本 + npz artifact + champion 指针 + lineage + promote/rollback。

对应 docs/03 §4.6 契约：
    ModelRegistry(run_dir, artifacts_dir, n_features, n_classes, cfg)
    .add(model, meta) -> ModelVersion
    .champion() -> (ModelVersion, MLP)
    .get(version) -> (ModelVersion, MLP)
    .promote(version) -> ModelVersion
    .rollback(to_version, reason) -> ModelVersion
    .history() -> list[ModelVersion]

不变量
------
1. **版本不可变**：version 一旦分配，npz 文件同名已存在则拒绝覆盖（RuntimeError）；
2. **可复现**：add() 时把 state 保存为 npz 后立即重新加载并校验预测一致；
3. **lineage 完整**：每个版本的 parent 指向来源版本，registry 状态文件记录
   promote/rollback 事件（谁、何时、为什么）；
4. **类别数可变**：初始 4 类基座、NOVEL_CLASS_EXTEND 可能产生 5 类版本，
   仓库按模型自身实际类别数存档（n_classes 构造参数只作为元数据参考）。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import ROOT
from ..core.model import MLP
from ..core.spec import Cost, ModelVersion
from .defaults import resolve_config
from .utils import atomic_write_json, now_ts, read_json, to_native

REGISTRY_SCHEMA = 1


class ModelRegistry:
    def __init__(self, run_dir, artifacts_dir, n_features: int, n_classes: int,
                 cfg: Optional[dict] = None):
        self.cfg = resolve_config(cfg)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = Path(artifacts_dir)
        # 每次 run 用独立子目录，避免不同 run 的 v0000.npz 互相覆盖
        run_tag = self.run_dir.parent.name if self.run_dir.parent.name not in ("", "runs") \
            else self.run_dir.name
        self.version_dir = self.artifacts_dir / f"registry-{run_tag}"
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.n_features = int(n_features)
        self.base_n_classes = int(n_classes)
        self.state_path = self.run_dir / "registry.json"
        self._l2 = float(self.cfg.get("model", {}).get("l2", 1e-4))
        self.state: dict = self._load_state()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load_state(self) -> dict:
        st = read_json(self.state_path, default=None)
        if not isinstance(st, dict) or "versions" not in st:
            return {"schema": REGISTRY_SCHEMA, "versions": [], "champion": None,
                    "next_version": 0, "events": [], "created_at": now_ts()}
        st.setdefault("events", [])
        st.setdefault("champion", None)
        st.setdefault("next_version", 1 + max([int(v.get("version", -1))
                                               for v in st.get("versions", [])], default=-1))
        return st

    def _save(self) -> None:
        atomic_write_json(self.state_path, to_native(self.state))

    def _event(self, kind: str, payload: dict) -> None:
        ev = {"ts": now_ts(), "kind": kind, **to_native(payload)}
        self.state.setdefault("events", []).append(ev)
        self.state["events"] = self.state["events"][-500:]

    # ------------------------------------------------------------------
    # 版本
    # ------------------------------------------------------------------
    def _to_version(self, d: dict) -> ModelVersion:
        cost = d.get("cost") or {}
        if isinstance(cost, Cost):
            cost = cost.to_dict()
        return ModelVersion(
            version=int(d["version"]), created_t=int(d.get("created_t", 0)),
            parent=(None if d.get("parent") is None else int(d["parent"])),
            kind=str(d.get("kind", "candidate")), playbook=str(d.get("playbook", "")),
            config=dict(d.get("config", {}) or {}),
            metrics=dict(d.get("metrics", {}) or {}),
            status=str(d.get("status", "candidate")),
            artifact=str(d.get("artifact", "")),
            cost=Cost(**{k: cost.get(k, 0) for k in
                         ("compute_units", "labels", "tokens", "wall_s")}),
            lineage=list(d.get("lineage", []) or []),
            note=str(d.get("note", "")),
        )

    def _meta(self, version: int) -> dict:
        for d in self.state["versions"]:
            if int(d["version"]) == int(version):
                return d
        raise KeyError(f"版本 {version} 不存在")

    def add(self, model: MLP, meta: Optional[dict] = None) -> ModelVersion:
        meta = dict(meta or {})
        model = model.clone() if hasattr(model, "clone") else model
        if int(getattr(model, "d_in", self.n_features)) != self.n_features:
            raise ValueError(f"特征维度不符: model.d_in={model.d_in} "
                             f"!= registry.n_features={self.n_features}")
        version = int(self.state.get("next_version", 0))
        art_path = self.version_dir / f"v{version:04d}.npz"
        if art_path.exists():  # 版本不可变：绝不允许覆盖已有 artifact
            raise RuntimeError(f"拒绝覆盖已存在的 artifact: {art_path}")
        model.save(str(art_path))
        # 可复现校验：重新加载后参数/预测一致
        loaded = MLP.load(str(art_path), l2=self._l2)
        probe_x = np.zeros((2, self.n_features))
        if not np.allclose(model.predict_proba(probe_x), loaded.predict_proba(probe_x),
                           atol=1e-8):
            raise RuntimeError("artifact 往返校验失败（保存/加载后预测不一致）")
        try:
            artifact_rel = str(art_path.relative_to(ROOT))
        except ValueError:
            artifact_rel = str(art_path)
        cost = meta.get("cost") or {}
        if isinstance(cost, Cost):
            cost = cost.to_dict()
        rec = {
            "version": version,
            "created_t": int(meta.get("created_t", 0)),
            "parent": (None if meta.get("parent") is None else int(meta["parent"])),
            "kind": str(meta.get("kind", "candidate")),
            "playbook": str(meta.get("playbook", "")),
            "config": dict(meta.get("config", {}) or {}),
            "metrics": dict(meta.get("metrics", {}) or {}),
            "status": str(meta.get("status", "candidate")),
            "artifact": artifact_rel,
            "cost": {k: float(cost.get(k, 0)) if k != "labels" else int(cost.get(k, 0))
                     for k in ("compute_units", "labels", "tokens", "wall_s")},
            "lineage": list(meta.get("lineage", []) or []),
            "note": str(meta.get("note", "")),
            "n_classes": int(getattr(model, "n_classes", self.base_n_classes)),
            "n_features": int(self.n_features),
            "created_at": now_ts(),
        }
        self.state["versions"].append(rec)
        self.state["next_version"] = version + 1
        self._event("add", {"version": version, "playbook": rec["playbook"],
                            "kind": rec["kind"], "parent": rec["parent"]})
        self._save()
        return self._to_version(rec)

    def get(self, version: int) -> tuple[ModelVersion, MLP]:
        rec = self._meta(version)
        art = rec.get("artifact", "")
        p = Path(art)
        if not p.is_absolute():
            p = ROOT / p
        model = MLP.load(str(p), l2=self._l2)
        return self._to_version(rec), model

    def champion(self) -> tuple[ModelVersion, MLP]:
        v = self.state.get("champion")
        if v is None:
            raise RuntimeError("registry 还没有 champion，请先 add()+promote()")
        return self.get(int(v))

    def history(self) -> list[ModelVersion]:
        return [self._to_version(d) for d in
                sorted(self.state["versions"], key=lambda x: int(x["version"]))]

    def n_versions(self) -> int:
        return len(self.state["versions"])

    # ------------------------------------------------------------------
    # 指针操作
    # ------------------------------------------------------------------
    def promote(self, version: int) -> ModelVersion:
        rec = self._meta(version)
        cur = self.state.get("champion")
        if cur is not None and int(cur) != int(version):
            self._meta(cur)["status"] = "archived"
            self._event("archive", {"version": int(cur)})
        rec["status"] = "champion"
        self.state["champion"] = int(version)
        self._event("promote", {"version": int(version),
                                "previous": None if cur is None else int(cur)})
        self._save()
        return self._to_version(rec)

    def rollback(self, to_version: int, reason: str = "") -> ModelVersion:
        target = self._meta(to_version)
        cur = self.state.get("champion")
        if cur is not None and int(cur) != int(to_version):
            self._meta(cur)["status"] = "rolled_back"
        target["status"] = "champion"
        self.state["champion"] = int(to_version)
        self._event("rollback", {"to_version": int(to_version),
                                 "from_version": None if cur is None else int(cur),
                                 "reason": str(reason)})
        self._save()
        return self._to_version(target)

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------
    def lineage_of(self, version: int) -> list[int]:
        chain, seen = [], set()
        cur = int(version)
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            try:
                cur = self._meta(cur).get("parent")
            except KeyError:
                break
            if cur is not None:
                cur = int(cur)
        return chain

    def summary(self) -> dict:
        metas = self.history()
        return {"n_versions": len(metas),
                "champion": self.state.get("champion"),
                "versions": [{"version": m.version, "status": m.status,
                              "playbook": m.playbook, "kind": m.kind,
                              "parent": m.parent,
                              "probe_f1": m.metrics.get("probe_macro_f1"),
                              "n_classes": self._meta(m.version).get("n_classes")}
                             for m in metas],
                "events": self.state.get("events", [])[-10:]}

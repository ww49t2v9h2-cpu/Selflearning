"""实验记录与可观测性：本地 SQLite/JSONL 镜像 + SwanLab 适配（SwanLab 兼容）。

设计目标
--------
1. **真用得上的记录系统**：闭环每一步（监测/诊断/实验/评估/上线/回滚）都写入本模块，
   单条指标写入 < 5 ms，离线、无网络、可复现、可审计；
2. **与 SwanLab 字段兼容**：`RunRecord`/`MetricPoint` 字段原样落库；SwanLab 只是
   "另一个后端"，默认 `mode=offline` 本地记录，一条命令 `swanlab sync` 即可上云；
3. **软依赖**：未安装 swanlab 时自动降级为 `LocalTracker`，并把降级原因写入
   `tracker_warnings.jsonl`，绝不因观测层故障中断主闭环；
4. **查询友好**：`query(filters)` 支持 group/kind/stage/phase/时间/标签等组合过滤，
   `export(fmt)` 导出 csv/json/md，供 `scripts/query_records.py` 与
   `scripts/gen_record_book.py`（《实验记录汇编》）复用。

目录约定（docs/03 §6）
--------------------
```
experiments/runs/
├── tracker.sqlite                 # 主索引（可查询）
├── records.jsonl                  # run 生命周期 JSONL 镜像（append-only）
├── tracker_warnings.jsonl         # 降级/异常告警
├── swanlog/                       # SwanLab 离线日志（mode=local/offline 时）
└── {run_id}/
    ├── record.json                # 最新 RunRecord + status/summary/swanlab 元信息
    ├── metrics.jsonl              # 指标流（每 step 一行）
    └── artifacts.json             # artifact 索引
experiments/artifacts/{run_id}/    # artifact 实体（step_name.ext）
```

公开 API（冻结，供 `src/loop.py`/`src/cli.py`/Harness 工具调用）
--------------------------------------------------------------
```python
tracker = make_tracker(cfg)                      # 按 cfg["tracker"]["backend"] 构建
run_id = tracker.start_run(RunRecord(...))
tracker.log_metrics(run_id, {"macro_f1": 0.82}, step=3, t=3)
tracker.log_artifact(run_id, Path("experiments/artifacts/x/model.npz"), step=3)
tracker.finish_run(run_id, {"best": 0.82}, status="finished")
runs = tracker.query({"group": "main", "kind": "experiment"})
tracker.export("md", Path("results/record_book/runs.md"))
```
"""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from ..core.spec import SCHEMA_VERSION, MetricPoint, RunRecord

# --------------------------------------------------------------------------
# 常量与工具函数
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT / "experiments" / "runs"
DEFAULT_ARTIFACTS_DIR = ROOT / "experiments" / "artifacts"

RUN_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*-\d{8}-\d{6}-[0-9a-f]{6}$")
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_.\u4e00-\u9fff-]+")

# SwanLab 合法 mode；不要在这里 import swanlab（软依赖）
SWANLAB_MODES = ("online", "local", "offline", "disabled")
BACKENDS = ("local", "swanlab", "both", "none")


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")


def _nonce(n: int = 6) -> str:
    """短随机 id（run_id 后缀）；可被测试注入替换。"""
    return os.urandom(n // 2 + 1).hex()[:n]


def make_run_id(group: str = "main", when: Optional[float] = None,
                nonce: Optional[str] = None) -> str:
    """生成约定格式 run_id：`{group}-{yyyyMMdd-HHMMSS}-{6hex}`（docs/03 §6）。"""
    g = re.sub(r"[^A-Za-z0-9_]", "-", str(group or "main")).strip("-") or "main"
    ts = datetime.fromtimestamp(_now() if when is None else float(when))
    return f"{g}-{ts.strftime('%Y%m%d-%H%M%S')}-{(nonce or _nonce(6))}"


def _json_default(obj: Any) -> Any:
    """JSON 兜底：numpy / Path / dataclass / set 等。"""
    if hasattr(obj, "tolist"):
        with contextlib.suppress(Exception):
            return obj.tolist()
    if hasattr(obj, "to_dict"):
        with contextlib.suppress(Exception):
            return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "__dict__"):
        with contextlib.suppress(Exception):
            return dict(obj.__dict__)
    return str(obj)


def jdumps(obj: Any, sort_keys: bool = True) -> str:
    """严格 JSON（禁 NaN，保证输出是合法 JSON，可被 SwanLab/前端消费）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys,
                      allow_nan=False, default=_json_default)


def jloads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any) -> Optional[float]:
    """把指标值转 float；bool→0/1；非法/非有限值返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    with contextlib.suppress(TypeError, ValueError):
        v = float(value)
        if v == v and v not in (float("inf"), float("-inf")):  # NaN/inf 过滤
            return v
    return None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _as_record(record: Any) -> RunRecord:
    """允许传 RunRecord 或 dict（loop 里两种写法都可能出现）。"""
    if isinstance(record, RunRecord):
        return record
    if isinstance(record, dict):
        known = set(RunRecord.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in record.items() if k in known}
        rec = RunRecord(**kwargs)
        # 未声明的字段（如 swanlab_url）作为附加信息保留
        extra = {k: v for k, v in record.items() if k not in known}
        if extra:
            rec.config = {**(rec.config or {}), "_extra": extra}
        return rec
    raise TypeError(f"start_run 需要 RunRecord 或 dict，收到 {type(record)!r}")


# --------------------------------------------------------------------------
# Tracker 协议
# --------------------------------------------------------------------------
class Tracker(Protocol):
    """实验记录后端协议（docs/03 §5.4）。"""

    def start_run(self, record: RunRecord) -> str: ...
    def log_metrics(self, run_id: str, metrics: dict, step: int, t: int = 0) -> None: ...
    def log_artifact(self, run_id: str, path: Path, step: int = 0) -> None: ...
    def finish_run(self, run_id: str, summary: dict, status: str = "finished") -> None: ...
    def query(self, filters: dict) -> list[dict]: ...
    def export(self, fmt: str, out: Path) -> Path: ...


# --------------------------------------------------------------------------
# 本地镜像 Tracker：SQLite + JSONL 双写
# --------------------------------------------------------------------------
_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    group_name      TEXT NOT NULL DEFAULT 'main',
    name            TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'loop',
    stage           TEXT NOT NULL DEFAULT 'system',
    config_json     TEXT NOT NULL DEFAULT '{}',
    summary_json    TEXT NOT NULL DEFAULT '{}',
    tags_json       TEXT NOT NULL DEFAULT '[]',
    t               INTEGER NOT NULL DEFAULT 0,
    phase           INTEGER NOT NULL DEFAULT 0,
    parent_run      TEXT,
    created_at      REAL NOT NULL DEFAULT 0,
    finished_at     REAL,
    status          TEXT NOT NULL DEFAULT 'running',
    swanlab_run_id  TEXT,
    swanlab_url     TEXT,
    swanlab_mode    TEXT,
    note            TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_group  ON runs(group_name);
CREATE INDEX IF NOT EXISTS idx_runs_kind   ON runs(kind);
CREATE INDEX IF NOT EXISTS idx_runs_stage  ON runs(stage);
CREATE INDEX IF NOT EXISTS idx_runs_phase  ON runs(phase);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE TABLE IF NOT EXISTS metrics (
    run_id  TEXT NOT NULL,
    key     TEXT NOT NULL,
    step    INTEGER NOT NULL,
    value   REAL NOT NULL,
    t       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, key, step)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE TABLE IF NOT EXISTS artifacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    step        INTEGER NOT NULL DEFAULT 0,
    name        TEXT NOT NULL DEFAULT '',
    src_path    TEXT NOT NULL DEFAULT '',
    stored_path TEXT NOT NULL DEFAULT '',
    size        INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(run_id, stored_path)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""

_QUERY_KEYS = {
    "run_id", "group", "kind", "stage", "phase", "t_min", "t_max",
    "since", "until", "created_after", "created_before", "status", "tag",
    "parent_run", "name_like", "swanlab_run_id", "limit", "offset", "order",
    "include_metrics", "include_artifacts", "include_config",
}

_EXPORT_COLUMNS = ["run_id", "group", "name", "kind", "stage", "status", "t",
                   "phase", "parent_run", "created_at", "finished_at", "tags",
                   "config", "summary", "metric_count", "artifact_count",
                   "swanlab_run_id", "swanlab_url", "note"]


class LocalTracker:
    """SQLite（查询）+ JSONL（可搬运/可 diff）双写的本地记录后端。

    - 单连接 + 可重入锁，UI/循环/脚本多线程调用安全；
    - WAL 模式 + 自动提交，单条 log_metrics 远低于 5 ms（4090 次/秒量级）；
    - 所有写入幂等（同一 run_id 重复 start/commit 不会炸）。
    """

    backend = "local"

    def __init__(self, root: Optional[str | Path] = None,
                 artifacts_dir: Optional[str | Path] = None,
                 db_path: Optional[str | Path] = None,
                 id_factory=None, clock=None):
        self.root = Path(root) if root else DEFAULT_RUNS_DIR
        self.root = self.root if self.root.is_absolute() else (ROOT / self.root)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = ROOT / self.artifacts_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.root / "tracker.sqlite"
        if not self.db_path.is_absolute():
            self.db_path = ROOT / self.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.records_jsonl = self.root / "records.jsonl"
        self.warnings_jsonl = self.root / "tracker_warnings.jsonl"
        self._lock = threading.RLock()
        self._id_factory = id_factory
        self._clock = clock or _now
        self.warnings: list[dict] = []
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # autocommit：每次写入独立、无需显式 commit
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_RUNS_DDL)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,))
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?)",
                (str(self._clock()),))

    # ------------------------------------------------------------------
    # 基础
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(Exception):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            with contextlib.suppress(Exception):
                self._conn.close()

    def _warn(self, kind: str, message: str, **extra) -> dict:
        item = {"kind": kind, "message": message, "ts": self._clock(), **extra}
        self.warnings.append(item)
        with contextlib.suppress(Exception):
            with self._lock:
                with open(self.warnings_jsonl, "a", encoding="utf-8") as f:
                    f.write(jdumps(item) + "\n")
        return item

    def health(self) -> dict:
        return {"backend": self.backend, "degraded": False,
                "root": str(self.root), "db": str(self.db_path),
                "warnings": list(self.warnings[-20:]),
                "schema_version": SCHEMA_VERSION}

    # ------------------------------------------------------------------
    # 写：run 生命周期
    # ------------------------------------------------------------------
    def start_run(self, record: RunRecord) -> str:
        rec = _as_record(record)
        with self._lock:
            if not rec.run_id:
                rec.run_id = (self._id_factory(rec.group) if self._id_factory
                              else make_run_id(rec.group, when=self._clock()))
            if not rec.created_at:
                rec.created_at = self._clock()
            run_dir = self.root / rec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.jsonl").touch()
            self._conn.execute(
                """INSERT INTO runs(run_id, group_name, name, kind, stage, config_json,
                                    summary_json, tags_json, t, phase, parent_run,
                                    created_at, status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'running')
                   ON CONFLICT(run_id) DO UPDATE SET
                     name=excluded.name, kind=excluded.kind, stage=excluded.stage,
                     config_json=excluded.config_json, tags_json=excluded.tags_json,
                     t=excluded.t, phase=excluded.phase,
                     parent_run=COALESCE(excluded.parent_run, runs.parent_run)""",
                (rec.run_id, rec.group, rec.name, rec.kind, rec.stage,
                 jdumps(rec.config or {}), jdumps(rec.summary or {}),
                 jdumps(rec.tags or []), int(rec.t), int(rec.phase),
                 rec.parent_run, float(rec.created_at)))
            self._write_record_json(rec.run_id)
            self._append_records_jsonl({"event": "start", "run_id": rec.run_id,
                                        "group": rec.group, "kind": rec.kind,
                                        "stage": rec.stage, "t": rec.t,
                                        "phase": rec.phase, "created_at": rec.created_at})
        return rec.run_id

    def update_run(self, run_id: str, patch: dict) -> dict:
        """局部更新 run 元信息（name/tags/config/summary/status/note/swanlab_*）。"""
        if not isinstance(patch, dict):
            raise TypeError("patch 需要 dict")
        allowed = {"name", "tags", "config", "summary", "status", "note",
                   "swanlab_run_id", "swanlab_url", "swanlab_mode", "parent_run",
                   "finished_at"}
        bad = set(patch) - allowed
        if bad:
            raise ValueError(f"update_run 不支持字段: {sorted(bad)}")
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                self._warn("update_unknown_run", f"update_run 未知 run_id={run_id}")
                return {}
            sets, vals = [], []
            for k, v in patch.items():
                if k in ("tags", "config", "summary"):
                    merge = (k == "summary") or (k == "config")
                    old = jloads(row[f"{k}_json"], {} if merge else [])
                    new = old if not isinstance(old, (dict, list)) else old
                    if merge and isinstance(new, dict) and isinstance(v, dict):
                        new = {**new, **v}
                    else:
                        new = v
                    sets.append(f"{k}_json=?")
                    vals.append(jdumps(new))
                elif k == "status":
                    sets.append("status=?"); vals.append(str(v))
                elif k == "finished_at":
                    sets.append("finished_at=?"); vals.append(float(v))
                else:
                    sets.append(f"{k}=?")
                    vals.append(v)
            vals.append(run_id)
            self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?", vals)
            self._write_record_json(run_id)
            return self.get_run(run_id) or {}

    def finish_run(self, run_id: str, summary: dict, status: str = "finished") -> None:
        if status not in ("finished", "failed", "crashed", "killed", "skipped"):
            self._warn("finish_bad_status", f"非标准 status={status}，按 finished 处理")
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                # 容错：finish 前没 start（loop 极端异常路径）→ 补一条最小记录
                rec = RunRecord(run_id=run_id, group="misc", name=run_id,
                                kind="loop", stage="system", created_at=self._clock())
                self.start_run(rec)
                row = self._conn.execute("SELECT * FROM runs WHERE run_id=?",
                                         (run_id,)).fetchone()
            old_summary = jloads(row["summary_json"], {}) or {}
            if not isinstance(old_summary, dict):
                old_summary = {"value": old_summary}
            merged = {**old_summary, **(summary or {})}
            ts = self._clock()
            self._conn.execute(
                "UPDATE runs SET summary_json=?, status=?, finished_at=? WHERE run_id=?",
                (jdumps(merged), status, ts, run_id))
            self._write_record_json(run_id)
            self._append_records_jsonl({"event": "finish", "run_id": run_id,
                                        "status": status, "finished_at": ts,
                                        "summary": merged})

    def _write_record_json(self, run_id: str) -> None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return
        record = {
            "run_id": row["run_id"],
            "group": row["group_name"],
            "name": row["name"],
            "kind": row["kind"],
            "stage": row["stage"],
            "config": jloads(row["config_json"], {}),
            "summary": jloads(row["summary_json"], {}),
            "tags": jloads(row["tags_json"], []),
            "t": row["t"],
            "phase": row["phase"],
            "parent_run": row["parent_run"],
            "created_at": row["created_at"],
            "created_at_iso": _iso(row["created_at"]) if row["created_at"] else None,
            "finished_at": row["finished_at"],
            "finished_at_iso": _iso(row["finished_at"]) if row["finished_at"] else None,
            "status": row["status"],
            "swanlab_run_id": row["swanlab_run_id"],
            "swanlab_url": row["swanlab_url"],
            "swanlab_mode": row["swanlab_mode"],
            "note": row["note"],
            "artifacts": self._artifacts(run_id),
            "schema_version": SCHEMA_VERSION,
        }
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(run_dir / "record.json", jdumps(record) + "\n")

    def _append_records_jsonl(self, item: dict) -> None:
        with contextlib.suppress(Exception):
            with open(self.records_jsonl, "a", encoding="utf-8") as f:
                f.write(jdumps(item) + "\n")

    # ------------------------------------------------------------------
    # 写：指标 / artifact
    # ------------------------------------------------------------------
    def log_metrics(self, run_id: str, metrics: dict, step: int, t: int = 0) -> None:
        if not isinstance(metrics, dict) or not metrics:
            return
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM runs WHERE run_id=?",
                                     (run_id,)).fetchone()
            if row is None:
                self._auto_stub_run(run_id)
            cleaned: dict[str, float] = {}
            dropped = 0
            for k, v in metrics.items():
                fv = _coerce_float(v)
                if fv is None:
                    dropped += 1
                else:
                    cleaned[str(k)] = fv
            if not cleaned:
                return
            step_i, t_i = int(step), int(t)
            self._conn.executemany(
                "INSERT OR REPLACE INTO metrics(run_id,key,step,value,t) VALUES(?,?,?,?,?)",
                [(run_id, k, step_i, v, t_i) for k, v in cleaned.items()])
            line = {"run_id": run_id, "step": step_i, "t": t_i,
                    "metrics": cleaned, "ts": self._clock()}
            with open(self.root / run_id / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(jdumps(line) + "\n")
            if dropped:
                self._warn("metrics_dropped", f"{run_id} step={step_i} 丢弃 {dropped} 个非数值指标",
                           run_id=run_id)

    def log_metrics_batch(self, run_id: str, rows: Iterable[dict]) -> int:
        """批量写：rows = [{"metrics": {...}, "step": int, "t": int}, ...]。"""
        n = 0
        for row in rows:
            if isinstance(row, MetricPoint):
                self.log_metrics(run_id, {row.key: row.value}, row.step, row.t)
            else:
                self.log_metrics(run_id, row.get("metrics", {}), row.get("step", 0),
                                 row.get("t", 0))
            n += 1
        return n

    def log_artifact(self, run_id: str, path: str | Path, step: int = 0,
                     name: Optional[str] = None, copy_file: bool = True) -> str:
        src = Path(path)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists() or not src.is_file():
            self._warn("artifact_missing", f"artifact 不存在: {src}")
            return ""
        with self._lock:
            if self._conn.execute("SELECT 1 FROM runs WHERE run_id=?",
                                  (run_id,)).fetchone() is None:
                self._auto_stub_run(run_id)
            safe = name or _SAFE_NAME_RE.sub("_", src.name)
            dest_dir = self.artifacts_dir / run_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{int(step)}_{safe}"
            if copy_file and src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            stored = dest if copy_file else src
            try:
                size = stored.stat().st_size
                digest = _sha256_file(stored)
            except OSError as exc:
                self._warn("artifact_stat_failed", str(exc))
                size, digest = 0, ""
            self._conn.execute(
                """INSERT OR REPLACE INTO artifacts(run_id, step, name, src_path,
                        stored_path, size, sha256, created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (run_id, int(step), safe, str(src), str(stored), int(size),
                 digest, self._clock()))
            self._write_artifacts_json(run_id)
            self._append_records_jsonl({"event": "artifact", "run_id": run_id,
                                        "step": int(step), "name": safe,
                                        "path": str(stored), "size": size,
                                        "sha256": digest})
            return str(stored)

    def _write_artifacts_json(self, run_id: str) -> None:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(run_dir / "artifacts.json", jdumps(self._artifacts(run_id)) + "\n")

    def _artifacts(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT step,name,src_path,stored_path,size,sha256,created_at "
            "FROM artifacts WHERE run_id=? ORDER BY step, id", (run_id,)).fetchall()
        return [{"step": r["step"], "name": r["name"], "src_path": r["src_path"],
                 "path": r["stored_path"], "size": r["size"],
                 "sha256": r["sha256"], "created_at": r["created_at"]} for r in rows]

    def _auto_stub_run(self, run_id: str) -> None:
        group = str(run_id).split("-")[0] or "misc"
        self.start_run(RunRecord(run_id=run_id, group=group, name=run_id,
                                 kind="loop", stage="system",
                                 created_at=self._clock()))
        self._warn("auto_stub_run", f"log 前未 start_run，已补建记录 run_id={run_id}")

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def get_run(self, run_id: str, include_metrics: bool = False) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?",
                                     (run_id,)).fetchone()
            if row is None:
                return None
            item = self._row_to_dict(row)
            item["artifacts"] = self._artifacts(run_id)
            item["metrics_summary"] = self.latest_metrics(run_id)
            item["metric_count"] = int(self._conn.execute(
                "SELECT COUNT(*) AS c FROM metrics WHERE run_id=?", (run_id,)
            ).fetchone()["c"])
            if include_metrics:
                item["metrics"] = self.metrics(run_id)
            return item

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "run_id": row["run_id"],
            "group": row["group_name"],
            "name": row["name"],
            "kind": row["kind"],
            "stage": row["stage"],
            "config": jloads(row["config_json"], {}),
            "summary": jloads(row["summary_json"], {}),
            "tags": jloads(row["tags_json"], []),
            "t": row["t"],
            "phase": row["phase"],
            "parent_run": row["parent_run"],
            "created_at": row["created_at"],
            "created_at_iso": _iso(row["created_at"]) if row["created_at"] else None,
            "finished_at": row["finished_at"],
            "finished_at_iso": _iso(row["finished_at"]) if row["finished_at"] else None,
            "status": row["status"],
            "swanlab_run_id": row["swanlab_run_id"],
            "swanlab_url": row["swanlab_url"],
            "swanlab_mode": row["swanlab_mode"],
            "note": row["note"],
        }

    def query(self, filters: Optional[dict] = None) -> list[dict]:
        filters = dict(filters or {})
        unknown = set(filters) - _QUERY_KEYS
        if unknown:
            raise ValueError(f"query 不支持的过滤键: {sorted(unknown)}；"
                             f"支持: {sorted(_QUERY_KEYS)}")
        include_metrics = bool(filters.get("include_metrics", False))
        include_artifacts = bool(filters.get("include_artifacts", False))
        include_config = bool(filters.get("include_config", True))
        limit = filters.get("limit")
        offset = int(filters.get("offset", 0) or 0)
        order = str(filters.get("order", "desc")).lower()
        if order not in ("asc", "desc"):
            raise ValueError("order 只能是 asc/desc")

        where, params = self._build_where(filters)
        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY created_at {order.upper()}, run_id {order.upper()}"
        sql += " LIMIT ? OFFSET ?"
        params.extend([int(limit) if limit is not None else -1, offset])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            out = []
            for row in rows:
                item = self._row_to_dict(row)
                if not include_config:
                    item["config"] = {}
                item["metric_count"] = int(self._conn.execute(
                    "SELECT COUNT(*) AS c FROM metrics WHERE run_id=?",
                    (row["run_id"],)).fetchone()["c"])
                if include_metrics:
                    item["metrics_summary"] = self.latest_metrics(row["run_id"])
                if include_artifacts:
                    item["artifacts"] = self._artifacts(row["run_id"])
                out.append(item)
            return out

    def _build_where(self, f: dict) -> tuple[list[str], list]:
        where: list[str] = []
        params: list = []
        for key, col in (("run_id", "run_id"), ("group", "group_name"),
                         ("kind", "kind"), ("stage", "stage"),
                         ("parent_run", "parent_run"), ("status", "status"),
                         ("swanlab_run_id", "swanlab_run_id")):
            if f.get(key) is None:
                continue
            val = f[key]
            if isinstance(val, (list, tuple, set)):
                vals = list(val)
                if not vals:
                    where.append("1=0")
                    continue
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                params.extend(vals)
            else:
                where.append(f"{col}=?")
                params.append(val)
        if f.get("phase") is not None:
            val = f["phase"]
            if isinstance(val, (list, tuple, set)):
                vals = [int(v) for v in val]
                if vals:
                    where.append(f"phase IN ({','.join('?' * len(vals))})")
                    params.extend(vals)
            else:
                where.append("phase=?")
                params.append(int(val))
        if f.get("t_min") is not None:
            where.append("t>=?"); params.append(int(f["t_min"]))
        if f.get("t_max") is not None:
            where.append("t<=?"); params.append(int(f["t_max"]))
        for key, op in (("since", ">="), ("created_after", ">="),
                        ("until", "<="), ("created_before", "<=")):
            if f.get(key) is None:
                continue
            ts = _to_epoch(f[key])
            where.append(f"created_at {op} ?")
            params.append(ts)
        if f.get("tag"):
            tags = f["tag"] if isinstance(f["tag"], (list, tuple, set)) else [f["tag"]]
            for tag in tags:
                where.append("tags_json LIKE ?")
                params.append(f'%"{str(tag)}"%')
        if f.get("name_like"):
            where.append("name LIKE ?")
            params.append(f"%{f['name_like']}%")
        return where, params

    def metrics(self, run_id: str, keys: Optional[Sequence[str]] = None,
                since_step: Optional[int] = None) -> list[dict]:
        sql = "SELECT run_id,key,step,value,t FROM metrics WHERE run_id=?"
        params: list = [run_id]
        if keys:
            keys = list(keys)
            sql += f" AND key IN ({','.join('?' * len(keys))})"
            params.extend(keys)
        if since_step is not None:
            sql += " AND step>=?"
            params.append(int(since_step))
        sql += " ORDER BY step ASC, key ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def latest_metrics(self, run_id: str,
                       keys: Optional[Sequence[str]] = None) -> dict[str, float]:
        with self._lock:
            sql = ("SELECT m.key, m.value FROM metrics m JOIN "
                   "(SELECT key, MAX(step) AS ms FROM metrics WHERE run_id=? "
                   "GROUP BY key) x ON m.key=x.key AND m.step=x.ms "
                   "WHERE m.run_id=?")
            params: list = [run_id, run_id]
            if keys:
                keys = list(keys)
                sql += f" AND m.key IN ({','.join('?' * len(keys))})"
                params.extend(keys)
            rows = self._conn.execute(sql, params).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def summary(self, run_id: str) -> Optional[dict]:
        row = self.get_run(run_id)
        return None if row is None else row.get("summary", {})

    def stats(self) -> dict:
        with self._lock:
            n_runs = self._conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
            n_metrics = self._conn.execute("SELECT COUNT(*) AS c FROM metrics").fetchone()["c"]
            n_artifacts = self._conn.execute("SELECT COUNT(*) AS c FROM artifacts").fetchone()["c"]
            by_group = {r["group_name"]: r["c"] for r in self._conn.execute(
                "SELECT group_name, COUNT(*) AS c FROM runs GROUP BY group_name").fetchall()}
            by_kind = {r["kind"]: r["c"] for r in self._conn.execute(
                "SELECT kind, COUNT(*) AS c FROM runs GROUP BY kind").fetchall()}
        return {"n_runs": int(n_runs), "n_metrics": int(n_metrics),
                "n_artifacts": int(n_artifacts), "by_group": by_group,
                "by_kind": by_kind, "root": str(self.root)}

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export(self, fmt: str, out: str | Path,
               filters: Optional[dict] = None) -> Path:
        fmt = str(fmt).lower().lstrip(".")
        if fmt not in ("csv", "json", "md", "markdown"):
            raise ValueError(f"export 支持 csv|json|md，收到 {fmt!r}")
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filters = dict(filters or {})
        filters.setdefault("include_metrics", True)
        filters.setdefault("include_artifacts", True)
        runs = self.query(filters)
        if fmt == "json":
            payload = {"schema_version": SCHEMA_VERSION, "exported_at": _iso(self._clock()),
                       "backend": self.backend, "n": len(runs),
                       "filters": {k: v for k, v in filters.items() if k != "include_metrics"},
                       "runs": runs}
            _atomic_write(out_path, jdumps(payload, sort_keys=False) + "\n")
            return out_path
        rows = [self._export_row(r) for r in runs]
        if fmt == "csv":
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_EXPORT_COLUMNS, restval="")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return out_path
        # markdown
        lines = [f"# 实验记录导出（{self.backend}）", "",
                 f"- 导出时间：{_iso(self._clock())}",
                 f"- 记录数：{len(runs)}",
                 f"- 过滤条件：`{jdumps({k: v for k, v in filters.items() if k not in ('include_metrics', 'include_artifacts')})}`",
                 "", "| " + " | ".join(_EXPORT_COLUMNS) + " |",
                 "|" + "|".join(["---"] * len(_EXPORT_COLUMNS)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|")
                                           for c in _EXPORT_COLUMNS) + " |")
        _atomic_write(out_path, "\n".join(lines) + "\n")
        return out_path

    def _export_row(self, run: dict) -> dict:
        return {
            "run_id": run.get("run_id", ""),
            "group": run.get("group", ""),
            "name": run.get("name", ""),
            "kind": run.get("kind", ""),
            "stage": run.get("stage", ""),
            "status": run.get("status", ""),
            "t": run.get("t", 0),
            "phase": run.get("phase", 0),
            "parent_run": run.get("parent_run") or "",
            "created_at": run.get("created_at_iso") or run.get("created_at", ""),
            "finished_at": (run.get("finished_at") or ""),
            "tags": ",".join(map(str, run.get("tags") or [])),
            "config": jdumps(run.get("config") or {}),
            "summary": jdumps(run.get("summary") or {}),
            "metric_count": run.get("metric_count", 0),
            "artifact_count": len(run.get("artifacts") or []),
            "swanlab_run_id": run.get("swanlab_run_id") or "",
            "swanlab_url": run.get("swanlab_url") or "",
            "note": run.get("note", ""),
        }


def _to_epoch(value: Any) -> float:
    """接受 epoch 秒 / 秒级数字字符串 / ISO8601（含 'Z'）/ 'YYYYmmdd-HHMMSS'。"""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    with contextlib.suppress(ValueError):
        return float(text)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d-%H%M%S",
                "%Y%m%d", "%Y-%m-%d"):
        with contextlib.suppress(ValueError):
            dt = datetime.strptime(text.rstrip("Z"), fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
    raise ValueError(f"无法解析时间: {value!r}（支持 epoch 秒或 ISO8601）")


# --------------------------------------------------------------------------
# SwanLab 适配器
# --------------------------------------------------------------------------
class SwanLabTracker:
    """SwanLab 兼容后端：本地镜像 + SwanLab（online/local/offline/disabled）。

    SwanLab 的 `init` 是**进程级单例**（同一时刻只有一条活跃 run），因此本适配器：

    - `start_run` 的第一条 run 打开 SwanLab 会话（"主 run"）；
    - 之后启动的子 run 默认 **挂载** 到同一会话，指标以 `{run_id}/{metric}` 命名空间
      记录（SwanLab 里能看到全部迭代实验曲线，本地镜像里每个 run 独立可查）；
    - `attach_subruns=False` 时切换为严格 1:1 模式（前一条未 finish 会被自动收尾），
      适合"每次实验一个 SwanLab run"的展示场景；
    - 任一 SwanLab 调用失败 → `degraded=True`，后续只写本地镜像并把告警写入
      `tracker_warnings.jsonl`；主闭环无感。

    安装与上云（官方评测环境/有网时）：
    ```bash
    pip install swanlab
    swanlab login                       # 云端模式需要 API Key
    # 本系统：cfg.tracker.backend=swanlab, cfg.tracker.swanlab_mode=offline 或 online
    swanlab sync experiments/runs/swanlog    # 离线记录一条命令上云
    swanlab watch experiments/runs/swanlog   # 本地看板（local/offline 均可）
    ```
    """

    backend = "swanlab"

    def __init__(self, mode: str = "offline", project: str = "zhixun-engine",
                 logdir: Optional[str | Path] = None,
                 mirror: Optional[LocalTracker] = None,
                 root: Optional[str | Path] = None,
                 artifacts_dir: Optional[str | Path] = None,
                 swanlab_module: Any = None,
                 attach_subruns: bool = True,
                 enabled: bool = True):
        self.mode = str(mode or "offline").lower()
        if self.mode not in SWANLAB_MODES:
            self.mode = "offline"
        self.project = project or "zhixun-engine"
        self.mirror = mirror or LocalTracker(root=root, artifacts_dir=artifacts_dir)
        self.logdir = Path(logdir) if logdir else (self.mirror.root / "swanlog")
        if not self.logdir.is_absolute():
            self.logdir = ROOT / self.logdir
        self.attach_subruns = bool(attach_subruns)
        self.enabled = bool(enabled) and self.mode != "disabled"
        self.degraded = False
        self.warnings: list[dict] = []
        self._sw = None          # swanlab 模块
        self._run = None         # SwanLabRun 对象
        self._active_run_id: Optional[str] = None
        self._run_map: dict[str, dict] = {}   # local run_id -> swan 元信息
        self._lock = threading.RLock()
        if self.enabled:
            self._sw = swanlab_module if swanlab_module is not None else self._import_swanlab()
            if self._sw is None:
                self._degrade("swanlab_missing",
                              "未安装 swanlab（pip install swanlab），已自动降级为 LocalTracker；"
                              "全部记录仍写入本地 SQLite/JSONL，可安装后重放/上云")
        elif self.mode == "disabled":
            self.warnings.append({"kind": "swanlab_disabled",
                                  "message": "swanlab_mode=disabled，仅写本地镜像", "ts": _now()})

    # -- 初始化 / 降级 -------------------------------------------------
    @staticmethod
    def _import_swanlab():
        try:  # 软依赖：允许完全没装
            import swanlab  # type: ignore
            return swanlab
        except Exception:  # pragma: no cover - 正常情况走不到（本机未装）
            return None

    def _degrade(self, kind: str, message: str) -> None:
        self.degraded = True
        item = {"kind": kind, "message": message, "ts": _now()}
        self.warnings.append(item)
        with contextlib.suppress(Exception):
            self.mirror._warn(kind, message)
        self._run = None
        self._active_run_id = None
        self._sw = None

    def _try(self, action: str, fn, *args, **kwargs):
        """执行 SwanLab 调用；任何异常都降级但不影响主流程。"""
        if not self.enabled or self.degraded or self._sw is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 观测层不允许打断主闭环
            self._degrade(f"swanlab_{action}_failed",
                          f"SwanLab {action} 失败（{type(exc).__name__}: {exc}），"
                          f"已降级为本地记录")
            return None

    # -- SwanLab 元信息 ------------------------------------------------
    @staticmethod
    def _swan_meta(run: Any) -> dict:
        """兼容 SwanLab 不同版本（0.4~0.8+），尽力提取 run id / url / 目录。"""
        meta: dict = {"run_id": "", "url": "", "project": "", "run_dir": "",
                      "swanlog_dir": ""}
        public = getattr(run, "public", None)
        rid = getattr(public, "run_id", None) if public is not None else None
        if not rid:
            rid = getattr(run, "id", None) or getattr(run, "run_id", None)
        meta["run_id"] = str(rid or "")
        if public is not None:
            meta["project"] = str(getattr(public, "project_name", "") or "")
            meta["run_dir"] = str(getattr(public, "run_dir", "") or "")
            meta["swanlog_dir"] = str(getattr(public, "swanlog_dir", "") or "")
            cloud = getattr(public, "cloud", None)
            url = getattr(cloud, "experiment_url", "") if cloud is not None else ""
            url = url or getattr(public, "experiment_url", "") or ""
            meta["url"] = str(url or "")
        meta["url"] = meta["url"] or str(getattr(run, "url", "") or "")
        return meta

    def _open_session(self, record: RunRecord) -> None:
        if self._sw is None or self.degraded:
            return
        self.logdir.mkdir(parents=True, exist_ok=True)
        cfg_small = {k: v for k, v in (record.config or {}).items()
                     if isinstance(v, (int, float, str, bool))}
        kwargs = dict(project=self.project,
                      experiment_name=record.name or record.run_id,
                      config=cfg_small or {"run_id": record.run_id},
                      logdir=str(self.logdir),
                      mode=self.mode,
                      tags=[str(t) for t in (record.tags or [])][:8])
        run = self._try("init", self._sw.init, **kwargs)
        if run is None and not self.degraded:
            # 兼容极老版本：tags 参数不存在
            kwargs.pop("tags", None)
            run = self._try("init", self._sw.init, **kwargs)
        if run is None:
            return
        self._run = run
        self._active_run_id = record.run_id
        meta = self._swan_meta(run)
        meta["mode"] = self.mode
        meta["attached"] = False
        meta["namespace"] = ""
        meta["logdir"] = str(self.logdir)
        self._run_map[record.run_id] = meta
        self._write_back(record.run_id, meta)

    def _write_back(self, run_id: str, meta: dict) -> None:
        with contextlib.suppress(Exception):
            self.mirror.update_run(run_id, {
                "swanlab_run_id": meta.get("run_id") or "",
                "swanlab_url": meta.get("url") or "",
                "swanlab_mode": meta.get("mode") or self.mode,
            })

    def _meta_for(self, run_id: str) -> Optional[dict]:
        meta = self._run_map.get(run_id)
        if meta:
            return meta
        if self._active_run_id and self.attach_subruns:
            # 挂载：共享主 run 的 swan id，指标加命名空间
            owner = self._run_map.get(self._active_run_id, {})
            meta = dict(owner)
            meta["attached"] = True
            meta["namespace"] = f"{run_id}/"
            self._run_map[run_id] = meta
            self._write_back(run_id, meta)
            return meta
        return None

    # -- Tracker 协议 --------------------------------------------------
    def start_run(self, record: RunRecord) -> str:
        run_id = self.mirror.start_run(record)
        with self._lock:
            if not self.enabled:
                return run_id
            if self.degraded:
                self._write_back_degraded(run_id)
                return run_id
            if self._run is None:
                rec = getattr(self.mirror, "get_run", lambda *_: None)(run_id) or {}
                pseudo = _as_record(record)
                pseudo.run_id = run_id
                pseudo.name = rec.get("name") or pseudo.name or run_id
                self._open_session(pseudo)
            else:
                if not self.attach_subruns:
                    # 严格 1:1：先收尾上一条，再开启新 run
                    self._finish_swan_session()
                    pseudo = _as_record(record)
                    pseudo.run_id = run_id
                    self._open_session(pseudo)
                else:
                    self._meta_for(run_id)
        return run_id

    def _write_back_degraded(self, run_id: str) -> None:
        with contextlib.suppress(Exception):
            self.mirror.update_run(run_id, {"swanlab_mode": self.mode,
                                            "swanlab_run_id": "",
                                            "swanlab_url": ""})

    def log_metrics(self, run_id: str, metrics: dict, step: int, t: int = 0) -> None:
        self.mirror.log_metrics(run_id, metrics, step, t)
        if not self.enabled or self.degraded or self._sw is None or self._run is None:
            return
        with self._lock:
            meta = self._meta_for(run_id)
            if meta is None:
                return
            ns = meta.get("namespace", "")
            cleaned = {f"{ns}{k}": v for k, v in metrics.items()
                       if _coerce_float(v) is not None}
            if not cleaned:
                return
            self._try("log", self._sw.log, cleaned, step=int(step))

    def log_artifact(self, run_id: str, path: str | Path, step: int = 0,
                     name: Optional[str] = None) -> str:
        stored = self.mirror.log_artifact(run_id, path, step=step, name=name)
        if self.enabled and not self.degraded and self._run is not None and stored:
            meta = self._meta_for(run_id) or {}
            ns = meta.get("namespace", "")
            with contextlib.suppress(Exception):
                # SwanLab 基础版本没有稳定 artifact API：以摘要指标记录（本地索引为权威）
                self._sw.log({f"{ns}_artifact_count": 1.0}, step=int(step))
        return stored

    def _finish_swan_session(self) -> None:
        if self._sw is not None and self._run is not None and not self.degraded:
            self._try("finish", self._sw.finish)
        self._run = None
        self._active_run_id = None

    def finish_run(self, run_id: str, summary: dict, status: str = "finished") -> None:
        self.mirror.finish_run(run_id, summary, status=status)
        with self._lock:
            if self.enabled and not self.degraded and self._run is not None:
                if run_id == self._active_run_id:
                    self._finish_swan_session()
                elif self._meta_for(run_id):
                    # 挂载 run：以命名空间指标记录一句总结开关（可追溯）
                    meta = self._run_map.get(run_id, {})
                    with contextlib.suppress(Exception):
                        self._sw.log({f"{meta.get('namespace', '')}_finished": 1.0}, step=0)

    def query(self, filters: Optional[dict] = None) -> list[dict]:
        return self.mirror.query(filters)

    def export(self, fmt: str, out: str | Path,
               filters: Optional[dict] = None) -> Path:
        return self.mirror.export(fmt, out, filters)

    # -- 便捷代理 ------------------------------------------------------
    def get_run(self, *args, **kwargs):
        return self.mirror.get_run(*args, **kwargs)

    def metrics(self, *args, **kwargs):
        return self.mirror.metrics(*args, **kwargs)

    def latest_metrics(self, *args, **kwargs):
        return self.mirror.latest_metrics(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.mirror.summary(*args, **kwargs)

    def stats(self):
        return self.mirror.stats()

    def update_run(self, *args, **kwargs):
        return self.mirror.update_run(*args, **kwargs)

    def close(self) -> None:
        with self._lock:
            self._finish_swan_session()
            self.mirror.close()

    def health(self) -> dict:
        installed = False
        with contextlib.suppress(Exception):
            import importlib.util
            installed = importlib.util.find_spec("swanlab") is not None
        return {
            "backend": self.backend,
            "mode": self.mode,
            "degraded": bool(self.degraded),
            "swanlab_installed": installed or (self._sw is not None),
            "project": self.project,
            "logdir": str(self.logdir),
            "active_run_id": self._active_run_id,
            "swanlab_runs": copy.deepcopy(self._run_map),
            "warnings": list(self.warnings[-20:]),
            "mirror": self.mirror.health(),
        }


# --------------------------------------------------------------------------
# 多后端与空后端
# --------------------------------------------------------------------------
class MultiTracker:
    """把一次记录扇出到多个后端；单后端故障不影响其它后端。"""

    backend = "multi"

    def __init__(self, trackers: Sequence[Any]):
        self.trackers = [t for t in trackers if t is not None]
        if not self.trackers:
            raise ValueError("MultiTracker 至少需要一个后端")
        self._lock = threading.RLock()

    def start_run(self, record: RunRecord) -> str:
        rec = _as_record(record)
        run_id = rec.run_id
        for i, tracker in enumerate(self.trackers):
            try:
                payload = rec if not run_id else _replace_run_id(rec, run_id)
                res = tracker.start_run(payload)
                run_id = run_id or res
            except Exception as exc:  # noqa: BLE001
                self._record_failure(i, "start_run", exc)
        return run_id

    def log_metrics(self, run_id: str, metrics: dict, step: int, t: int = 0) -> None:
        for i, tracker in enumerate(self.trackers):
            try:
                tracker.log_metrics(run_id, metrics, step, t)
            except Exception as exc:  # noqa: BLE001
                self._record_failure(i, "log_metrics", exc)

    def log_artifact(self, run_id: str, path: str | Path, step: int = 0,
                     name: Optional[str] = None) -> str:
        stored = ""
        for i, tracker in enumerate(self.trackers):
            try:
                res = tracker.log_artifact(run_id, path, step=step, name=name)
                stored = stored or (res or "")
            except Exception as exc:  # noqa: BLE001
                self._record_failure(i, "log_artifact", exc)
        return stored

    def finish_run(self, run_id: str, summary: dict, status: str = "finished") -> None:
        for i, tracker in enumerate(self.trackers):
            try:
                tracker.finish_run(run_id, summary, status=status)
            except Exception as exc:  # noqa: BLE001
                self._record_failure(i, "finish_run", exc)

    def _record_failure(self, idx: int, action: str, exc: Exception) -> None:
        item = {"tracker_index": idx, "action": action,
                "error": f"{type(exc).__name__}: {exc}", "ts": _now()}
        with self._lock:
            with contextlib.suppress(Exception):
                setattr(self.trackers[idx], "warnings",
                        list(getattr(self.trackers[idx], "warnings", [])) + [item])

    def query(self, filters: Optional[dict] = None) -> list[dict]:
        merged: dict[str, dict] = {}
        for i, tracker in enumerate(self.trackers):
            try:
                for run in tracker.query(filters):
                    rid = run.get("run_id")
                    if rid and rid not in merged:
                        merged[rid] = run
            except Exception as exc:  # noqa: BLE001
                self._record_failure(i, "query", exc)
        runs = list(merged.values())
        order = str((filters or {}).get("order", "desc")).lower()
        runs.sort(key=lambda r: (r.get("created_at") or 0, r.get("run_id") or ""),
                  reverse=(order != "asc"))
        limit = (filters or {}).get("limit")
        if limit is not None:
            runs = runs[:int(limit)]
        return runs

    def export(self, fmt: str, out: str | Path,
               filters: Optional[dict] = None) -> Path:
        for tracker in self.trackers:
            try:
                return tracker.export(fmt, out, filters)
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("所有后端 export 均失败")

    def health(self) -> dict:
        return {"backend": "multi",
                "trackers": [t.health() if hasattr(t, "health") else {"backend": str(t)}
                             for t in self.trackers]}

    def get_run(self, run_id: str, include_metrics: bool = False):
        for tracker in self.trackers:
            with contextlib.suppress(Exception):
                res = tracker.get_run(run_id, include_metrics=include_metrics)
                if res:
                    return res
        return None

    def metrics(self, *args, **kwargs):
        for tracker in self.trackers:
            with contextlib.suppress(Exception):
                return tracker.metrics(*args, **kwargs)
        return []

    def latest_metrics(self, *args, **kwargs):
        for tracker in self.trackers:
            with contextlib.suppress(Exception):
                return tracker.latest_metrics(*args, **kwargs)
        return {}

    def summary(self, *args, **kwargs):
        for tracker in self.trackers:
            with contextlib.suppress(Exception):
                return tracker.summary(*args, **kwargs)
        return None

    def close(self) -> None:
        for tracker in self.trackers:
            with contextlib.suppress(Exception):
                tracker.close()


def _replace_run_id(rec: RunRecord, run_id: str) -> RunRecord:
    clone = copy.copy(rec)
    clone.run_id = run_id
    return clone


class NullTracker:
    """backend=none：内存最小实现，接口不报错（用于极端关闭观测场景）。"""

    backend = "none"

    def __init__(self):
        self._runs: dict[str, dict] = {}
        self._metrics: dict[str, list[dict]] = {}

    def start_run(self, record: RunRecord) -> str:
        rec = _as_record(record)
        rid = rec.run_id or make_run_id(rec.group)
        self._runs[rid] = {"run_id": rid, "group": rec.group, "name": rec.name,
                           "kind": rec.kind, "stage": rec.stage,
                           "created_at": rec.created_at or _now(), "tags": rec.tags,
                           "config": rec.config, "summary": {}, "status": "running",
                           "t": rec.t, "phase": rec.phase, "parent_run": rec.parent_run,
                           "swanlab_run_id": None, "swanlab_url": None,
                           "swanlab_mode": None, "note": "", "finished_at": None}
        self._metrics.setdefault(rid, [])
        return rid

    def log_metrics(self, run_id: str, metrics: dict, step: int, t: int = 0) -> None:
        for k, v in (metrics or {}).items():
            fv = _coerce_float(v)
            if fv is not None:
                self._metrics.setdefault(run_id, []).append(
                    {"run_id": run_id, "key": str(k), "step": int(step),
                     "value": fv, "t": int(t)})

    def log_artifact(self, run_id: str, path, step: int = 0, name=None) -> str:
        return str(path)

    def finish_run(self, run_id: str, summary: dict, status: str = "finished") -> None:
        run = self._runs.setdefault(run_id, {"run_id": run_id, "group": "misc",
                                             "name": run_id, "kind": "loop",
                                             "stage": "system", "created_at": _now(),
                                             "tags": [], "config": {}, "t": 0,
                                             "phase": 0, "parent_run": None,
                                             "swanlab_run_id": None,
                                             "swanlab_url": None})
        run["summary"] = {**run.get("summary", {}), **(summary or {})}
        run["status"] = status
        run["finished_at"] = _now()

    def query(self, filters: Optional[dict] = None) -> list[dict]:
        return copy.deepcopy(list(self._runs.values()))

    def export(self, fmt: str, out, filters: Optional[dict] = None) -> Path:
        path = Path(out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(jdumps({"backend": "none", "runs": self.query(filters)},
                               sort_keys=False) + "\n", encoding="utf-8")
        return path

    def metrics(self, run_id: str, *args, **kwargs):
        return copy.deepcopy(self._metrics.get(run_id, []))

    def latest_metrics(self, run_id: str, keys=None) -> dict:
        out = {}
        for row in self._metrics.get(run_id, []):
            if keys and row["key"] not in keys:
                continue
            out[row["key"]] = row["value"]
        return out

    def summary(self, run_id: str):
        return self._runs.get(run_id, {}).get("summary", {})

    def get_run(self, run_id: str, include_metrics: bool = False):
        run = copy.deepcopy(self._runs.get(run_id))
        if run and include_metrics:
            run["metrics"] = self.metrics(run_id)
        return run

    def update_run(self, run_id: str, patch: dict) -> dict:
        self._runs.setdefault(run_id, {"run_id": run_id}).update(patch)
        return self._runs[run_id]

    def stats(self) -> dict:
        return {"n_runs": len(self._runs), "n_metrics": sum(len(v) for v in self._metrics.values())}

    def health(self) -> dict:
        return {"backend": "none", "degraded": False, "warnings": []}

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
def make_tracker(cfg: Optional[dict] = None,
                 root: Optional[str | Path] = None,
                 artifacts_dir: Optional[str | Path] = None,
                 backend: Optional[str] = None,
                 swanlab_module: Any = None,
                 attach_subruns: bool = True) -> Any:
    """按 `cfg["tracker"]` 构建记录后端（默认 LocalTracker，SwanLab 为软依赖）。

    cfg["tracker"] = {
        "backend": "local"|"swanlab"|"both"|"none",
        "swanlab_mode": "online"|"local"|"offline"|"disabled",
        "project": "zhixun-engine",
        "dir": "experiments/runs",
    }
    """
    tcfg = dict((cfg or {}).get("tracker") or {})
    be = str(backend or tcfg.get("backend", "local")).lower()
    if be not in BACKENDS:
        raise ValueError(f"未知 tracker.backend={be!r}，支持 {BACKENDS}")
    run_root = Path(root) if root else (ROOT / str(tcfg.get("dir", "experiments/runs")))
    art_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    if be == "none":
        return NullTracker()
    local = LocalTracker(root=run_root, artifacts_dir=art_dir)
    if be == "local":
        return local
    swan = SwanLabTracker(
        mode=str(tcfg.get("swanlab_mode", "offline")),
        project=str(tcfg.get("project", "zhixun-engine")),
        logdir=run_root / "swanlog",
        mirror=local,
        swanlab_module=swanlab_module,
        attach_subruns=attach_subruns,
    )
    if be == "both":
        # both：SwanLabTracker 内部已含本地镜像；这里显式保留多后端形态，
        # 便于未来增加第二后端（如外部 HTTP 服务）而不改调用方。
        return MultiTracker([swan])
    return swan


def build_tracker(cfg: Optional[dict] = None, run_ctx: Any = None, **kwargs) -> Any:
    """loop 集成入口：`build_tracker(cfg, run_ctx) -> Tracker`。

    参数
    ----
    cfg : 全局配置（读 cfg["tracker"]）。
    run_ctx : 可选的 `src.core.config.RunContext`；提供时使用其
              `outdir`（= experiments/runs）与 `artifacts_dir` 作为记录根目录，
              保证同一次闭环 run 的产物落在同一棵目录树里。
    kwargs : 透传 `make_tracker`（root/artifacts_dir/backend/swanlab_module/
             attach_subruns），显式参数优先于 run_ctx。

    返回对象满足 Tracker 协议；SwanLab 不可用时返回（含本地镜像的）
    SwanLabTracker，其 query/export 由本地镜像代理，主流程无感。
    """
    explicit_root = kwargs.pop("root", None)
    explicit_art = kwargs.pop("artifacts_dir", None)
    root = explicit_root
    art = explicit_art
    if run_ctx is not None:
        root = root or getattr(run_ctx, "outdir", None)
        art = art or getattr(run_ctx, "artifacts_dir", None)
    return make_tracker(cfg, root=root, artifacts_dir=art, **kwargs)


__all__ = [
    "ROOT", "DEFAULT_RUNS_DIR", "DEFAULT_ARTIFACTS_DIR", "SWANLAB_MODES",
    "RUN_ID_RE", "make_run_id", "jdumps", "jloads", "Tracker", "LocalTracker",
    "SwanLabTracker", "MultiTracker", "NullTracker", "make_tracker",
    "build_tracker",
]

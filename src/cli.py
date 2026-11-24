"""命令行入口（封闭评测时等价于"一条命令无人干预完成闭环"）。

用法：
  PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
  $PY -m src.cli run      --config configs/main.json --group main --seed 0
  $PY -m src.cli suite    --config configs/main.json --groups main,scenarios
  $PY -m src.cli audit-verify --path experiments/audit/<run>.jsonl
  $PY -m src.cli repro    --result results/main/synth_S7_full_mix_seed0.json
  $PY -m src.cli info
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .core.config import ROOT, load_config, save_config, set_global_seed


def _print_result(result: dict) -> None:
    h, s = result["hidden"], result["static_baseline_hidden"]
    c = result["cost"]
    cmp_ = result["comparison"]
    print("\n" + "=" * 72)
    print(f"run_id      : {result['meta']['run_id']}")
    print(f"scenario    : {result['meta']['scenario']}  seed={result['meta']['seed']}")
    print(f"driver      : {result['meta']['driver']}")
    print(f"静态基线    : stream macro-F1 = {s['stream_macro_f1']:.4f}  "
          f"final phase = {s['final_phase_macro_f1']:.4f}")
    print(f"自进化系统  : stream macro-F1 = {h['stream_macro_f1']:.4f}  "
          f"final phase = {h['final_phase_macro_f1']:.4f}")
    print(f"流式增益    : {cmp_['stream_gain_points']:+.2f} 百分点 "
          f"(胜 {cmp_['win_windows']}/{cmp_['n_windows']} 窗口)")
    print(f"累计成本    : compute={c['compute_units']:.1f}u labels={c['labels']} "
          f"tokens={c['tokens']} scalar={c['scalar']:.3f}")
    print(f"部署/回滚   : {sum(1 for x in result['timeline'] if x.get('deploy'))} 次部署 / "
          f"{sum(1 for x in result['timeline'] if x.get('rollback'))} 次回滚")
    print(f"审计链校验  : {result['audit']['verified']}")
    print("=" * 72 + "\n")


def cmd_run(args) -> int:
    from .loop import run_one

    cfg = load_config(args.config, overrides=json.loads(args.set) if args.set else None)
    if args.scenario:
        cfg["benchmark"]["scenario"] = args.scenario
    result = run_one(cfg, group=args.group, seed=args.seed)
    _print_result(result)
    return 0


def cmd_suite(args) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_all import run_matrix  # noqa: F401

    summary = run_matrix(load_config(args.config), groups=args.groups.split(","),
                         seeds=[int(s) for s in args.seeds.split(",")],
                         max_runs=args.max_runs)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:4000])
    return 0


def verify_audit(path: Path) -> tuple[bool, int, str]:
    """校验 sha256 哈希链。返回 (ok, n, last_hash)。"""
    prev = ""
    n = 0
    last = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            body = {k: v for k, v in rec.items() if k != "hash"}
            canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            h = hashlib.sha256((prev + canon).encode("utf-8")).hexdigest()
            if rec.get("hash") != h:
                return False, n, prev
            prev = h
            last = h
            n += 1
    return True, n, last


def cmd_audit_verify(args) -> int:
    """优先使用 engine.safety.AuditLog 校验（唯一哈希链实现），无引擎时回退本地实现。"""
    from .engine.safety import AuditLog

    path = Path(args.path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    all_ok = True
    for p in files:
        try:
            log = AuditLog(p)
            ok = bool(log.verify())
            n = len(log)
            last = log.last_hash()
        except Exception:
            ok, n, last = verify_audit(p)
        all_ok = all_ok and ok
        print(f"{'OK ' if ok else 'BAD'} {p}  records={n} last_hash={(last or '-')[:16]}")
    return 0 if all_ok else 1


def cmd_repro(args) -> int:
    from .core.evaluator import StreamEvaluation
    from .loop import run_one

    with open(args.result, "r", encoding="utf-8") as f:
        old = json.load(f)
    cfg = old["config"]
    new = run_one(cfg, group="repro", seed=old["meta"]["seed"])
    a, b = old["method_curve"], new["method_curve"]
    same = len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    print(f"复现：{old['meta']['run_id']} vs {new['meta']['run_id']} → "
          f"{'完全一致' if same else '不一致！'}")
    return 0 if same else 1


def cmd_info(args) -> int:
    from .core.config import DEFAULT_CONFIG

    print(json.dumps({"root": str(ROOT), "config_keys": sorted(DEFAULT_CONFIG.keys())},
                     ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="智训引擎·工业模型自进化系统")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="跑一次完整闭环")
    p.add_argument("--config", default="configs/main.json")
    p.add_argument("--group", default="main")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default=None)
    p.add_argument("--set", default=None, help="JSON 覆盖，如 '{\"budget\":{\"labels_total\":120}}'")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("suite", help="跑实验矩阵（基线/场景/消融/预算扫描）")
    p.add_argument("--config", default="configs/main.json")
    p.add_argument("--groups", default="main,scenarios,ablations,budget,route")
    p.add_argument("--seeds", default="7,17,27")
    p.add_argument("--max-runs", type=int, default=0)
    p.set_defaults(fn=cmd_suite)

    p = sub.add_parser("audit-verify", help="校验审计日志哈希链")
    p.add_argument("--path", required=True)
    p.set_defaults(fn=cmd_audit_verify)

    p = sub.add_parser("repro", help="用记录中的配置重跑并比对曲线")
    p.add_argument("--result", required=True)
    p.set_defaults(fn=cmd_repro)

    p = sub.add_parser("info", help="打印配置键")
    p.set_defaults(fn=cmd_info)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

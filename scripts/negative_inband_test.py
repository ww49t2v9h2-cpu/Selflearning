"""带内扰动负向测试（J3 式）：证明我们的门禁在"带内攻击"下 fail-closed，且判据状态不会被悄悄改写。

背景（配套研究 J3 / criteriond，未投稿）：同一带内攻击（残差 r=0.99·band，400 轮）下，
滚动中位数尺度规则到达"接受一切"、EMA 到达"拒绝一切"、固定阈值两者皆不——失效有方向。
我们的设计：固定门禁阈值 + 双门不变量（被拒样本只进隔离区，不写已接受参考域）。

本脚本是**独立负向测试**（不改变 161 项单元测试），分三部分：
  A. 门禁边界负向测试：带内增益（+0.4pt / +0.99pt / NaN / 缺失 / 类召回塌陷）必须拒绝；
     恰好/超过阈值（+1.0pt / +1.1pt）才放行（其余条件满足时）。
  B. 重复带内扰动：10 次带内候选（+0.4~0.99pt）全部拒绝，无累积放行。
  C. 双门不变量的源码级断言：`active_ref` 只能由 `reset_reference` / `on_promote` 写入，
     `_maybe_rebase` 只能写 `provisional_ref`；避免"被拒样本悄悄更新已接受参考域"。

用法：
  PY=.../python3
  $PY scripts/negative_inband_test.py
输出：
  results/tables/negative_inband_test.json / .md
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.gates import Gate  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "tables"


def mk_candidate(gain: float, *, cand_f1=None, baseline_f1=None,
                 recall=0.60, base_recall=0.60, ece=0.10, base_ece=0.10,
                 cost=0.0) -> dict:
    cand = {"probe_macro_f1": (0.80 + gain) if cand_f1 is None else cand_f1,
            "baseline_probe_macro_f1": 0.80 if baseline_f1 is None else baseline_f1,
            "class_recall_min": recall,
            "baseline_class_recall_min": base_recall,
            "ece": ece, "baseline_ece": base_ece, "cost": cost}
    return cand


INCUMBENT = {"probe_macro_f1": 0.80, "class_recall_min": 0.60, "ece": 0.10}
CTX = {"playbook": "RECALIBRATE_AFFINE",
       "budget": {"labels_total": 260, "labels_used": 100}}


def gate_case(gate: Gate, name: str, candidate: dict, expect_pass: bool,
              incumbent: dict | None = None) -> dict:
    r = gate.evaluate(candidate, incumbent or INCUMBENT, CTX)
    ok = bool(r.passed) == expect_pass
    failed = [k for k, v in (r.criteria or {}).items()
              if isinstance(v, dict) and v.get("pass") is False]
    return {"case": name, "expect": "pass" if expect_pass else "reject",
            "got": "pass" if r.passed else "reject", "ok": ok,
            "decision": r.decision, "reason": r.reason,
            "failed_criteria": failed,
            "criteria": {k: (v.get("pass") if isinstance(v, dict) else v)
                         for k, v in (r.criteria or {}).items()}}


def source_invariants() -> list[dict]:
    """AST 检查：active_ref 的写入函数集合；provisional_ref 的写入位置。"""
    path = ROOT / "src" / "engine" / "monitor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    active_writers, provisional_writers = set(), set()

    class V(ast.NodeVisitor):
        def __init__(self):
            self.fn = None

        def visit_FunctionDef(self, node):
            old, self.fn = self.fn, node.name
            self.generic_visit(node)
            self.fn = old

        def visit_Assign(self, node):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    if t.attr == "active_ref":
                        active_writers.add(self.fn)
                    elif t.attr == "provisional_ref":
                        provisional_writers.add(self.fn)
            self.generic_visit(node)

    V().visit(tree)
    checks = [
        {"check": "active_ref 写入者 ⊆ {reset_reference, on_promote}",
         "ok": active_writers <= {"reset_reference", "on_promote"},
         "detail": sorted(active_writers)},
        {"check": "_maybe_rebase 不写 active_ref",
         "ok": "_maybe_rebase" not in active_writers,
         "detail": "_maybe_rebase writes active_ref" if "_maybe_rebase" in active_writers else "ok"},
        {"check": "provisional_ref 由 _maybe_rebase 写入（隔离区）",
         "ok": "_maybe_rebase" in provisional_writers,
         "detail": sorted(provisional_writers)},
    ]
    return checks


def main() -> int:
    gate = Gate({})
    cases = [
        gate_case(gate, "带内 +0.4pt（<1pt 阈值）", mk_candidate(0.004), False),
        gate_case(gate, "带内 +0.99pt（临界下方）", mk_candidate(0.0099), False),
        gate_case(gate, "恰好 +1.0pt（阈值边界）", mk_candidate(0.010), True),
        gate_case(gate, "超阈值 +1.1pt", mk_candidate(0.011), True),
        gate_case(gate, "NaN 候选指标（fail-closed）",
                  mk_candidate(0.0, cand_f1=float("nan")), False),
        gate_case(gate, "基线缺失但 incumbent 可用 → 回退 incumbent（记录来源）",
                  mk_candidate(0.02, baseline_f1=float("nan")), True),
        gate_case(gate, "候选与 incumbent 都缺失基线（fail-closed）",
                  mk_candidate(0.02, baseline_f1=float("nan"), cand_f1=0.82),
                  False, incumbent={"class_recall_min": 0.60, "ece": 0.10}),
        gate_case(gate, "类召回塌陷（-0.2）",
                  mk_candidate(0.05, recall=0.40, base_recall=0.60), False),
    ]
    # B. 重复带内扰动（10 次）
    import random
    rng = random.Random(2026)
    repeated = []
    for i in range(10):
        g = 0.004 + 0.0055 * rng.random()  # 0.4%–0.95%
        r = gate.evaluate(mk_candidate(g), INCUMBENT, CTX)
        repeated.append({"i": i, "gain_pt": round(g * 100, 3), "passed": bool(r.passed)})
    repeated_ok = all(not x["passed"] for x in repeated)

    src_checks = source_invariants()
    src_ok = all(c["ok"] for c in src_checks)
    a_ok = all(c["ok"] for c in cases)
    all_ok = a_ok and repeated_ok and src_ok

    payload = {"method": "带内扰动负向测试（J3 式；独立脚本，不改单元测试计数）",
               "gate_config": {"min_gain_points": gate.min_gain,
                                "max_recall_drop": gate.max_recall_drop},
               "A_gate_cases": cases, "B_repeated_inband": repeated,
               "C_source_invariants": src_checks,
               "summary": {"A_ok": a_ok, "B_ok": repeated_ok, "C_ok": src_ok, "all_ok": all_ok}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "negative_inband_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 带内扰动负向测试（J3 式，自动生成）", "",
             "> 参考：配套研究 J3（criteriond，未投稿）——同一带内攻击 r=0.99 下，滚动中位数尺度到",
             "> 「接受一切」、EMA 到「拒绝一切」、固定阈值两者皆不。我们是固定阈值 + 双门不变量，",
             "> 本脚本验证带内候选被门禁拒绝、且判据状态不会被悄悄改写。", "",
             "## A. 门禁边界负向测试", "",
             "| 用例 | 期望 | 实际 | 通过 |", "|---|---|---|---|"]
    for c in cases:
        lines.append(f"| {c['case']} | {c['expect']} | {c['got']} | {'✅' if c['ok'] else '❌'} |")
    lines += ["", "## B. 重复带内扰动（10 次，符合增益 0.4–0.95pt）", "",
              f"- 全部拒绝：{'✅' if repeated_ok else '❌'}（放行 {sum(1 for x in repeated if x['passed'])}/10）", "",
              "## C. 双门不变量（源码级断言）", "",
              "| 检查 | 结果 | 明细 |", "|---|---|---|"]
    for c in src_checks:
        lines.append(f"| {c['check']} | {'✅' if c['ok'] else '❌'} | {c['detail']} |")
    lines += ["", f"**总体：{'✅ 通过' if all_ok else '❌ 失败'}**（A={'✅' if a_ok else '❌'} / "
                  f"B={'✅' if repeated_ok else '❌'} / C={'✅' if src_ok else '❌'}）", "",
              "边界：本脚本是固定阈值 + 双门语义的负向验证，不是对 J3 多副本系统的复现；",
              "J3 的两终态数字见 docs/06 §6 与配套研究论文（未投稿）。", ""]
    (OUT / "negative_inband_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {OUT/'negative_inband_test.json'} / .md")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

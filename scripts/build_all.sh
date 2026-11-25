#!/usr/bin/env bash
# 一键构建参赛包：测试 → 基线 → 主实验矩阵 → 图 → 验收 → 交付物
# 用法：bash scripts/build_all.sh [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python3
FIGPY="${FIGPY:-/tmp/plotenv/bin/python}"

echo "==[1/13] 单元测试 =="
$PY -m unittest discover -s tests

echo "==[2/13] 冒烟 =="
$PY scripts/smoke.py

echo "==[3/13] 基线套件（B0/B1/B2 + AI4I）=="
$PY scripts/baseline_suite.py --include-ai4i

echo "==[4/13] 主实验矩阵（S1-S7 × 3 seeds + 消融 + 预算扫描 + 真实数据 + 路线对比）=="
if [ "${1:-}" = "--quick" ]; then
  $PY scripts/run_all.py --groups main,real --seeds 7
else
  $PY scripts/run_all.py --groups main,scenarios,ablations,budget,real,route --seeds 7,17,27
fi

echo "==[5/13] S2 零标注仿射分析 =="
$PY scripts/analyze_affine_s2.py

echo "==[6/13] 图表 =="
if [ -x "$FIGPY" ]; then "$FIGPY" scripts/make_figures.py; else echo "  (跳过：未找到 matplotlib venv, FIGPY=$FIGPY)"; fi

echo "==[7/13] 验收（含可复现校验）=="
$PY scripts/acceptance.py --repro || echo "  ⚠ 验收未全通过，请检查 results/acceptance.json"

echo "==[8/13] 官方评分五步对齐 =="
$PY scripts/analyze_official_rubric.py

echo "==[9/13] 门禁操作点曲线（J2 式）=="
$PY scripts/analyze_gate_operating_curve.py

echo "==[10/13] criteriond 不变量对照检查（J3 式）=="
$PY scripts/check_invariants.py

echo "==[11/13] 带内扰动负向测试（J3 式）=="
$PY scripts/negative_inband_test.py

echo "==[12/13] 决赛数据包-B 抗脏数据压力测试 =="
$PY scripts/stress_dirty_data.py

echo "==[13/13] 交付物（方案 docx / PPT / 系统镜像）=="
$PY scripts/build_deliverables.py --pptx --zip
$PY scripts/gen_record_book.py --out-md deliverables/实验记录汇编.md --out-xlsx deliverables/实验记录汇编.xlsx

echo
echo "完成。下一步（人工）："
echo "  1) DSH office_convert: deliverables/智训引擎_参赛方案.docx → deliverables/智训引擎_参赛方案.pdf"
echo "  2) 复核 deliverables/ 与 docs/14 的数字一致性"

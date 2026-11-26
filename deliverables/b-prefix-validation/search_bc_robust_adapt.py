"""系统搜索 B/C 跨阶段稳健的无标签适应方法。

不用隐藏标签；目标阶段前 20% 只估计特征统计量，后 80% 评估。
比较均值/尺度联合收缩、仅尺度收缩、逐特征收缩及前缀拟合的分位数混合。
输出 results/official/bc_prefix_adapt/summary.{json,md}
"""
from __future__ import annotations
import json, sys
from statistics import NormalDist
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness_image'))
import contest_submission as cs
OUT=ROOT/'results/official/bc_prefix_adapt'; OUT.mkdir(parents=True,exist_ok=True)

def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2)))
def mae(y,p): return float(np.mean(np.abs(np.asarray(p)-np.asarray(y))))

def features(model,x):
    flat=cs.flatten_official(x)
    return np.asarray([float(flat.get(cs._path_key(p),0.0)) for p in model.paths],dtype=np.float64)
def pred_matrix(model, x, mean, scale):
    normalized = np.clip((x - mean) / scale, -10.0, 10.0)
    hidden = np.tanh(normalized @ model.w1 + model.b1)
    values = (hidden @ model.w2 + model.b2).reshape(-1)
    return np.maximum(
        0.0,
        values * float(model.meta.get("target_scale", 1.0))
        + float(model.meta.get("target_mean", 0.0)),
    )

def prefix_quantile_map(prefix, x, mean, scale):
    normal = NormalDist()
    size = len(prefix)
    scores = np.asarray([normal.inv_cdf((i + 0.5) / (size + 1))
                         for i in range(size + 1)])
    mapped = x.copy()
    for column, values in enumerate(prefix.T):
        ordered = np.sort(values)
        if ordered[0] == ordered[-1]:
            continue
        left = np.searchsorted(ordered, x[:, column], side="left")
        right = np.searchsorted(ordered, x[:, column], side="right")
        z = (scores[left] + scores[right]) / 2.0
        mapped[:, column] = mean[column] + scale[column] * z
    return mapped

def load_model(path):
    z=np.load(path,allow_pickle=False); meta=json.loads(str(z['metadata']))
    return cs.OfficialModel(meta,z['w1'],z['b1'],z['w2'],z['b2'],z['mean'],z['scale'])
def metrics(y,p): return {'rmse':rmse(y,p),'mae':mae(y,p)}
def shrink_stats(parent_mean,parent_scale,target_mean,target_scale,alpha):
    return parent_mean+alpha*(target_mean-parent_mean), parent_scale+alpha*(target_scale-parent_scale)

def eval_stage(xs,ys,cut,model, prefix_size=None):
    if prefix_size is None:
        prefix_size = cut
    xa = np.asarray([features(model, x) for x in xs[:prefix_size]])
    xt = np.asarray([features(model, x) for x in xs[cut:]])
    yt = ys[cut:]
    tm, ts = xa.mean(0), xa.std(0)
    ts[ts < 1e-12] = model.scale[ts < 1e-12]
    res = {"static": metrics(yt, pred_matrix(model, xt, model.mean, model.scale))}
    for alpha in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        mean, scale = shrink_stats(model.mean, model.scale, tm, ts, alpha)
        res[f"shrink{alpha}"] = metrics(yt, pred_matrix(model, xt, mean, scale))
        res[f"scale{alpha}"] = metrics(yt, pred_matrix(model, xt, model.mean, scale))
    shift = np.abs(tm - model.mean) / model.scale
    mean, scale = shrink_stats(model.mean, model.scale, tm, ts, 1.0 / (1.0 + shift ** 2))
    res["perfeat"] = metrics(yt, pred_matrix(model, xt, mean, scale))
    mapped = prefix_quantile_map(xa, xt, model.mean, model.scale)
    for alpha in (0.05, 0.1, 0.2, 0.5, 1.0):
        blended = xt + alpha * (mapped - xt)
        res[f"quantile{alpha}"] = metrics(
            yt, pred_matrix(model, blended, model.mean, model.scale)
        )
    return res

def bearing_groups(path):
    groups = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                groups.setdefault(row["meta"]["bearing"], []).append(row)
    return {name: sorted(rows, key=lambda row: row["meta"]["snapshot"])
            for name, rows in sorted(groups.items())}

def run_b():
    public = ROOT / "data/official/cpu-b-rul-20261110-v1/public"
    model = load_model(ROOT / "results/official/B_best/checkpoint.npz")
    groups = bearing_groups(public / "validation.jsonl")
    training_groups = bearing_groups(public / "train.jsonl")
    references = {
        name: np.asarray([features(model, row["inputs"])
                          for group in training_groups.values()
                          for row in (group if length is None else group[:length])])
        for name, length in (("5", 5), ("20", 20), ("100", 100), ("all", None))
    }
    rows=[]
    for name,rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda x:x['meta']['snapshot']); xs=[r['inputs'] for r in rr]; ys=[float(r['answer']) for r in rr]
        cut=max(5,int(len(xs)*.2)); ev=eval_stage(xs,ys,cut,model); ev['bearing']=name; ev['n']=len(rr)
        ev["prefix_sensitivity"] = {
            str(fraction): eval_stage(xs, ys, cut, model, max(5, int(len(xs) * fraction)))["quantile0.2"]
            for fraction in (0.05, 0.1)
        }
        x = np.asarray([features(model, inputs) for inputs in xs])
        ev["stateless_static"] = metrics(ys, pred_matrix(model, x, model.mean, model.scale))
        ev["stateless"] = {}
        for name, reference in references.items():
            mapped = prefix_quantile_map(reference, x, model.mean, model.scale)
            ev["stateless"][name] = {
                str(alpha): metrics(ys, pred_matrix(
                    model, x + alpha * (mapped - x), model.mean, model.scale
                ))
                for alpha in (0.05, 0.1, 0.2, 0.5, 1.0)
            }
        rows.append(ev)
    return rows

def load_cmapss(fd):
    data=np.loadtxt(ROOT/'data/external/cmapss'/f'train_{fd}.txt'); units=data[:,0].astype(int); cyc=data[:,1].astype(int)
    last={}
    for u,c in zip(units,cyc): last[u]=max(last.get(u,0),c)
    xs=[{'settings':row[2:5].tolist(),'sensors':row[5:26].tolist()} for row in data]
    ys=np.asarray([last[u]-c for u,c in zip(units,cyc)],float); return xs,ys

def run_c():
    model=load_model(ROOT/'results/official/C_best/checkpoint.npz'); rows=[]
    for fd in ('FD002','FD003','FD004'):
        xs,ys=load_cmapss(fd); cut=max(20,int(len(xs)*.2)); ev=eval_stage(xs,ys,cut,model); ev['subset']=fd; ev['n']=len(xs); rows.append(ev)
    return rows

def main():
    b=run_b(); c=run_c(); summary={'B':b,'C':c}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding='utf-8')
    methods = [key for key, value in b[0].items() if isinstance(value, dict) and "rmse" in value and key != "stateless_static"]
    lines = [
        "# B/C 前缀拟合无标签适应", "",
        "B 使用公开验证轴承，C 使用外部 C-MAPSS FD002/FD003/FD004。",
        "主实验用前 20% 的输入拟合变换并在后 80% 评分。B 至少使用 5 个前缀样本。",
        "混合权重在同一公开验证集上比较。平台推理入口逐条调用，本实验假设可以按实体和时间获得前缀。", "",
        "## B",
    ]
    for row in b:
        lines.append(f"\n### {row['bearing']} (n={row['n']})")
        lines.append('| method | RMSE | MAE |')
        lines.append('|---|---|---|')
        for m in methods: lines.append(f"| {m} | {row[m]['rmse']:.2f} | {row[m]['mae']:.2f} |")
        lines.append("\n固定后 80% 评分，分位数混合权重 0.2 的前缀长度敏感性")
        for fraction, scores in row["prefix_sensitivity"].items():
            lines.append(f"- {float(fraction):.0%} 前缀，RMSE {scores['rmse']:.2f}，MAE {scores['mae']:.2f}")
    lines += ["", "## B 逐条推理对照", "",
              "映射只用公开训练轴承拟合。对公开验证轴承的全部样本评分。",
              "| 轴承 | 静态 RMSE | 前 5 条 | 前 20 条 | 前 100 条 | 全部 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in b:
        baseline = row["stateless_static"]["rmse"]
        values = [results["0.2"]["rmse"] for results in row["stateless"].values()]
        lines.append(f"| {row['bearing']} | {baseline:.2f} | "
                     + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines += ["", "表中混合权重为 0.2。其余权重与 MAE 见 summary.json。", "", "## C"]
    for row in c:
        lines.append(f"\n### {row['subset']} (n={row['n']})")
        lines.append('| method | RMSE | MAE |')
        lines.append('|---|---|---|')
        for m in methods: lines.append(f"| {m} | {row[m]['rmse']:.2f} | {row[m]['mae']:.2f} |")
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()

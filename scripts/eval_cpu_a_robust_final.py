"""官方 CPU-A 上最终的鲁棒闭环配置（r1）：monitor 重校准 + 动作白名单（排除 PRIOR_CORRECTION）。
3 seeds 保存完整结果，供报告/复核。"""
import copy, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from src.core.config import RunContext, load_config
from src.loop import SelfEvolvingLoop, new_run_id
from eval_official_cpu_a import build_benchmark
OUT=ROOT/'results'/'official'/'cpu_a_evolving_robust'; OUT.mkdir(parents=True, exist_ok=True)
CFG_OVER={
 'monitor': {'mmd_threshold':0.20,'ks_count_threshold':6,'probe_max_labels':16,'probe_labels':16,
             'probe_min_labels':10,'adapt_reserve_frac':0.35,'probe_every':5,'proxy_drop_threshold':0.025},
 'gates': {'min_gain_points':0.003,'min_probe_gain':0.0},
 'budget': {'labels_total':360},
 'diagnosis': {'label_shift_tv':0.30},
 'safety': {'allowed_playbooks': ['NOOP','RECALIBRATE_AFFINE','SHADOW_ONLY','ROLLBACK','LAST_LAYER_FT']},
}
def main():
    rows=[]; summary={'config':'r1_no_prior (monitor recalibrated + whitelist)',
                      'protocol':'base=official batch1; stream=raw batches3-10 (70% visible/30% hidden)',
                      'overrides':CFG_OVER,'seeds':{}}
    for seed in (7,17,27):
        cfg=copy.deepcopy(load_config('configs/main.json')); cfg['seed']=seed; cfg['run']['name']='official-cpu-a-robust'
        cfg['data']={'sanitize':True,'clip_quantile':0.9995,'fill':'median'}
        for sect in ('monitor','diagnosis'):
            cfg.setdefault('engine',{}).setdefault(sect,{}).update(CFG_OVER[sect])
        for sect in ('gates','safety','budget'):
            cfg.setdefault(sect,{}).update(CFG_OVER[sect])
        bench=build_benchmark(seed)
        loop=SelfEvolvingLoop(cfg, RunContext(cfg=cfg, run_id=new_run_id(f'official-cpu-a-robust-s{seed}')), benchmark=bench)
        res=loop.run()
        (OUT/f'seed{seed}.json').write_text(json.dumps(res,ensure_ascii=False,indent=1,default=str),encoding='utf-8')
        comp=res.get('comparison',{})
        row={'seed':seed,'static':res['static_baseline_hidden']['stream_macro_f1'],
             'method':res['hidden']['stream_macro_f1'],'gain':comp.get('stream_gain_points'),
             'labels':int(res['cost']['labels']),'compute_units':res['cost']['compute_units'],
             'deploys':sum(1 for x in res['timeline'] if x.get('deploy')),
             'rollbacks':sum(1 for x in res['timeline'] if x.get('rollback')),
             'triggers':sum(1 for x in res['timeline'] if x.get('triggered'))}
        rows.append(row); summary['seeds'][str(seed)]=row
        print(row)
    summary['mean']={k:float(np.mean([r[k] for r in rows])) for k in ('static','method','gain','labels')}
    summary['all_positive']=bool(all(r['gain']>0 for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding='utf-8')
    lines=['# 官方 CPU-A 鲁棒闭环配置（3 seeds）','',f"- 平均：静态 **{summary['mean']['static']:.4f}** → 自进化 **{summary['mean']['method']:.4f}**（**{summary['mean']['gain']:+.2f}pt**）；标注 {summary['mean']['labels']:.0f}；3 seeds 全部为正：{'✅' if summary['all_positive'] else '❌'}",'',
           '| seed | 静态 | 自进化 | 增益(pt) | 标注 | 部署/回滚 |','|---|---|---|---|---|---|']
    for r in rows: lines.append(f"| {r['seed']} | {r['static']:.4f} | {r['method']:.4f} | {r['gain']:+.2f} | {r['labels']} | {r['deploys']}/{r['rollbacks']} |")
    lines+=['','配置（相对 configs/main.json）：','```json',json.dumps(CFG_OVER,ensure_ascii=False,indent=1),'```',
            '说明：动作白名单排除 PRIOR_CORRECTION（官方 A 上探针先验估计不稳）；README/报告中的"官方数据"章节引用本结果。','']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()

"""在 c4 配置基础上做鲁棒性变体：限制动作白名单/提高门禁/加大探针，3 seeds 验证。"""
import copy, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from src.core.config import RunContext, load_config
from src.loop import SelfEvolvingLoop, new_run_id
from eval_official_cpu_a import build_benchmark
OUT=ROOT/'results'/'official'/'cpu_a_robust'; OUT.mkdir(parents=True, exist_ok=True)
BASE={
 'monitor': {'mmd_threshold':0.20,'ks_count_threshold':6,'probe_max_labels':16,'probe_labels':16,
             'probe_min_labels':10,'adapt_reserve_frac':0.35,'probe_every':5,'proxy_drop_threshold':0.025},
 'gates': {'min_gain_points':0.003,'min_probe_gain':0.0},
 'budget': {'labels_total':360},
 'diagnosis': {'label_shift_tv':0.30},
}
VARIANTS={
 'r1_no_prior': {'safety': {'allowed_playbooks': ['NOOP','RECALIBRATE_AFFINE','SHADOW_ONLY','ROLLBACK','LAST_LAYER_FT']}},
 'r2_no_prior_gate': {'safety': {'allowed_playbooks': ['NOOP','RECALIBRATE_AFFINE','SHADOW_ONLY','ROLLBACK','LAST_LAYER_FT']},
                      'gates': {'min_gain_points':0.01,'min_probe_gain':0.003}},
 'r3_no_prior_bigprobe': {'safety': {'allowed_playbooks': ['NOOP','RECALIBRATE_AFFINE','SHADOW_ONLY','ROLLBACK','LAST_LAYER_FT']},
                          'monitor': {'probe_max_labels':24,'probe_labels':24,'probe_min_labels':12,
                                      'probe_every':6,'adapt_reserve_frac':0.25},
                          'budget': {'labels_total':480}},
}
def apply(cfg, over):
    out=copy.deepcopy(cfg); out['seed']=7; out['run']['name']='official-cpu-a-robust'
    out['data']={'sanitize':True,'clip_quantile':0.9995,'fill':'median'}
    merged=copy.deepcopy(BASE)
    for k,v in over.items():
        merged.setdefault(k,{}).update(v)
    for sect in ('monitor','diagnosis'):
        out.setdefault('engine',{}).setdefault(sect,{}).update(merged.get(sect,{}))
    for sect in ('gates','safety','budget'):
        out.setdefault(sect,{}).update(merged.get(sect,{}))
    return out
def run(name, over, seed):
    cfg=apply(load_config('configs/main.json'), over); cfg['seed']=seed
    if name=='r3_no_prior_bigprobe' and 'budget' in over:
        cfg['budget'].update(over['budget'])
    bench=build_benchmark(seed)
    loop=SelfEvolvingLoop(cfg, RunContext(cfg=cfg, run_id=new_run_id(f'robust-{name}-s{seed}')), benchmark=bench)
    r=loop.run(); comp=r.get('comparison',{})
    return {'name':name,'seed':seed,'method':r['hidden']['stream_macro_f1'],
            'static':r['static_baseline_hidden']['stream_macro_f1'],'gain':comp.get('stream_gain_points'),
            'labels':r['cost']['labels'],'deploys':sum(1 for x in r['timeline'] if x.get('deploy')),
            'rollbacks':sum(1 for x in r['timeline'] if x.get('rollback'))}
def main():
    summary=[]
    for name, over in VARIANTS.items():
        rows=[run(name, over, s) for s in (7,17,27)]
        summary += rows
        gains=[x['gain'] for x in rows]
        print(name, [round(g,2) for g in gains], 'mean', round(float(np.mean(gains)),2))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding='utf-8')
    lines=['# CPU-A 鲁棒性变体（3 seeds）','','| 变体 | seed7 | seed17 | seed27 | 平均 |','|---|---|---|---|---|']
    for name in VARIANTS:
        g=[x['gain'] for x in summary if x['name']==name]
        lines.append(f'| {name} | {g[0]:+.2f} | {g[1]:+.2f} | {g[2]:+.2f} | {np.mean(g):+.2f} |')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()

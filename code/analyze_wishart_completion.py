"""Convergence comparisons use forecasts averaged BEFORE scoring."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import ndtr
from expanded_experiment import score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wishart_completion';O=OUT/'analysis'


def records():
    rows=[]
    for p in (OUT/'results').glob('*.json'):
        if p.name.startswith('y'):rows.append(json.loads(p.read_text()))
    x=pd.DataFrame(rows)
    if len(x):x['antithetic']=x.get('antithetic',pd.Series(False,index=x.index)).fillna(False).astype(bool);x['defensive']=x.get('defensive',pd.Series(0.,index=x.index)).fillna(0.);x['lag']=x.get('lag',pd.Series(1,index=x.index)).fillna(1).astype(int);x['moment_auxiliary']=x.get('moment_auxiliary',pd.Series(False,index=x.index)).fillna(False).astype(bool)
    if len(x):x['aux_power']=x.get('aux_power',pd.Series(1.,index=x.index)).fillna(1.)
    if len(x):x['canonical']=x.get('canonical',pd.Series(False,index=x.index)).fillna(False).astype(bool)
    if len(x):x['gauge_paths']=x.get('gauge_paths',pd.Series(False,index=x.index)).fillna(False).astype(bool)
    return x


def load(group):
    parts=[]
    for r in group.to_dict('records'):
        f=pd.read_csv(ROOT/r['forecasts'],parse_dates=['date']);parts.append(f)
    return pd.concat(parts,ignore_index=True)


def ensemble(f):
    cols=['date','fold','portfolio','alpha'];g=f.groupby(cols)[['loss','var','es']].mean().reset_index()
    g['fz0']=score(g.loss,g['var'],g.es,g.alpha);g['hit']=(g.loss>g['var']).astype(int);g['es_residual']=g.es-g['var']-np.maximum(g.loss-g['var'],0)/g.alpha
    g['model']='Heston-Wishart';g['seed']=0;return g


def summarize():
    r=records()
    if r.empty:return
    r.to_csv(O/'run_inventory.csv',index=False);rows=[];comparisons=[]
    config=['particles','substeps','predictive','warmup','moves','antithetic','defensive','lag','moment_auxiliary','aux_power','canonical','gauge_paths']
    for cfg,g in r.groupby(config):
        x=load(g);en=ensemble(x)
        for (year,pn,a),z in en.groupby(['fold','portfolio','alpha']):
            raw=x[(x.fold==year)&(x.portfolio==pn)&(x.alpha==a)];bank_scores=raw.groupby('seed').fz0.mean();banks=raw.seed.nunique()
            rows.append(dict(zip(config,cfg),year=int(year),portfolio=pn,alpha=a,banks=banks,fz0=z.fz0.mean(),hits=z.hit.sum(),bank_score_sd=bank_scores.std(),bank_score_se=bank_scores.std()/np.sqrt(banks),es_bank_relative_rms=float(np.sqrt(np.mean((raw.groupby('date').es.std()/raw.groupby('date').es.mean())**2))) if banks>1 else np.nan))
    # All matched configuration pairs are retained, not only favorable ones.
    for year,g in r.groupby('year'):
        configs=list(g.groupby(config))
        for i,(ca,ga) in enumerate(configs):
            for cb,gb in configs[i+1:]:
                fa=ensemble(load(ga));fb=ensemble(load(gb));pair=fa.merge(fb,on=['date','fold','portfolio','alpha'],suffixes=('_a','_b'),validate='one_to_one')
                for (pn,a),z in pair.groupby(['portfolio','alpha']):
                    comparisons.append(dict(year=int(year),portfolio=pn,alpha=a,config_a=str(ca),config_b=str(cb),banks_a=len(ga),banks_b=len(gb),delta_fz0=float((z.fz0_b-z.fz0_a).mean()),es_relative_rms=float(np.sqrt(np.mean(((z.es_b-z.es_a)/z.es_b)**2))),var_relative_rms=float(np.sqrt(np.mean(((z.var_b-z.var_a)/z.var_b)**2))),max_abs_es_change=float(abs(z.es_b-z.es_a).max())))
    pd.DataFrame(rows).to_csv(O/'ensemble_runs.csv',index=False);pd.DataFrame(comparisons).to_csv(O/'convergence_pairs.csv',index=False)
    print(pd.DataFrame(rows).query("portfolio=='equal' and alpha==.01").to_string(index=False))
    if comparisons:print(pd.DataFrame(comparisons).query("portfolio=='equal' and alpha==.01").to_string(index=False))

if __name__=='__main__':summarize()

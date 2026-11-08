"""Conditioning and missing-data checks for the expanded experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import tex_table,fmt,bootstrap_means,holm
from revision_analysis import label
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'expansion';R=BASE/'results';O=BASE/'analysis';O.mkdir(exist_ok=True)
def interval(d):
 mean=d.mean(0);b=bootstrap_means(d.to_numpy(),B=9999)-mean.to_numpy();lo=mean-np.quantile(b,.975,axis=0);hi=mean-np.quantile(b,.025,axis=0)
 p=np.minimum(1,2*np.minimum((1+(b>=mean.to_numpy()).sum(0))/10000,(1+(b<=mean.to_numpy()).sum(0))/10000))
 return pd.DataFrame(dict(model=d.columns,delta=mean.to_numpy(),low=lo.to_numpy(),high=hi.to_numpy(),p=p,p_holm=holm(p)))
def filters():
 paths=list(R.glob('crsp5cov_*.csv.gz'));assert len(paths)==100,len(paths)
 cov=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in paths],ignore_index=True)
 diag=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in R.glob('crsp5_*_0_*.csv.gz')],ignore_index=True)
 cov.to_csv(O/'crsp5cov_forecasts.csv.gz',index=False)
 a=cov[(cov.portfolio=='equal')&(cov.alpha==.01)].pivot(index='date',columns='model',values='fz0').sort_index()
 b=diag[(diag.portfolio=='equal')&(diag.alpha==.01)].pivot(index='date',columns='model',values='fz0').sort_index();assert a.shape==b.shape==(2514,10)
 inf=interval(a-b).set_index('model');rows=[];summary=[]
 for name in a.columns:
  z=inf.loc[name];g=cov[(cov.portfolio=='equal')&(cov.alpha==.01)&cov.model.eq(name)];delta2020=float((a[name]-b[name])[a.index.year==2020].mean())
  row=dict(model=name,diag_fz0=float(b[name].mean()),cov_fz0=float(a[name].mean()),hits_percent=float(100*g.hit.mean()),es_moment=float(g.es_residual.mean()/g.es.mean()),delta_2020=delta2020,**z.to_dict());summary.append(row)
  rows.append([label(name),fmt(row['diag_fz0']),fmt(row['cov_fz0']),fmt(z.delta),'['+fmt(z.low)+', '+fmt(z.high)+']',fmt(z.p_holm),fmt(delta2020)])
 pd.DataFrame(summary).to_csv(O/'filter_comparison.csv',index=False)
 (O/'filter_table.tex').write_text(tex_table('Conditioning ablation on the five-basket panel, equal-weight one-percent loss, seed 0. Every model is refitted and selected by the same validation FZ0 criterion under each filter. Negative differences favor full covariance. Intervals use 9,999 paired circular draws and 20-day blocks; Holm adjustment is within these ten filter contrasts. The final column isolates 2020 descriptively. This separate experiment was specified before its own fits, after some diagonal-filter results existed.','tab:filters',['Model','Diagonal','Full cov.','$\\Delta$','95\\% interval','Holm $p$','$\\Delta$ 2020'],rows))
 # Independent baseline identity: the new full-filter FHS equals the earlier
 # empirical dynamic-covariance benchmark, up to arithmetic precision.
 old=pd.read_csv(ROOT/'results/revision/dynamic_forecasts.csv.gz',parse_dates=['date']);old=old[old.model.eq('EWMAcov-FHS')]
 match=cov[cov.model.eq('Historical')].merge(old,on=['date','portfolio','alpha'],suffixes=('_new','_old'),validate='one_to_one')
 err=float(abs(match.fz0_new-match.fz0_old).max());assert err<1e-8,err
 (O/'filter_identity_check.json').write_text(json.dumps(dict(passed=True,max_fz0_difference=err,rows=len(match)),indent=2))

def missing():
 p=O/'crsp100_forecasts.csv.gz';raw=pd.read_csv(p,parse_dates=['date']);weights={'equal':np.ones(100)/100,'top10':np.r_[np.ones(10)/10,np.zeros(90)],'largest':np.eye(100)[0]};masks=[]
 for year in range(2016,2026):
  m=pd.read_csv(BASE/'data'/f'crsp100_{year}_missing.csv.gz',index_col=0,parse_dates=True);m=m[m.index.year==year].astype(bool)
  for pn,w in weights.items():masks.extend(dict(date=dt,portfolio=pn,missing_weight=float(v)) for dt,v in zip(m.index,m.values@w))
 masks=pd.DataFrame(masks);a=raw.merge(masks,on=['date','portfolio'],validate='many_to_one');a['stress_loss']=a.loss+100*a.missing_weight
 a['stress_fz0']=np.log(a.es)+a['var']/a.es-1+np.maximum(a.stress_loss-a['var'],0)/(a.alpha*a.es)
 g=a[(a.portfolio=='equal')&(a.alpha==.01)].groupby(['date','model'],as_index=False).agg(fz0=('fz0','mean'),stress_fz0=('stress_fz0','mean'),missing_weight=('missing_weight','first'))
 rows=[];summary=[]
 for name,z in g.groupby('model'):
  base=g[g.model.eq('Historical')].set_index('date');z=z.set_index('date');complete=z.missing_weight.eq(0)
  v=dict(model=name,base_fz0=float(z.fz0.mean()),stress_fz0=float(z.stress_fz0.mean()),base_delta_fhs=float((z.fz0-base.fz0).mean()),stress_delta_fhs=float((z.stress_fz0-base.stress_fz0).mean()),complete_dates=int(complete.sum()),complete_delta_fhs=float((z.fz0-base.fz0)[complete].mean()))
  summary.append(v);rows.append([label(name),fmt(v['base_delta_fhs']),fmt(v['stress_delta_fhs']),fmt(v['complete_delta_fhs'])])
 pd.DataFrame(summary).to_csv(O/'crsp100_missing_stress.csv',index=False);n=summary[0]['complete_dates']
 audit=dict(test_dates=2514,complete_dates=n,max_missing_equal_weight=float(masks[masks.portfolio.eq('equal')].missing_weight.max()),mean_missing_equal_weight=float(masks[masks.portfolio.eq('equal')].missing_weight.mean()))
 (O/'missing_audit.json').write_text(json.dumps(audit,indent=2))
 (O/'missing_table.tex').write_text(tex_table(f'100-stock missing-data sensitivity: mean one-percent score difference from FHS. Stress holds forecasts fixed and assigns a loss of 100\\% to every missing security-day return; repeated missing days make this deliberately severe, not a plausible liquidation path. Complete-date results retain the same {n} dates for all models and are an ex-post diagnostic, not a replacement universe.','tab:missing100',['Model','Baseline','Missing-return stress','Complete dates'],rows))

def main():
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['filters','missing','all'],default='all');a=p.parse_args()
 if a.part in ['filters','all']:filters()
 if a.part in ['missing','all']:missing()
 print('Sensitivity complete',a.part)
if __name__=='__main__':main()

"""Partial missing-return stress and numerical scale of the covariance ridge."""
from pathlib import Path
import numpy as np,pandas as pd,json
from analyze import tex_table
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round5/analysis'

def main():
    masks=[];ridge=[]
    for year in range(2016,2026):
        m=pd.read_csv(ROOT/'expansion/data'/f'crsp100_{year}_missing.csv.gz',index_col=0,parse_dates=True)
        m=m[m.index.year==year].astype(bool)
        masks.extend(dict(date=str(date.date()),missing_weight=float(v)) for date,v in zip(m.index,m.mean(axis=1)))
        f=pd.read_csv(ROOT/'expansion/data'/f'crsp100_{year}.csv',index_col=0,parse_dates=True);x=f.values*100
        cov=np.cov(x[:252],rowvar=False);mins=[];ratios=[];cond=[]
        for date,value in zip(f.index,x):
            if date.year==year:
                eigen=np.linalg.eigvalsh(cov);mins.append(eigen[0]);ratios.append(1e-8/eigen[0]);cond.append((eigen[-1]+1e-8)/(eigen[0]+1e-8))
            cov=.94*cov+.06*np.outer(value,value)
        ridge.append(dict(year=year,ridge=1e-8,min_eigenvalue=float(np.min(mins)),max_relative_min_eigenvalue_shift=float(np.max(ratios)),median_condition=float(np.median(cond)),max_condition=float(np.max(cond))))
    mask=pd.DataFrame(masks)
    f=pd.read_csv(ROOT/'referee_round4/analysis/crsp100_forecasts.csv.gz');f=f[(f.alpha==.01)&(f.portfolio=='equal')].copy()
    f['date']=pd.to_datetime(f.date).dt.strftime('%Y-%m-%d');f=f.merge(mask,on='date',validate='many_to_one')
    f['stress_loss']=f.loss+20*f.missing_weight
    f['stress_fz0']=np.log(f.es)+f['var']/f.es-1+np.maximum(f.stress_loss-f['var'],0)/(f.alpha*f.es)
    g=f.groupby(['date','model'],as_index=False).agg(fz0=('fz0','mean'),stress_fz0=('stress_fz0','mean'),missing_weight=('missing_weight','first'))
    base=g[g.model=='Historical'].set_index('date');rows=[];records=[]
    for name,z in g.groupby('model'):
        z=z.set_index('date');complete=z.missing_weight==0
        vals=dict(model=name,baseline=float((z.fz0-base.fz0).mean()),stress20=float((z.stress_fz0-base.stress_fz0).mean()),complete=float((z.fz0-base.fz0)[complete].mean()),complete_dates=int(complete.sum()))
        records.append(vals);rows.append(['FHS' if name=='Historical' else name,*[f'{vals[k]:.3f}' for k in ['baseline','stress20','complete']]])
    pd.DataFrame(records).to_csv(O/'missing20_summary.csv',index=False)
    (O/'missing20_table.tex').write_text(tex_table('100-stock construction sensitivity using current primary forecasts. Columns give one-percent FZ0 differences from FHS. The partial stress assigns a 20 percent loss to each missing security-day while holding forecasts fixed; it is a scoring perturbation, not a reconstructed trading or liquidation path. The complete-date column uses the same 1,003 dates for every model. Neither diagnostic identifies missing returns.','tab:missing100',['Model','Baseline','20\\% stress','Complete dates'],rows,'lrrr'))
    (O/'ridge_audit.json').write_text(json.dumps(ridge,indent=2))
    print('CONSTRUCTION',len(records),'ridge max relative',max(r['max_relative_min_eigenvalue_shift'] for r in ridge),flush=True)

if __name__=='__main__':main()

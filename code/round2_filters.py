"""Matched portfolio-filter and tail ablations using existing GARCH parameters."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy import stats
from experiment import es,ewma
from expanded_experiment import score
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round2';checks=[]
def main():
 for panel in ['crsp5','crsp100']:
  rows=[]
  for year in range(2016,2026):
   p=ROOT/'data/crsp_size_portfolios.csv' if panel=='crsp5' else ROOT/'expansion/data'/f'crsp100_{year}.csv'
   f=pd.read_csv(p,index_col=0,parse_dates=True);x=f.values*100;d=x.shape[1]
   tr=(f.index.year>=max(2006,year-11))&(f.index.year<=year-2);te=f.index.year==year;start=np.where(tr)[0][0];stop=np.where(te)[0][-1]+1
   pars=pd.read_csv(ROOT/'results/revision/dynamic_parameters.csv') if panel=='crsp5' else pd.read_csv(ROOT/'expansion/results'/f'crsp100_{year}_dynamic_parameters.csv')
   pars=pars[pars.year.eq(year)].set_index('portfolio')
   weights={'equal':np.ones(d)/d,'small':np.eye(d)[0],'large':np.eye(d)[-1]} if d==5 else {'equal':np.ones(d)/d,'top10':np.r_[np.ones(10)/10,np.zeros(90)],'largest':np.eye(d)[0]}
   original=pd.read_csv(ROOT/'results/revision/dynamic_forecasts.csv.gz') if panel=='crsp5' else pd.read_csv(ROOT/'expansion/results'/f'crsp100_{year}_dynamic.csv.gz')
   for pn,w in weights.items():
    loss=-x@w;r=pars.loc[pn];v=r.initial;vs=[]
    for t in range(start,stop):vs.append(v);v=r.omega+r.arch*(loss[t]-r.mu)**2+r.beta*v
    vs=np.array(vs);sigma=np.sqrt(vs[np.where(te)[0]-start]);z=(loss[tr]-r.mu)/np.sqrt(vs[:tr.sum()])
    sc0=ewma(loss[:,None])[:,0];scm=ewma((loss-r.mu)[:,None])[:,0]
    laws={'GARCH-FHS':(r.mu,sigma,z),'Portfolio-EWMA-FHS':(0.,sc0[te],loss[tr]/sc0[tr]),'EWMA-FHS-matched':(r.mu,scm[te],(loss[tr]-r.mu)/scm[tr])}
    for a in [.01,.025]:
     qt=stats.t.isf(a,r.nu);reconstructed=r.mu+sigma*np.sqrt((r.nu-2)/r.nu)*qt
     prev=original[(original.fold==year)&(original.portfolio==pn)&(original.model=='GARCH-t')&(original.alpha==a)].sort_values('date');assert len(prev)==te.sum()
     err=float(abs(prev['var'].values-reconstructed).max());assert err<1e-8,(panel,year,pn,err)
     checks.append(dict(panel=panel,year=year,portfolio=pn,alpha=a,garch_var_max_error=err))
     for name,(mu,sc,resid) in laws.items():
      q=mu+sc*np.quantile(resid,1-a,method='inverted_cdf');e=mu+sc*es(resid,a);actual=loss[te];assert np.all(e>=q)&np.all(e>0)
      for i,date in enumerate(f.index[te]):rows.append(dict(date=str(date.date()),fold=year,seed=0,model=name,portfolio=pn,alpha=a,loss=actual[i],var=q[i],es=e[i],hit=int(actual[i]>q[i]),fz0=score(actual[i],q[i],e[i],a),es_residual=e[i]-q[i]-max(actual[i]-q[i],0)/a))
  pd.DataFrame(rows).to_csv(O/'results'/(panel+'_matched_filters.csv.gz'),index=False);print(panel,'filter rows',len(rows),flush=True)
 (O/'analysis/filter_state_checks.json').write_text(json.dumps(checks,indent=2))
if __name__=='__main__':main()

"""Retrospective tail checks from retained fits; no model is refitted here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats, special, signal
from round2_selection import dataset
from analyze import bootstrap_means, tex_table

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'referee_round5/analysis'
OUT.mkdir(parents=True,exist_ok=True)

def finite_tail(rho,nu,u,n):
    v,w=special.roots_legendre(n); v=(v+1)/2; w=w/2
    x=stats.t.ppf(u*v,nu)[:,None]; r=np.asarray(rho)[None,:]
    a=stats.t.ppf(u,nu)
    return w@stats.t.cdf((a-r*x)*np.sqrt((nu+1)/((nu+x*x)*(1-r*r))),nu+1)

def tail_checks():
    records=[]; daily=[]; quadrature=[]
    for panel in ['crsp5','crsp100']:
        for year in range(2016,2026):
            data=dataset(panel,year)
            fit=json.loads((ROOT/'expansion/checkpoints'/f'{panel}_{year}_0_Meta-t.meta.json').read_text())
            P=np.asarray(fit['P']); nu=fit['nu']; marg=np.asarray(fit['marg']); d=len(P)
            ii,jj=np.triu_indices(d,1); rho=P[ii,jj]
            lam=2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),nu+1)
            for u in [.05,.01]:
                q=marg[:,1]+marg[:,2]*stats.t.ppf(u,marg[:,0])
                hit=(data['T']<q).astype(float); obs=hit.T@hit/len(hit)/u
                pred=finite_tail(rho,nu,u,256); fine=finite_tail(rho,nu,u,512)
                err=float(np.max(abs(fine-pred)))
                if err>1e-5:
                    pred=fine; fine=finite_tail(rho,nu,u,1024);err=float(np.max(abs(fine-pred)))
                assert err<3e-5, (panel,year,u,err)
                quadrature.append(dict(panel=panel,year=year,u=u,max_abs_refinement=err))
                records.extend(dict(panel=panel,year=year,u=u,i=int(i),j=int(j),rho=float(r),nu_c=nu,lambda_limit=float(l),finite_model=float(p),finite_observed=float(o),n_dates=len(hit)) for i,j,r,l,p,o in zip(ii,jj,rho,lam,fine,obs[ii,jj]))
                k=hit.sum(1); obs_day=k*(k-1)/(d*(d-1)*u)
                daily.extend(dict(panel=panel,year=year,date=str(date.date()),u=u,observed=float(o),predicted=float(fine.mean()),marginal_rate=float(m),lambda_limit=float(lam.mean())) for date,o,m in zip(data['dates'],obs_day,hit.mean(1)))
            print('TAIL',panel,year,flush=True)
    pairs=pd.DataFrame(records); pairs.to_csv(OUT/'tail_pairs.csv.gz',index=False)
    dates=pd.DataFrame(daily);dates.to_csv(OUT/'tail_daily.csv.gz',index=False)
    result=[]
    for (panel,u),g in dates.groupby(['panel','u']):
        diff=(g.observed-g.predicted).values; m=diff.mean()
        boot=bootstrap_means(diff[:,None],block=20,B=9999)[:,0]-m
        low,high=m-np.quantile(boot,.975),m-np.quantile(boot,.025)
        result.append(dict(panel=panel,u=u,lambda_limit=g.lambda_limit.mean(),model=g.predicted.mean(),observed=g.observed.mean(),marginal_rate=g.marginal_rate.mean(),delta=m,low=low,high=high,n=len(g)))
    pd.DataFrame(result).to_csv(OUT/'tail_summary.csv',index=False)
    (OUT/'quadrature_checks.json').write_text(json.dumps(quadrature,indent=2))
    rows=[[r['panel'].replace('crsp',''),f"{100*r['u']:g}",f"{r['lambda_limit']:.3f}",f"{r['model']:.3f}",f"{r['observed']:.3f}",f"{100*r['marginal_rate']:.2f}",f"[{r['low']:.3f}, {r['high']:.3f}]"] for r in result]
    (OUT/'tail_table.tex').write_text(tex_table('Meta-t pairwise downside calibration. The model column is the pair-averaged finite-level $C_{ij}(u,u)/u$, not the asymptotic limit. Observed uses held-out returns and training-fitted marginal quantiles; marginal exceedance rates diagnose their calibration. Intervals concern observed minus predicted, using paired 20-day blocks and no refitting. Dates, not pairs, are the resampling units. These four intervals are descriptive and unadjusted.','tab:r5-tail',['Panel','$u$ (\\%)','$\\overline\\lambda_L$','Model','Observed','Marg. \\%','95\\% difference'],rows,'lrrrrrl'))

def main():
    tail_checks()

if __name__=='__main__':main()

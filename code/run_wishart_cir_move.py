"""Separate numerical audit; fixed parameters and original acceptance gates."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from run_wishart_cir import getfit, save, systematic_uniform
from wishart_cir import initial, integrals, conditional, project, risk
from wishart_cir_move import update_move
from expanded_experiment import score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'completion_audit'


def run(year,n,steps,bank,predictive,warmup,moves):
    tag=f'move_{year}_n{n}_s{steps}_b{bank}_p{predictive}_w{warmup}_m{moves}'
    path=OUT/'results'/(tag+'.json')
    if path.exists():return
    start=time.time();p=getfit('crsp5',year)
    f=pd.read_csv(ROOT/'data/crsp_size_portfolios.csv',index_col=0,parse_dates=True);x=f.to_numpy()*100
    te=np.flatnonzero(f.index.year==year); first=te[0]-warmup
    rng=np.random.default_rng(np.random.SeedSequence([870000,year,bank]));y,v=initial(p,n,rng)
    rows=[];diags=[]
    for t in range(first,te[-1]+1):
        if t>=te[0]:
            prng=np.random.default_rng(np.random.SeedSequence([880000,year,bank,t]))
            idx=systematic_uniform(n,predictive,prng)
            _,_,rr=integrals(y[idx],v[idx],p,prng,steps);m,c=conditional(p,rr[steps])
            for pn,w in {'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}.items():
                loc,sd=project(m,c,w)
                for alpha in [.01,.025]:
                    r=risk(loc,sd,alpha);loss=float(-x[t]@w)
                    rows.append(dict(date=str(f.index[t].date()),portfolio=pn,alpha=alpha,loss=loss,**r,
                                     fz0=float(score(loss,r['var'],r['es'],alpha))))
        y,v,d=update_move(y,v,p,x[t],rng,steps,moves)
        diags.append(dict(date=str(f.index[t].date()),test=bool(t>=te[0]),**d))
        if (t-first+1)%100==0:
            print(tag,'day',t-first+1,'sec',round(time.time()-start,1),'accept',round(d['move_acceptance'],3),flush=True)
            pd.DataFrame(rows).to_csv(path.with_suffix('.partial.csv'),index=False)
    pd.DataFrame(rows).to_csv(path.with_suffix('.csv.gz'),index=False)
    pd.DataFrame(diags).to_csv(OUT/'results'/(tag+'_filter.csv.gz'),index=False)
    save(path,dict(year=year,particles=n,steps=steps,bank=bank,predictive=predictive,warmup=warmup,
                   moves=moves,seconds=time.time()-start,parameters=p,n_test=len(te),method='exact-ratio independent resample-move'))
    print('DONE',tag,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,default=2025);ap.add_argument('--particles',type=int,default=8192)
    ap.add_argument('--steps',type=int,default=16);ap.add_argument('--bank',type=int,required=True)
    ap.add_argument('--predictive',type=int,default=32768);ap.add_argument('--warmup',type=int,default=504)
    ap.add_argument('--moves',type=int,default=3);a=ap.parse_args();run(a.year,a.particles,a.steps,a.bank,a.predictive,a.warmup,a.moves)

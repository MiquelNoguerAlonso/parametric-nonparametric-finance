"""Validate completeness, time separation, calibration accounting and Heston MGF."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,passed,detail):
    checks.append({'check':name,'passed':bool(passed),'detail':detail})
    if not passed:raise AssertionError(name+': '+str(detail))

def main():
    archives=list((ROOT/'data').glob('*.npz'))+list((ROOT/'checkpoints').glob('*.npz'))
    for path in archives:
        with np.load(path) as archive:
            for key in archive.files:
                assert np.isfinite(archive[key]).all(),(path.name,key)
    check('Readable finite data and checkpoint archives',True,len(archives))
    for study,expected in [('t',60),('metat',66),('heston',54),('crsp',200)]:
        rec=[]
        for p in (ROOT/'results').glob(study+'_*.json'):
            v=json.loads(p.read_text())
            if v.get('study')==study:rec.append(v)
        check('Completed model records: '+study,len(rec)==expected,{'actual':len(rec),'expected':expected})
        if study=='crsp':
            check('Chronological train/validation/test separation',all(r['train_end']<r['validation_start']<=r['validation_end']<r['test_start'] for r in rec),True)
        else:
            errors=[abs(m['total_error']-m['fit_error']-m['sample_error']) for r in rec for m in r['metrics']]
            check('ES discrepancy identity: '+study,max(errors)<1e-10,max(errors))
    fc=pd.read_csv(ROOT/'results'/'crsp_all_forecasts.csv.gz')
    check('Unique forecast keys',not fc.duplicated(['date','seed','model','portfolio','alpha']).any(),len(fc))
    check('ES at least VaR',bool((fc.es>=fc['var']-1e-9).all()),float((fc.es-fc['var']).min()))
    s=np.log(fc.es)+fc['var']/fc.es-1+np.maximum(fc.loss-fc['var'],0)/(fc.alpha*fc.es)
    check('Independent FZ0 reconstruction',float(np.max(np.abs(s-fc.fz0)))<1e-9,float(np.max(np.abs(s-fc.fz0))))
    counts=fc.groupby(['model','seed','portfolio','alpha']).size()
    check('Same out-of-sample dates for every model',counts.nunique()==1,counts.unique().tolist())
    src=json.loads((ROOT/'data'/'crsp_data_audit.json').read_text())
    check('Unique annual CRSP keys',all(x['duplicates']==0 for x in src['files']),src['rows'])
    # Independently solve the scalar exact Heston Riccati equations; compare
    # a small exponential moment against the approximate joint-bank simulator.
    x=np.load(ROOT/'data'/'heston_truth.npz')['X'];R=x.reshape(-1,5,5)[:,0,:].sum(1)*np.sqrt(.04/252)
    for u in [-10.,0.,1.,10.]:
        def fun(t,y):
            C,D=y;return [3*.04*D,.5*.5**2*D*D+(-.7*.5*u-3)*D+.5*(u*u-u)]
        sol=solve_ivp(fun,[0,5/252],[0.,0.],rtol=1e-11,atol=1e-13)
        C,D=sol.y[:,-1];exact=np.exp(C)*(1-(.04/.96)*D)**(-.96)
        vals=np.exp(u*R.astype(float));mc=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)))
        # Fixed bank approximation adds sampling/discretization uncertainty:
        # this is a diagnostic tolerance, not a formal distributional test.
        tolerance=max(5*se,.001)
        check('Heston exact MGF versus simulator u='+str(u),abs(exact-mc)<tolerance,
            {'exact':float(exact),'monte_carlo':mc,'mc_se':se,'tolerance':tolerance})
    (ROOT/'results'/'output_checks.json').write_text(json.dumps(checks,indent=2))
    print(f'{len(checks)} output checks passed')

if __name__=='__main__':main()

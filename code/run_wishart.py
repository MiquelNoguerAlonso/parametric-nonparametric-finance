"""Reproduce the additional fitted Heston--Wishart benchmark."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from wishart_model import fit_smm, sample_windows, filter_series
from experiment import heston, es, stylised
from expanded_experiment import score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'wishart_extension'
NAME='Heston-Wishart'
for directory in ['results','fits','analysis','logs']:(OUT/directory).mkdir(exist_ok=True,parents=True)


def save(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def fit(study, unit, windows, ito_scale, daily=None):
    path=OUT/'fits'/f'{study}_{unit}.json'
    if path.exists():
        previous=json.loads(path.read_text())
        assert previous['training_sha256']==hashlib.sha256(windows.tobytes()).hexdigest()
        return previous['parameters']
    start=time.time()
    par,diag=fit_smm(windows,seed=720000+unit,ito_scale=ito_scale,daily=daily)
    save(path,dict(study=study,unit=unit,parameters=par,fit=diag,seconds=time.time()-start,
                  training_sha256=hashlib.sha256(windows.tobytes()).hexdigest()))
    print(study,unit,'fit',par['beta'],par['kappa'],par['rho'],'seconds',round(time.time()-start,1),flush=True)
    return par


def synthetic(unit, substeps=4, sensitivity=False):
    tag=f'heston_{unit}_'+('refined' if sensitivity else 'primary')
    path=OUT/'results'/(tag+'.json')
    if path.exists():return
    start=time.time();x=heston(4000,50000+unit).astype(float).reshape(-1,5,5)
    par=fit('heston',unit,x,np.sqrt(.04/252))
    losses=[];moments=[]
    batches=4 if sensitivity else 10
    for b in range(batches):
        samples=sample_windows(par,100000,900000+unit*100+b,substeps=substeps)
        losses.append(-samples.sum((1,2))/5)
        moments.append(stylised(samples.reshape(-1,25)))
        print(tag,'batch',b+1,'of',batches,flush=True)
    loss=np.concatenate(losses)
    reference=json.loads((ROOT/'expansion/data/heston_reference.json').read_text());metrics=[]
    for a in [.01,.001]:
        est=es(loss,a);target=reference[f'equal_{a}']['es'];emp=es(-x.sum((1,2))/5,a)
        metrics.append(dict(portfolio='equal',alpha=a,var=float(np.quantile(loss,1-a,method='inverted_cdf')),
            es=est,truth=target,sample_es=emp,fit_error=est-emp,sample_error=emp-target,total_error=est-target,
            scenario_mc_se=float(np.std([es(v,a) for v in losses],ddof=1)/np.sqrt(batches)),
            reference_mc_se=reference[f'equal_{a}']['mc_se']))
    validation=heston(1000,60000+unit).astype(float).reshape(-1,5,5)
    vsc=sample_windows(par,20000,810000+unit,substeps=substeps)
    vl=-vsc.sum((1,2))/5;q=float(np.quantile(vl,.99,method='inverted_cdf'));e=es(vl,.01)
    save(path,dict(study='heston',dataset=unit,seed=0,model=NAME,n_scenarios=len(loss),n_train=4000,n_val=1000,
        n_test=None,nll=None,nll_kind='not evaluated',parameters=par,metrics=metrics,
        validation_fz0=float(np.mean(score(-validation.sum((1,2))/5,q,e))),validation_used_for_selection=False,
        dependence={k:float(np.mean([a[k] for a in moments])) for k in moments[0]},substeps=substeps,
        seconds=time.time()-start,checkpoint_selection='training-only fixed-grid simulated moments'))
    print(tag,'finished',metrics[0]['total_error'],'seconds',round(time.time()-start,1),flush=True)


def crsp(year, sensitivity=False):
    tag=f'crsp5_{year}_'+('refined' if sensitivity else 'primary');path=OUT/'results'/(tag+'.json')
    if path.exists():return
    start=time.time();source=ROOT/'data/crsp_size_portfolios.csv'
    f=pd.read_csv(source,index_col=0,parse_dates=True);x=f.to_numpy()*100
    tr=(f.index.year>=max(2006,year-11))&(f.index.year<=year-2);va=f.index.year==year-1;te=f.index.year==year
    assert f.index[tr].max()<f.index[va].min()<f.index[te].min()
    windows=np.lib.stride_tricks.sliding_window_view(x[tr],5,axis=0).copy()
    par=fit('crsp5',year,windows,0.,daily=x[tr])
    first=np.flatnonzero(tr)[0];end=np.flatnonzero(te)[-1]+1;test_start=np.flatnonzero(te)[0]-first
    weights={'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}
    particles,predictive,substeps=(1024,8192,8) if sensitivity else (512,4096,4)
    pred,diag=filter_series(par,x[first:end],730000+year,n_particles=particles,n_predictive=predictive,
        substeps=substeps,forecast_from=test_start,weights=weights,
        callback=lambda t:print(tag,'filter day',t,flush=True))
    rows=[]
    for item in pred:
        t=first+item['t'];actual=x[t]
        for (pn,a),(q,e) in item['forecasts'].items():
            loss=float(-actual@weights[pn])
            rows.append(dict(date=str(f.index[t].date()),fold=year,model=NAME,seed=0,portfolio=pn,alpha=a,
                loss=loss,var=q,es=e,hit=int(loss>q),fz0=float(score(loss,q,e,a)),
                es_residual=float(e-q-max(loss-q,0)/a)))
    frame=pd.DataFrame(rows);frame.to_csv(path.with_suffix('.csv.gz'),index=False)
    pd.DataFrame(diag).to_csv(OUT/'results'/(tag+'_filter.csv.gz'),index=False)
    testdiag=[d for d in diag if d['t']>=test_start]
    save(path,dict(study='crsp5',fold=year,model=NAME,seed=0,parameters=par,n_train=int(tr.sum()),
        n_val=int(va.sum()),n_test=int(te.sum()),train_start=str(f.index[tr].min().date()),
        train_end=str(f.index[tr].max().date()),validation_start=str(f.index[va].min().date()),
        validation_end=str(f.index[va].max().date()),test_start=str(f.index[te].min().date()),test_end=str(f.index[te].max().date()),
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),particles=particles,predictive_paths=predictive,
        return_integration_substeps=substeps,forecast_method='conditional Gaussian mixture quantile/ES',
        median_test_ess=float(np.median([d['ess'] for d in testdiag])),min_test_ess=float(min(d['ess'] for d in testdiag)),
        fraction_test_ess_under_10=float(np.mean([d['ess']<10 for d in testdiag])),
        median_test_parent_ess=float(np.median([d['parent_ess'] for d in testdiag])),
        min_test_parent_ess=float(min(d['parent_ess'] for d in testdiag)),
        filter_proposal='observation-guided auxiliary, frozen Gaussian bridge with importance correction',
        mean_test_approximate_nll=float(np.mean([d['nll'] for d in testdiag])),
        seconds=time.time()-start,parameter_selection='training-only fixed-grid simulated moments',
        test_information='each realized return enters the state filter only after its forecast'))
    print(tag,'finished FZ',frame[(frame.portfolio=='equal')&(frame.alpha==.01)].fz0.mean(),'seconds',round(time.time()-start,1),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['heston','crsp5'],required=True)
    parser.add_argument('--units',type=int,nargs='+',required=True);parser.add_argument('--refined',action='store_true');a=parser.parse_args()
    for unit in a.units:
        if a.study=='heston':synthetic(unit,8 if a.refined else 4,a.refined)
        else:crsp(unit,a.refined)

if __name__=='__main__':main()

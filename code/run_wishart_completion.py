"""Numerical-only Wishart completion runs, with immutable fitted parameters."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from wishart_tempered import TemperedWishart,jr,j,roots,jax
from wishart_block import BlockWishart
from expanded_experiment import score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wishart_completion'
@jax.jit
def canonicalise(y):
    sig=y.swapaxes(1,2)@y
    return j.zeros_like(y).at[:,:y.shape[2],:].set(roots(sig)[0])

def run(year,bank,particles,steps,predictive,warmup,moves=3,limit=None,savepaths=False,antithetic=False,defensive=0.,lag=1,moment_auxiliary=False,aux_power=1.,canonical=False,gauge_paths=False):
    tag=f'y{year}_b{bank}_n{particles}_s{steps}_p{predictive}_w{warmup}_m{moves}'+(f'_limit{limit}' if limit else '')+('_anti' if antithetic else '')+(f'_def{int(defensive*100)}' if defensive else '')+(f'_lag{lag}' if lag>1 else '')+('_moment' if moment_auxiliary else '')+(f'_ap{int(aux_power*100)}' if aux_power!=1. else '')+('_canon' if canonical else '')+('_pathgauge' if gauge_paths else '')
    path=OUT/'results'/(tag+'.json')
    if path.exists():print('EXISTS',tag,flush=True);return
    begin=time.time();f=pd.read_csv(ROOT/'data/crsp_size_portfolios.csv',index_col=0,parse_dates=True);x=f.to_numpy()*100
    te=np.flatnonzero(f.index.year==year);first=max(0,te[0]-warmup);end=te[-1]+1
    if limit is not None:end=min(end,te[0]+limit)
    parfile=ROOT/'wishart_extension/fits'/f'crsp5_{year}.json';par=json.loads(parfile.read_text())['parameters']
    scales={0:(),1:(.4,),2:(.25,.55),3:(.2,.4,.6),5:(.4,.7,1.,.7,1.)}[moves]
    kernel=TemperedWishart(par,steps,scales=scales,antithetic=antithetic,defensive=defensive,moment_auxiliary=moment_auxiliary,aux_power=aux_power,gauge_paths=gauge_paths)
    block_kernels={l:BlockWishart(par,steps,lag=l,scales=scales,antithetic=antithetic,moment_auxiliary=moment_auxiliary,aux_power=aux_power,gauge_paths=gauge_paths) for l in range(2,lag+1)}
    history=[];observations=[]
    key=jr.PRNGKey(910000+year*100+bank)
    # Date-indexed streams separate observation updates and predictive sampling.
    y=kernel.initial(jr.fold_in(key,0),particles);y=canonicalise(y) if canonical else y;rows=[];diagnostics=[];params=[]
    weights={'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}
    ww=j.asarray(list(weights.values()));aa=j.array([.01,.025]);checkpoint=OUT/'checkpoints'/(tag+'.npz')
    state=OUT/'checkpoints'/(tag+'.json')
    if checkpoint.exists() and state.exists():
        data=np.load(checkpoint);y=j.asarray(data['y']);history=[j.asarray(v) for v in data['history']] if 'history' in data else [];observations=list(data['observations']) if 'observations' in data else [];c=json.loads(state.read_text());first_run=c['next_index'];rows=c['rows'];diagnostics=c['diagnostics'];print('RESUMING',tag,first_run,flush=True)
    else:first_run=first
    for t in range(first_run,end):
        if t>=te[0]:
            mean,cov=kernel.predict(y,jr.fold_in(key,2*t+1),predictive)
            q,e=kernel.risk(mean,cov,ww,aa);q=np.asarray(q);e=np.asarray(e)
            if not (np.isfinite(q).all() and np.isfinite(e).all() and (e>=q).all()):raise FloatingPointError('invalid predictive risk')
            if savepaths:
                pp=OUT/'checkpoints'/tag;pp.mkdir(exist_ok=True)
                # Loss-mixture parameters permit exact pooling across banks.
                loc=-np.asarray(mean)@np.asarray(ww).T;sd=np.sqrt(np.einsum('pi,nij,pj->np',np.asarray(ww),np.asarray(cov),np.asarray(ww)))
                np.savez_compressed(pp/f'{t}.npz',loc=loc,sd=sd)
            for i,pn in enumerate(weights):
                for aidx,alpha in enumerate([.01,.025]):
                    loss=float(-x[t]@weights[pn]);v=float(q[i,aidx]);es=float(e[i,aidx])
                    rows.append(dict(date=str(f.index[t].date()),fold=year,model='Heston-Wishart',seed=bank,portfolio=pn,alpha=alpha,loss=loss,var=v,es=es,hit=int(loss>v),fz0=float(score(loss,v,es,alpha)),es_residual=es-v-max(loss-v,0)/alpha))
        if lag>1:
            history.append(y);observations.append(x[t]);history=history[-lag:];observations=observations[-lag:]
            if len(history)>1:y,d=block_kernels[len(history)].update(history[0],j.asarray(observations),jr.fold_in(key,2*t+2))
            else:y,d=kernel.update(y,j.asarray(x[t]),jr.fold_in(key,2*t+2))
        else:y,d=kernel.update(y,j.asarray(x[t]),jr.fold_in(key,2*t+2))
        if canonical:y=canonicalise(y)
        diag=np.asarray(d)
        if not np.isfinite(diag).all() or diag[0]<1.-1e-10 or not np.isfinite(np.asarray(y)).all():raise FloatingPointError((tag,t,diag))
        diagnostics.append(dict(date=str(f.index[t].date()),t=int(t),test=bool(t>=te[0]),gamma=diag[0],stages=diag[1],stage_min_ess=diag[2],parent_ess=diag[3],raw_correction_ess=diag[4],move_acceptance=diag[5],nll=diag[6]))
        if (t-first+1)%100==0 or t==end-1:
            np.savez_compressed(checkpoint,y=np.asarray(y),history=np.asarray(history),observations=np.asarray(observations));state.write_text(json.dumps(dict(next_index=t+1,rows=rows,diagnostics=diagnostics)))
            print(tag,'day',t-first+1,'/',end-first,'stages',diag[1],'accept',round(diag[5],3),'seconds',round(time.time()-begin,1),flush=True)
    pred=pd.DataFrame(rows);dg=pd.DataFrame(diagnostics);pred.to_csv(path.with_suffix('.csv.gz'),index=False);dg.to_csv(OUT/'results'/(tag+'_filter.csv.gz'),index=False)
    dtest=dg[dg.test];primary=pred[(pred.portfolio=='equal')&(pred.alpha==.01)]
    result=dict(tag=tag,year=year,bank=bank,particles=particles,substeps=steps,predictive=predictive,warmup=warmup,moves=moves,antithetic=antithetic,defensive=defensive,lag=lag,moment_auxiliary=moment_auxiliary,aux_power=aux_power,canonical=canonical,gauge_paths=gauge_paths,normalizer_kind=('joint block' if lag>1 else 'daily'),parameters=par,parameter_file_sha256=hashlib.sha256(parfile.read_bytes()).hexdigest(),data_sha256=hashlib.sha256((ROOT/'data/crsp_size_portfolios.csv').read_bytes()).hexdigest(),n_forecast_dates=len(primary),forecast_start=str(primary.date.min()),forecast_end=str(primary.date.max()),mean_fz0=float(primary.fz0.mean()),hits=int(primary.hit.sum()),min_stage_ess=float(dtest.stage_min_ess.min()),mean_stages=float(dtest.stages.mean()),max_stages=float(dtest.stages.max()),mean_move_acceptance=float(dtest.move_acceptance.mean()),min_parent_ess=float(dtest.parent_ess.min()),raw_fraction_ess_lt10=float((dtest.raw_correction_ess<10).mean()),seconds=time.time()-begin,forecasts=str(path.with_suffix('.csv.gz').relative_to(ROOT)),diagnostics=str((OUT/'results'/(tag+'_filter.csv.gz')).relative_to(ROOT)),retrospective_numerical_repair=True)
    path.write_text(json.dumps(result,indent=2));print('DONE',tag,result['mean_fz0'],'seconds',round(result['seconds'],1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--years',nargs='+',type=int,required=True);ap.add_argument('--banks',nargs='+',type=int,default=[0]);ap.add_argument('--particles',type=int,default=1024);ap.add_argument('--steps',type=int,default=8);ap.add_argument('--predictive',type=int,default=8192);ap.add_argument('--warmup',type=int,default=504);ap.add_argument('--moves',type=int,default=3);ap.add_argument('--limit',type=int);ap.add_argument('--savepaths',action='store_true');ap.add_argument('--antithetic',action='store_true');ap.add_argument('--defensive',type=float,default=0.);ap.add_argument('--lag',type=int,default=1);ap.add_argument('--moment-auxiliary',action='store_true');ap.add_argument('--aux-power',type=float,default=1.);ap.add_argument('--canonical',action='store_true');ap.add_argument('--gauge-paths',action='store_true');a=ap.parse_args()
    for year in a.years:
        for bank in a.banks:run(year,bank,a.particles,a.steps,a.predictive,a.warmup,a.moves,a.limit,a.savepaths,a.antithetic,a.defensive,a.lag,a.moment_auxiliary,a.aux_power,a.canonical,a.gauge_paths)
if __name__=='__main__':main()

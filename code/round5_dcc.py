"""Training-only five-basket CCC/DCC controls with analytic portfolio risk.

Conditional Gaussian and variance-normalized Student laws share the existing
coordinate GARCH scales. No test value selects parameters or specifications.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy import stats,optimize,signal,special
from revision_analysis import inference
from analyze import tex_table
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'referee_round5'; A=OUT/'analysis'
(OUT/'dcc').mkdir(parents=True,exist_ok=True)

def correlations(z,Qbar,a,b):
    outer=z[:,:,None]*z[:,None,:]
    delta=signal.lfilter([a],[1.,-b],outer-Qbar,axis=0)
    Q=np.concatenate([Qbar[None],Qbar[None]+delta[:-1]])
    den=np.sqrt(np.diagonal(Q,axis1=1,axis2=2)); R=Q/(den[:,:,None]*den[:,None,:])
    return R

def nll(z,R,nu=None):
    ld=np.linalg.slogdet(R)[1]; q=np.einsum('ti,ti->t',z,np.linalg.solve(R,z[...,None])[...,0]); d=z.shape[1]
    if nu is None:return .5*(d*np.log(2*np.pi)+ld+q)
    return .5*ld+.5*d*np.log((nu-2)*np.pi)+special.gammaln(nu/2)-special.gammaln((nu+d)/2)+.5*(nu+d)*np.log1p(q/(nu-2))

def fit(z,Qbar,dynamic,student):
    fixed=correlations(z,Qbar,0.,0.); starts=[(.01,.95),(.05,.9),(.1,.8)] if dynamic else [(0.,0.)]
    def objective(p):
        a,b=p[:2] if dynamic else (0.,0.); nu=p[-1] if student else None
        return float(np.mean(nll(z,correlations(z,Qbar,a,b) if dynamic else fixed,nu)))
    bounds=([(0.,.999),(0.,.999)] if dynamic else [])+([(2.05,200.)] if student else [])
    if not bounds:return [],dict(success=True,objective=objective([]),starts=[])
    trials=[]
    for a,b in starts:
        p=([a,b] if dynamic else [])+([8.] if student else [])
        result=optimize.minimize(objective,p,method='SLSQP',bounds=bounds,constraints=([{'type':'ineq','fun':lambda p:.999-p[0]-p[1]}] if dynamic else []),options={'ftol':1e-10,'maxiter':400})
        trials.append(dict(parameters=result.x.tolist(),objective=float(result.fun),success=bool(result.success),message=str(result.message),iterations=int(result.nit)))
    good=[r for r in trials if r['success'] and np.isfinite(r['objective'])]
    if not good:raise RuntimeError(trials)
    best=min(good,key=lambda r:r['objective'])
    return best['parameters'],dict(success=True,objective=best['objective'],starts=trials)

def main():
    rows=[];checks=[]
    for year in range(2016,2026):
        data=np.load(ROOT/'referee_round3/filters'/f'crsp5_{year}.npz')
        x=data['returns']; means=data['garch_means']; scales=data['garch_scales']; z=(x-means)/scales
        train=data['train'];test=data['test'];Qbar=z[train].T@z[train]/train.sum()+1e-10*np.eye(5)
        for dynamic in [False,True]:
            for student in [False,True]:
                name=('DCC' if dynamic else 'CCC')+('-t' if student else '-G'); path=OUT/'dcc'/f'{year}_{name}.json'
                if path.exists(): saved=json.loads(path.read_text()); p=saved['parameters']; diag=saved['optimizer']
                else:
                    p,diag=fit(z[train],Qbar,dynamic,student)
                    path.write_text(json.dumps(dict(year=year,model=name,parameters=p,optimizer=diag,Qbar=Qbar.tolist(),n_train=int(train.sum()),train_end=str(data['dates'][train][-1]),test_start=str(data['dates'][test][0]),parameter_fit='training only; validation updates states'),indent=2))
                a,b=p[:2] if dynamic else (0.,0.);nu=p[-1] if student else None
                R=correlations(z,Qbar,a,b);Rt=R[test];w=np.ones(5)/5;v=scales[test]*w
                sd=np.sqrt(np.einsum('ti,tij,tj->t',v,Rt,v)); offset=-means@w; loss=-x[test]@w
                logloss=nll(z[test],Rt,nu)+np.log(scales[test]).sum(1)
                # Literal recursion verifies the vectorized filtering implementation.
                Q=Qbar.copy();maxerr=0.
                for j in range(len(z)):
                    den=np.sqrt(np.diag(Q)); maxerr=max(maxerr,float(abs(R[j]-Q/np.outer(den,den)).max()))
                    Q=(1-a-b)*Qbar+a*np.outer(z[j],z[j])+b*Q
                assert maxerr<1e-10 and np.linalg.eigvalsh(R).min()>0
                # A forecast at t is unchanged by modifying observations at t or later.
                cut=np.where(test)[0][0]+10;zz=z.copy();zz[cut:]+=3
                assert np.max(abs(correlations(zz,Qbar,a,b)[:cut+1]-R[:cut+1]))<1e-12
                checks.append(dict(year=year,model=name,recursion_max_error=maxerr,minimum_eigenvalue=float(np.linalg.eigvalsh(R).min()),predictability_check=True))
                for alpha in [.01,.025]:
                    if student:
                        quant=stats.t.ppf(1-alpha,nu);norm=np.sqrt((nu-2)/nu)
                        q=offset+sd*norm*quant;e=offset+sd*norm*(nu+quant*quant)/(nu-1)*stats.t.pdf(quant,nu)/alpha
                    else:
                        quant=stats.norm.ppf(1-alpha);q=offset+sd*quant;e=offset+sd*stats.norm.pdf(quant)/alpha
                    assert (e>q).all() and (e>0).all()
                    score=np.log(e)+q/e-1+np.maximum(loss-q,0)/(alpha*e)
                    rows.extend(dict(date=str(date),year=year,model=name,alpha=alpha,loss=float(l),var=float(vv),es=float(ee),fz0=float(s),hit=int(l>vv),nll=float(n)) for date,l,vv,ee,s,n in zip(data['dates'][test],loss,q,e,score,logloss))
                print('DCC',year,name,p,flush=True)
    forecasts=pd.DataFrame(rows);forecasts.to_csv(A/'dcc_forecasts.csv.gz',index=False)
    (A/'dcc_checks.json').write_text(json.dumps(checks,indent=2))
    baseline=pd.read_csv(ROOT/'referee_round4/analysis/crsp5_augmented_daily.csv.gz',index_col=0)
    baseline.index=pd.to_datetime(baseline.index).strftime('%Y-%m-%d')
    primary=pd.read_csv(ROOT/'referee_round4/analysis/crsp5_forecasts.csv.gz')
    g=forecasts[forecasts.alpha==.01]; new=g.pivot(index='date',columns='model',values='fz0');new.index=pd.to_datetime(new.index).strftime('%Y-%m-%d')
    local=pd.concat([baseline[['Historical','Coord-Historical']],new],axis=1)
    assert len(local)==2514 and local.notna().all().all()
    contrasts=[]
    for ref in ['Historical','Coord-Historical']:
        r=inference(local[[ref,*new.columns]],ref);r=r[r.model!=ref];r['fz0']=r.model.map(new.mean());contrasts.append(r)
    result=pd.concat(contrasts);from analyze import holm
    result['p_holm']=holm(result.p.values);result.to_csv(A/'dcc_summary.csv',index=False)
    # Encompassing retrospective primary-plus-DCC family, without replacing primary inference.
    original=primary[(primary.alpha==.01)&(primary.portfolio=='equal')].groupby(['date','model']).fz0.mean().unstack();original.index=pd.to_datetime(original.index).strftime('%Y-%m-%d')
    broad=pd.concat([original,new],axis=1);assert broad.shape==(2514,21) and broad.notna().all().all()
    inference(broad,'Historical').to_csv(A/'dcc_broad21.csv',index=False)
    rows=[]
    for r in result.itertuples():
        rows.append([r.model,'EWMA-FHS' if r.benchmark=='Historical' else 'GARCH-FHS',f'{r.fz0:.3f}',f'{r.delta:.3f}',f'[{r.low:.3f}, {r.high:.3f}]',f'{r.p_holm:.3f}'])
    (A/'dcc_table.tex').write_text(tex_table('Five-basket correlation controls, 2,514 dates, one-percent equal-portfolio FZ0. CCC and DCC share coordinate-GARCH scales; G and t denote Gaussian and variance-normalized Student laws. The second reference is coordinate-GARCH-FHS. Eight contrasts are Holm-adjusted together. Fits and this diagnostic family are retrospective; the original primary family is unchanged.','tab:r5-dcc',['Model','Reference','FZ0','$\\Delta$','95\\% interval','Holm/8'],rows,'llrrlr'))
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()

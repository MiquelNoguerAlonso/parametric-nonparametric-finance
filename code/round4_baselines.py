"""Fit and evaluate the two added statistical joint laws."""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from round4_data import get_data,logdet,joint_event,ROOT,OUT
from round4_parametric import JointStudent,ShrinkGaussian
from expanded_experiment import score
from experiment import es

def analytic_forecasts(model,data,name,unit):
    rows=[];s=data['scales'];means=data.get('means',np.zeros(len(data['w'])))
    for pn,w in data['weights'].items():
        proj=np.einsum('tij,i->tj',s,w) if s.ndim==3 else s*w
        actual=-data['returns']@w;offset=-float(means@w)
        for a in [.01,.025]:
            q,e=model.risk(proj,a);q+=offset;e+=offset
            if (e<=0).any():raise RuntimeError('Nonpositive analytic ES')
            ss=score(actual,q,e,a);res=e-q-np.maximum(actual-q,0)/a
            for j,date in enumerate(data['dates']):rows.append(dict(date=str(date.date()),fold=unit,model=name,seed=0,portfolio=pn,alpha=a,loss=actual[j],var=q[j],es=e[j],hit=int(actual[j]>q[j]),fz0=ss[j],es_residual=res[j]))
    return pd.DataFrame(rows)

def work(study,unit,family='original'):
    data=get_data(study,unit,family)
    for name,cls in [('Student-MLE',JointStudent),('Gaussian-LW',ShrinkGaussian)]:
        tag=f'{family}_{study}_{unit}_{name}';result=OUT/'results'/(tag+'.json');path=OUT/'checkpoints'/(tag+'.json')
        if result.exists():continue
        start=time.time();model=cls.load(path) if path.exists() else cls().fit(data['X']);model.save(path)
        nll=-model.logpdf(data['T'])+logdet(data);np.savez_compressed(OUT/'pointwise'/(tag+'.npz'),nll=nll)
        r=dict(study=study,unit=unit,family=family,model=name,nll=float(nll.mean()),pointwise=f'referee_round4/pointwise/{tag}.npz',checkpoint=f'referee_round4/checkpoints/{tag}.json')
        if study.startswith('crsp'):
            pred=analytic_forecasts(model,data,name,unit);fn=tag+'_forecasts.csv.gz';pred.to_csv(OUT/'results'/fn,index=False);r['forecasts']='referee_round4/results/'+fn
            r['joint_event']=joint_event(model.sample(100000,190000+10*unit),data)
        else:
            truth=json.loads((ROOT/'expansion/data'/f'{study}_reference.json').read_text());metrics=[]
            for a in [.01,.001]:
                q,e=model.risk(data['w'],a);target=truth[f'equal_{a}']['es'];emp=es(-data['X']@data['w'],a)
                metrics.append(dict(portfolio='equal',alpha=a,var=float(q[0]),es=float(e[0]),truth=target,sample_es=emp,total_error=float(e[0]-target),fit_error=float(e[0]-emp),sample_error=emp-target,scenario_mc_se=0.,evaluation='analytic linear projection'))
            r['metrics']=metrics
        r['seconds']=time.time()-start;r['df']=getattr(model,'nu',None);r['shrinkage']=getattr(model,'shrinkage',None)
        result.write_text(json.dumps(r,indent=2));print('BASELINE',tag,'NLL',r['nll'],'df',r['df'],'sec',round(r['seconds'],2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--unit',required=True,type=int);p.add_argument('--family',default='original');a=p.parse_args();work(a.study,a.unit,a.family)

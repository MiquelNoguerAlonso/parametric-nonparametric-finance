"""Repaired forecasts, matched density uncertainty, and joint-event diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from analyze import tex_table,fmt,holm,bootstrap_means
from revision_analysis import inference,mcs,calibration,pformat
from round4_data import ROOT,OUT
from round4_repair import names
O=OUT/'analysis';R=OUT/'results'
LABEL={'Historical':'FHS','VAE-t':'VAE-tlatent','Student-MLE':'Student-$t$ MLE','Gaussian-LW':'Gaussian-LW','Portfolio-EWMA-FHS':'Port.-EWMA-FHS'}
def lab(s):return LABEL.get(s,s)
def read(p):return json.loads(p.read_text())
def record(study,unit,name,family='original',seed=0):
    neural=name.startswith(('Flow','VAE','WGAN'));tag=f'{family}_{study}_{unit}_'+(f'{seed}_' if neural else '')+name
    return read(R/(tag+'.json'))
def evaluation(study,unit,name,rule='fz1',family='original',seed=0):
    z=record(study,unit,name,family,seed);return z['evaluations'][rule] if 'evaluations' in z else z
def pred(ev,alpha=.01,portfolio='equal'):
    x=pd.read_csv(ROOT/ev['forecasts'],parse_dates=['date']);x=x[(x.alpha==alpha)&(x.portfolio==portfolio)].set_index('date').sort_index()
    assert x.index.is_unique;return x
def interval(r):return f'[{r.low:.3f}, {r.high:.3f}]'
def contrast(s):
    x=pd.DataFrame({'x':s,'zero':np.zeros(len(s))});return inference(x,'zero').set_index('model').loc['x'].to_dict()
def adjust(rows):
    for x,p in zip(rows,holm([v['p'] for v in rows])):x['p_holm']=float(p)
def matrix(panel,alpha=.01,seed=None):
    x=pd.read_csv(O/f'{panel}_forecasts.csv.gz',parse_dates=['date']);x=x[(x.portfolio=='equal')&(x.alpha==alpha)]
    if seed is not None:x=x[x.seed==seed]
    return x.groupby(['date','model']).fz0.mean().unstack().sort_index(),x

def collect():
    for panel in ['crsp5','crsp100']:
        old=pd.read_csv(ROOT/'referee_round2/analysis'/f'{panel}_forecasts.csv.gz',parse_dates=['date'])
        nn=names(panel,'original');frames=[old[~old.model.isin(nn)]]
        for year in range(2016,2026):
            for name in nn:
                for seed in ([0,1] if panel=='crsp5' else [0]):
                    ev=evaluation(panel,year,name,seed=seed);f=pd.read_csv(ROOT/ev['forecasts'],parse_dates=['date']);f['model']=name;f['seed']=seed;frames.append(f)
            for name in ['Student-MLE','Gaussian-LW']:
                f=pd.read_csv(ROOT/evaluation(panel,year,name)['forecasts'],parse_dates=['date']);frames.append(f)
        allx=pd.concat(frames,ignore_index=True)
        assert len(allx.model.unique())==17
        assert not allx.duplicated(['date','model','seed','portfolio','alpha']).any()
        allx.to_csv(O/f'{panel}_forecasts.csv.gz',index=False)

def primary():
    rows=[];power=[];cal=[];robust=[];concentrated=[];mc_sensitivity=[];pairs=[]
    for panel in ['crsp5','crsp100']:
        for alpha in [.01,.025]:
            scores,raw=matrix(panel,alpha);assert scores.shape==(2514,17) and scores.notna().all().all()
            inf=inference(scores,'Historical').set_index('model');mc=mcs(scores).set_index('model');table=[]
            for name in scores.mean().sort_values().index:
                a=inf.loc[name];f=raw[raw.model==name]
                r=dict(panel=panel,alpha=alpha,model=name,fz0=float(scores[name].mean()),hits_percent=float(100*f.hit.mean()),es_moment=float(f.es_residual.mean()/f.es.mean()),delta=float(a.delta),low=float(a.low),high=float(a.high),p=float(a.p),p_holm=float(a.p_holm),mcs_p=float(mc.loc[name,'p_mcs']))
                rows.append(r);table.append([lab(name),fmt(r['fz0']),fmt(r['hits_percent'],2),fmt(r['es_moment']),fmt(r['delta']),interval(a),pformat(a.p_holm),pformat(r['mcs_p'])])
            suffix='primary' if alpha==.01 else 'secondary'
            caption=f'Repaired {panel.replace("crsp","")}-coordinate equal-portfolio {100*alpha:g}\\% forecasts on 2,514 dates. Seventeen models, sixteen within-panel FHS contrasts. Neural candidates include initialization and early updates; Wishart CRSP is excluded. Basic intervals and matching tests use 9,999 circular resamples with 20-day blocks. MCS membership is not a ranking of retained models.'
            (O/f'{panel}_{suffix}.tex').write_text(tex_table(caption,'tab:r4-'+panel+'-'+suffix,['Model','FZ0','Hits \\%','ES mom.','$\\Delta$ FHS','95\\% interval','Holm $p$','MCS $p$'],table))
            if alpha!=.01:continue
            for block in [10,20,40]:
                for subset in ['full','reduced']:
                    ss=scores if subset=='full' else scores.drop(columns=['EWMAcov-FHS','Portfolio-EWMA-FHS'])
                    mm=mc.reset_index() if block==20 and subset=='full' else mcs(ss,block=block)
                    for rr in mm.to_dict('records'):mc_sensitivity.append(dict(panel=panel,subset=subset,**rr))
            for lhs,rhs in [('Flow-meta-t','Historical'),('Flow-meta-t','VAE-G'),('VAE-G','Historical')]:
                d=scores[lhs]-scores[rhs];a=contrast(d);sd=(bootstrap_means(d.values[:,None],B=9999)[:,0]-d.mean()).std(ddof=1)
                pairs.append(dict(panel=panel,left=lhs,right=rhs,bootstrap_se=float(sd),**a))
            for (name,seed),f in raw.groupby(['model','seed']):cal.append(dict(panel=panel,model=name,seed=int(seed),**calibration(f)))
            diff=scores[['GARCH-FHS','GARCH-EVT']].subtract(scores.Historical,axis=0);noise=bootstrap_means(diff.values,B=9999)-diff.mean().values
            for j,name in enumerate(diff):
                v=inf.loc[name];n=noise[:,j];res=dict(panel=panel,model=name,delta=v.delta,halfwidth=(v.high-v.low)/2)
                for family in [1,16]:
                    lo,hi=np.quantile(n,[.05/(2*family),1-.05/(2*family)]);left,right=0.,10*n.std()
                    for _ in range(60):
                        mid=(left+right)/2
                        if np.mean((n-mid<lo)|(n-mid>hi))>=.8:right=mid
                        else:left=mid
                    res[f'mde80_{family}']=right
                power.append(res)
            # Explicit common-date influence diagnostics; ex-post, not alternative primary samples.
            losses=raw.groupby('date').loss.mean().sort_index()
            for desc,keep in [('No 2020',scores.index.year!=2020)]+[(f'Remove {k} largest losses',~scores.index.isin(losses.nlargest(k).index)) for k in [5,10,20]]:
                z=inference(scores.loc[keep],'Historical');z['panel']=panel;z['sample']=desc;robust+=z.to_dict('records')
            entire=pd.read_csv(O/f'{panel}_forecasts.csv.gz',parse_dates=['date'])
            for pn in sorted(set(entire.portfolio)-{'equal'}):
                f=entire[(entire.portfolio==pn)&(entire.alpha==.01)];s=f.groupby(['date','model']).fz0.mean().unstack().sort_index()
                # Alternative-rule archives contain only equal portfolios; primary records must have these exposures.
                s=s.dropna(axis=1);z=inference(s,'Historical');z['panel']=panel;z['portfolio']=pn;z['fz0']=z.model.map(s.mean());concentrated+=z.to_dict('records')
    pd.DataFrame(rows).to_csv(O/'primary_summary.csv',index=False);pd.DataFrame(power).to_csv(O/'power.csv',index=False)
    pd.DataFrame(cal).to_csv(O/'calibration.csv',index=False);pd.DataFrame(robust).to_csv(O/'robustness.csv',index=False);pd.DataFrame(concentrated).to_csv(O/'concentrated.csv',index=False)
    pd.DataFrame(mc_sensitivity).to_csv(O/'mcs_sensitivity.csv',index=False);pd.DataFrame(pairs).to_csv(O/'pairwise_precision.csv',index=False)
    (O/'power.tex').write_text(tex_table('Conditional resolution diagnostic at one percent. MDE is the location shift giving 80\\% power under retained bootstrap noise, with unadjusted 5\\% or conservative sixteen-contrast Bonferroni thresholds. It does not simulate refitting or provide prospective power.','tab:r4-power',['Panel','Contrast','CI half-width','MDE 80\\%','MDE/16'],[[r['panel'].replace('crsp',''),r['model'],fmt(r['halfwidth']),fmt(r['mde80_1']),fmt(r['mde80_16'])] for r in power],'llrrr'))

def synthetic():
    allrows=[];decomp=[]
    for study in ['t','metat','heston']:
        modelnames=['Historical','Gaussian','Gaussian-LW','Student-MLE','Meta-t']+names(study,'original')+(['LU-Flow-G','LU-Flow-t','Heston-Wishart'] if study=='heston' else [])
        rows=[]
        for name in modelnames:
            errors={a:[] for a in [.01,.001]};sampling=[];fit=[];nlls=[];mc=[]
            for unit in range(10):
                if name.startswith('LU-'):ev=evaluation(study,unit,name[3:],family='lu')
                elif name.startswith(('Flow','VAE','WGAN')) or name in ['Student-MLE','Gaussian-LW']:ev=evaluation(study,unit,name)
                elif name=='Heston-Wishart':
                    wp=ROOT/'wishart_completion/primary'
                    p=(wp/f'heston_{unit}.json') if (wp/'manifest.json').exists() else ROOT/'wishart_extension/results'/f'heston_{unit}_primary.json'
                    ev=read(p)
                else:
                    base='referee_round2' if study=='heston' and name=='Meta-t' else 'expansion';ev=read(ROOT/base/'results'/f'{study}_{unit}_0_{name}.json')
                for a in [.01,.001]:
                    v=next(v for v in ev['metrics'] if v['alpha']==a and v.get('portfolio','equal')=='equal');errors[a].append(v['sample_error'] if name=='Historical' else v['total_error'])
                    if a==.01:sampling.append(v['sample_error']);fit.append(0. if name=='Historical' else v['fit_error']);mc.append(0. if name=='Historical' else v.get('scenario_mc_se',0.))
            e=np.array(errors[.01]);s=np.array(sampling);f=np.array(fit);d=e*e-s*s
            r=dict(study=study,model=name,rmse=float(np.sqrt(np.mean(e*e))),rmse_001=float(np.sqrt(np.mean(np.square(errors[.001])))),delta_mse=float(d.mean()),se=float(d.std(ddof=1)/np.sqrt(10)),skill=float(1-np.sqrt(np.mean(e*e)/np.mean(s*s))),scenario_mc_se=float(np.mean(mc)))
            allrows.append(r);decomp.append(dict(study=study,model=name,total_mse=float(np.mean(e*e)),fit_mse=float(np.mean(f*f)),sampling_mse=float(np.mean(s*s)),cross=float(2*np.mean(f*s))))
            rows.append([('Empirical ES' if name=='Historical' else lab(name)),fmt(r['rmse']),fmt(r['rmse_001']),fmt(r['delta_mse']),fmt(r['se']),fmt(r['skill'])])
        title={'t':'Bivariate Student target','metat':'Heterogeneous meta-$t$ target','heston':'Correlated scalar-Heston weekly-window target'}[study]
        (O/f'{study}_risk.tex').write_text(tex_table(title+'. Ten paired estimation datasets. Neural estimates use the repaired one-percent checkpoint rule; added Student and shrinkage-Gaussian risks are analytic. '+('Wishart uses 131,072 stationary conditional-Gaussian paths at 32 steps per day; its tail integrals are evaluated analytically. Other generators use one million evaluation scenarios. ' if study=='heston' and (ROOT/'wishart_completion/primary/manifest.json').exists() else 'Other generators use one million evaluation scenarios. ')+'SEs are dataset dispersion estimates, not automatic normal-theory tests under heavy tails.','tab:r4-syn-'+study,['Model','RMSE 1\\%','RMSE 0.1\\%','$\\Delta$ MSE','SE','Skill'],rows))
    pd.DataFrame(allrows).to_csv(O/'synthetic_summary.csv',index=False);pd.DataFrame(decomp).to_csv(O/'decomposition.csv',index=False)

def density():
    rows=[];syn=[]
    models=['Gaussian','Gaussian-LW','Student-MLE','Meta-t','Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H']
    for panel in ['crsp5','crsp100']:
        columns={}
        for name in models:
            parts=[]
            for year in range(2016,2026):
                ev=evaluation(panel,year,name,rule='density')
                parts.append(np.load(ROOT/ev['pointwise'])['nll'])
            columns[name]=np.concatenate(parts)
        x=pd.DataFrame(columns);assert len(x)==2514;inf=inference(x,'Gaussian').set_index('model');table=[]
        for name in x.mean().sort_values().index:
            a=inf.loc[name];r=dict(panel=panel,model=name,nll=float(x[name].mean()),**{k:float(a[k]) for k in ['delta','low','high','p','p_holm']});rows.append(r)
            table.append([lab(name),fmt(r['nll'],3),fmt(a.delta),interval(a),pformat(a.p_holm)])
        x.to_csv(O/f'{panel}_pointwise_nll.csv.gz',index=False)
        (O/f'{panel}_density.tex').write_text(tex_table(f'{panel.replace("crsp","")}-coordinate joint NLL, matched validation-likelihood selection and seed zero. Paired differences from the full-covariance Gaussian use common dates, 20-day blocks and eight-contrast Holm adjustment. VAE NLL uses IWAE-64 and is upward-biased in expectation; its realized estimate is not an exact bound. Inference conditions on fitted states.','tab:r4-density-'+panel,['Model','NLL','$\\Delta$ Gaussian','95\\% interval','Holm $p$'],table))
    for study in ['t','metat','heston']:
        mods=['Gaussian','Gaussian-LW','Student-MLE','Meta-t']+[n for n in names(study,'original') if not n.startswith('WGAN')]+(['LU-Flow-G','LU-Flow-t'] if study=='heston' else [])
        arrays={}
        for name in mods:
            parts=[]
            for unit in range(10):
                ev=evaluation(study,unit,name[3:] if name.startswith('LU-') else name,rule='density',family='lu' if name.startswith('LU-') else 'original')
                parts.append(np.load(ROOT/ev['pointwise'])['nll'])
            arrays[name]=np.stack(parts)
        table=[]
        for name in sorted(mods,key=lambda s:arrays[s].mean()):
            d=arrays[name]-arrays['Gaussian'];dataset_se=d.mean(1).std(ddof=1)/np.sqrt(10);test_se=d.mean(0).std(ddof=1)/np.sqrt(d.shape[1])
            r=dict(study=study,model=name,nll=float(arrays[name].mean()),delta_gaussian=float(d.mean()),dataset_se=float(dataset_se),shared_test_se=float(test_se));syn.append(r)
            table.append([lab(name),fmt(r['nll']),fmt(r['delta_gaussian']),fmt(dataset_se,4),fmt(test_se,4)])
        if study in ['t','metat']:
            oracle=np.load(OUT/'pointwise'/f'oracle_{study}.npz')['nll']
            table.insert(0,['True density',fmt(oracle.mean()),'--','--',fmt(oracle.std(ddof=1)/np.sqrt(len(oracle)),4)])
        caption=f'{study} synthetic matched likelihood-selected NLL. Paired differences use the same ten fits and shared 20,000 test observations. Dataset SE conditions on the test sample; shared-test SE conditions on the fitted models. They are separate empirical dispersion estimates and are not added as independent errors. A Student-$t_3$ target can give infinite variance to quadratic log-density contrasts, so these columns are not calibrated Gaussian confidence intervals.'
        if study in ['t','metat']:caption+=' The true-density row reports its own same-sample NLL and test SE, rather than a Gaussian contrast.'
        (O/f'{study}_density.tex').write_text(tex_table(caption,'tab:r4-density-'+study,['Model','NLL','$\\Delta$ Gaussian','Dataset SE','Test SE'],table))
    pd.DataFrame(rows).to_csv(O/'density_crsp.csv',index=False);pd.DataFrame(syn).to_csv(O/'density_synthetic.csv',index=False)

def joint():
    rows=[]
    for panel in ['crsp5','crsp100']:
        modelnames=['Historical','Gaussian','Gaussian-LW','Student-MLE','Meta-t']+names(panel,'original');scores={};prob={};observed=None
        for name in modelnames:
            pp=[];bb=[];yy=[]
            for year in range(2016,2026):
                ev=evaluation(panel,year,name);j=ev['joint_event'];pp.extend([j['probability']]*len(j['event']));bb.extend(j['brier']);yy.extend(j['event'])
            scores[name]=bb;prob[name]=np.mean(pp)
            if observed is None:observed=np.array(yy)
            else:assert np.array_equal(observed,yy)
        x=pd.DataFrame(scores);inf=inference(x,'Historical').set_index('model');table=[]
        for name in x.mean().sort_values().index:
            a=inf.loc[name];r=dict(panel=panel,model=name,brier=x[name].mean(),predicted_probability=prob[name],observed_frequency=observed.mean(),**{k:float(a[k]) for k in ['delta','low','high','p','p_holm']});rows.append(r)
            table.append([lab(name),fmt(r['brier'],4),fmt(prob[name],3),fmt(a.delta,4),f'[{a.low:.4f}, {a.high:.4f}]',pformat(a.p_holm)])
        count=2 if panel=='crsp5' else 5
        (O/f'{panel}_joint.tex').write_text(tex_table(f'Joint lower-tail event: at least {count} coordinates fall below their respective training fifth-percentile innovation thresholds. Observed test frequency is {observed.mean():.3f}. Each model forecasts the same event; FHS is exact empirical probability, others use 100,000 scenarios. Brier differences use eleven-contrast within-panel Holm correction and 20-day date blocks. This new retrospective diagnostic is separate from portfolio-risk inference.','tab:r4-joint-'+panel,['Model','Brier','Pred. prob.','$\\Delta$ FHS','95\\% interval','Holm $p$'],table))
    pd.DataFrame(rows).to_csv(O/'joint_summary.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['collect','primary','synthetic','density','joint','all'],default='all');a=p.parse_args()
    for name in ['collect','primary','synthetic','density','joint']:
        if a.part in ['all',name]:globals()[name]();print('ANALYZED',name,flush=True)

if __name__=='__main__':main()

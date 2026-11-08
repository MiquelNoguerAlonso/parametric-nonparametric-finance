"""Reviewer-driven reanalysis from saved forecasts and simulation records.

Never changes archived fitted forecasts. Intervals and equal-tailed p-values
invert the same centered circular bootstrap construction. MCS uses TR and eR.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import xlogy
from analyze import bootstrap_means,holm,tex_table,fmt
from experiment import sample_law
from models import MetaT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';OUT=RES/'revision';OUT.mkdir(exist_ok=True)
LABEL={'Historical':'FHS','VAE-t':'VAE-tlatent'}
def label(name):return LABEL.get(name,name)
def pformat(x):return '$<0.001$' if x<.001 else fmt(x,3)

def inference(scores,benchmark,block=20,B=9999):
    d=scores.subtract(scores[benchmark],axis=0);m=d.mean().values
    boots=bootstrap_means(d.values,block=block,B=B);centered=boots-m
    # Invert the centered distribution: basic equal-tailed interval.
    lo,hi=m-np.quantile(centered,.975,axis=0),m-np.quantile(centered,.025,axis=0)
    p=np.minimum(1,2*np.minimum((1+(centered>=m).sum(0))/(B+1),(1+(centered<=m).sum(0))/(B+1)))
    non=np.array(scores.columns!=benchmark);adj=np.ones(len(p));adj[non]=holm(p[non])
    return pd.DataFrame(dict(model=scores.columns,benchmark=benchmark,delta=m,low=lo,high=hi,p=p,p_holm=adj))

def mcs(scores,block=20,B=9999):
    x=scores.values;means=x.mean(0);boot=bootstrap_means(x,block=block,B=B)
    noise=boot-means;d=means[:,None]-means[None,:]
    # Var of pairwise bootstrap mean differences, preserving pairing.
    variance=np.var(noise[:,:,None]-noise[:,None,:],axis=0,ddof=1)
    se=np.sqrt(variance);np.fill_diagonal(se,np.inf)
    t=d/se;tb=(noise[:,:,None]-noise[:,None,:])/se
    active=list(range(x.shape[1]));rows=[];cumulative=0.
    while len(active)>1:
        observed=np.max(np.abs(t[np.ix_(active,active)]))
        null=np.max(np.abs(tb[:,active][:,:,active]),axis=(1,2))
        p=(1+(null>=observed).sum())/(B+1);cumulative=max(cumulative,p)
        worst=active[int(np.argmax(np.max(t[np.ix_(active,active)],axis=1)))]
        rows.append(dict(model=scores.columns[worst],step=len(rows)+1,p_step=p,p_mcs=cumulative,block=block))
        active.remove(worst)
    rows.append(dict(model=scores.columns[active[0]],step=len(rows)+1,p_step=1.,p_mcs=1.,block=block))
    return pd.DataFrame(rows)

def calibration(g):
    g=g.sort_values('date');a=float(g.alpha.iloc[0]);h=g.hit.values.astype(int);n=len(h);k=int(h.sum());ph=k/n
    lr=2*(xlogy(k,ph/a)+xlogy(n-k,(1-ph)/(1-a)))
    v=np.column_stack([h-a,g.es_residual.values/g.es.values])
    # Predictable instruments: constant and lagged exceedance deviation.
    lag=np.r_[0.,h[:-1]-a];z=np.column_stack([v,v*lag[:,None]])[1:]
    mean=z.mean(0);zz=z-mean;N=len(z);cov=zz.T@zz/N
    for j in range(1,21):
        gamma=zz[j:].T@zz[:-j]/N;cov+=(1-j/21)*(gamma+gamma.T)
    rank=np.linalg.matrix_rank(cov);wald=float(N*mean@np.linalg.pinv(cov)@mean)
    return dict(n=n,hits=k,kupiec_lr=float(lr),kupiec_p=float(st.chi2.sf(lr,1)),cc_wald=wald,cc_df=rank,cc_p=float(st.chi2.sf(wald,rank)))

def main():
    raw=pd.read_csv(RES/'crsp_all_forecasts.csv.gz',parse_dates=['date'])
    dyn=pd.read_csv(OUT/'dynamic_forecasts.csv.gz',parse_dates=['date']) if (OUT/'dynamic_forecasts.csv.gz').exists() else None
    allraw=raw if dyn is None else pd.concat([raw,dyn],ignore_index=True)
    daily=allraw.groupby(['date','model','portfolio','alpha'],as_index=False).agg(fz0=('fz0','mean'),hit=('hit','mean'),es=('es','mean'),es_residual=('es_residual','mean'),loss=('loss','mean'))
    old=pd.read_csv(RES/'crsp_summary.csv');nll=old[old.portfolio.eq('equal')&old.alpha.eq(.01)].set_index('model').nll.to_dict()
    inference_rows=[];mcs_rows=[];tables=[];summary=[]
    for alpha in [.01,.025]:
        g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(alpha)]
        scores=g.pivot(index='date',columns='model',values='fz0').sort_index()
        for benchmark in ['Historical','Gaussian']:
            r=inference(scores,benchmark);r['alpha']=alpha;inference_rows.append(r)
        for block in ([10,20,40] if alpha==.01 else [20]):
            r=mcs(scores,block);r['alpha']=alpha;mcs_rows.append(r)
        inf=inference_rows[-2].set_index('model');mc=mcs_rows[-2 if alpha==.01 else -1].set_index('model')
        rows=[]
        for name in scores.mean().sort_values().index:
            a=g[g.model==name];r=inf.loc[name]
            row=dict(alpha=alpha,model=name,fz0=float(a.fz0.mean()),hits_percent=100*float(a.hit.mean()),es_moment=float(a.es_residual.mean()/a.es.mean()),nll=nll.get(name,np.nan),delta_fhs=r.delta,low=r.low,high=r.high,p=r.p,p_holm=r.p_holm,mcs_p=mc.loc[name,'p_mcs'])
            summary.append(row)
            rows.append([label(name),fmt(row['fz0']),fmt(row['hits_percent'],2),fmt(row['es_moment']),fmt(r.delta),'['+fmt(r.low)+', '+fmt(r.high)+']',fmt(row['mcs_p'])])
        tables.append(tex_table(f'Equal-weight CRSP losses, {100*alpha:g}\\% tail probability, 2,514 test dates. FHS is the reference; negative score differences favor the row model. Intervals invert the centered paired circular bootstrap (9,999 draws, 20-day blocks). MCS columns give monotone elimination $p$-values for the range statistic; values at least 0.05 enter the 95\\% set. The set includes the three added dynamic benchmarks. Hit percentages average seed-specific exceedance rates; calibration tests are performed separately per seed. All reanalysis and new benchmark choices are retrospective.',f'tab:revised-{alpha}',['Model','FZ0','Hits \\%','ES moment','$\\Delta$ FHS','95\\% interval','MCS $p$'],rows))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'primary_summary.csv',index=False)
    pd.concat(inference_rows).to_csv(OUT/'inference.csv',index=False)
    mc=pd.concat(mcs_rows);mc.to_csv(OUT/'mcs.csv',index=False)
    (OUT/'primary_tables.tex').write_text('\n'.join(tables))
    # Calibration must use integer hit counts from each forecast sequence.
    cal=[]
    for (model,seed,alpha),g in allraw[allraw.portfolio.eq('equal')].groupby(['model','seed','alpha']):
        cal.append(dict(model=model,seed=seed,alpha=alpha,**calibration(g)))
    cal=pd.DataFrame(cal);cal.to_csv(OUT/'calibration.csv',index=False)
    rows=[]
    for name,g in cal[cal.alpha.eq(.01)].groupby('model'):
        pair=lambda key:' / '.join(pformat(x) for x in g.sort_values('seed')[key])
        rows.append([label(name),' / '.join(map(str,g.sort_values('seed').hits)),pair('kupiec_p'),pair('cc_p')])
    table=tex_table('Calibration at 1\\% for the equal-weight portfolio. Slash-separated entries are seeds 0 and 1; deterministic new benchmarks have one entry. UC is the Kupiec likelihood-ratio test. CC is a four-moment joint VaR--ES Wald diagnostic using a constant and the lagged hit deviation, with Bartlett HAC bandwidth 20. These unadjusted diagnostics condition on saved forecasts and do not account for refitting uncertainty; non-rejection is not proof of full conditional calibration.','tab:calibration',['Model','Hit counts','UC $p$','CC $p$'],rows)
    (OUT/'calibration_table.tex').write_text(table)
    density=old[old.portfolio.eq('equal')&old.alpha.eq(.01)].sort_values('nll')
    rows=[[label(r.model),'IWAE-64 estimate' if r.model.startswith('VAE') else 'Exact density',fmt(r.nll,4)] for r in density.itertuples() if np.isfinite(r.nll)]
    (OUT/'density_table.tex').write_text(tex_table('CRSP joint negative log-likelihood, averaged over the original annual forecasts and two seeds. Lower is better. VAE entries estimate an importance-weighted bound and have an upward NLL bias in expectation; their finite Monte Carlo realizations are not exact likelihoods. Portfolio-specific GARCH benchmarks do not specify a comparable joint density, and GANs and FHS have no reported continuous density.','tab:density',['Model','Likelihood evaluation','NLL'],rows,'llr'))
    # Crash removal uses the same ex-post dates for every model. Descriptive only.
    g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(.01)]
    scores=g.pivot(index='date',columns='model',values='fz0').sort_index();loss=g.groupby('date').loss.first()
    masks={'All':np.ones(len(scores),bool),'No 2020':scores.index.year!=2020}
    for k in [5,10,20]:masks[f'Drop {k} losses']=~scores.index.isin(loss.nlargest(k).index)
    rows=[];rob=[]
    for name in scores.columns:
        row=[label(name)]
        for desc,mask in masks.items():
            delta=float((scores[name]-scores.Historical)[mask].mean());row.append(fmt(delta))
            rob.append(dict(model=name,subset=desc,n=int(np.sum(mask)),delta_fhs=delta,mean_score=float(scores[name][mask].mean())))
        rows.append(row)
    pd.DataFrame(rob).to_csv(OUT/'crash_robustness.csv',index=False)
    (OUT/'robustness_table.tex').write_text(tex_table('Mean FZ0 difference from FHS at 1\\%. The same dates are removed for all models. Top-$k$ days are ranked by realized equal-portfolio losses, selected ex post. This is a crash-influence diagnostic, not a backtest of a strategy that could omit those days.','tab:crash',['Model',*masks],rows))
    # Original construction audit on exactly its original 1,999 replicates.
    primary=raw[raw.portfolio.eq('equal')&raw.alpha.eq(.01)].groupby(['date','model']).fz0.mean().unstack()
    d=primary.subtract(primary.Gaussian,axis=0);m=d.mean().values;b=bootstrap_means(d.values)
    lo,hi=np.quantile(b,[.025,.975],axis=0);p=(1+(np.abs(b-m)>=np.abs(m)).sum(0))/(len(b)+1)
    pct=np.minimum(1,2*np.minimum((1+(b<=0).sum(0))/2000,(1+(b>=0).sum(0))/2000))
    audit=pd.DataFrame(dict(model=d.columns,delta=m,percentile_low=lo,percentile_high=hi,centered_absolute_p=p,percentile_tail_p=pct))
    audit.to_csv(OUT/'legacy_bootstrap_audit.csv',index=False)
    # Dense parameter export: one entry for every fold, seed, model.
    pars=[]
    for p in sorted((ROOT/'checkpoints').glob('crsp_*.meta.json')):
        a=json.loads(p.read_text());parts=p.name.split('_');pars.append(dict(year=int(parts[1]),seed=int(parts[2]),model=parts[3].split('.')[0],**a))
    (OUT/'meta_parameters_full.json').write_text(json.dumps(pars,indent=2))
    rows=[]
    for a in pars:
        if a['seed'] or a['model']!='Meta-t':continue
        fs=[json.loads((RES/f"crsp_{a['year']}_{seed}_Flow-t.json").read_text())['info']['base_df'] for seed in [0,1]]
        rows.append([a['year'],*[fmt(v[0],2) for v in a['marg']],fmt(a['nu'],2),f'{min(fs):.2f}--{max(fs):.2f}'])
    (OUT/'parameters_table.tex').write_text(tex_table('Archived CRSP degrees of freedom by annual fold: five marginal meta-$t$ parameters, copula parameter, and fitted Flow-t base range across seeds. Plain meta-$t$ estimation is deterministic across seeds. Locations, scales, and all correlation entries are in the complete JSON export. WGAN-t uses fixed $\\nu=3$ in every fold.','tab:parameters',['Year','$\\nu_1$','$\\nu_2$','$\\nu_3$','$\\nu_4$','$\\nu_5$','$\\nu_c$','Flow-t $\\nu$'],rows))
    # Oracle NLL on the identical test sample, not population entropy.
    oracle=[]
    for law in ['t','metat']:
        test=sample_law(law,20000,999)
        if law=='t':
            lp=st.multivariate_t.logpdf(test,shape=[[1,.5],[.5,1]],df=3)
            entropy=float(st.multivariate_t(shape=[[1,.5],[.5,1]],df=3).entropy())
        else:
            mt=MetaT();mt.marg=np.array([[3,0,1],[8,0,1.]]);mt.P=np.array([[1,.5],[.5,1]]);mt.nu=4;mt.d=2;lp=mt.logpdf(test);entropy=None
        oracle.append(dict(study=law,oracle_nll=float(-lp.mean()),iid_test_se=float(lp.std(ddof=1)/np.sqrt(len(lp))),population_entropy=entropy))
    pd.DataFrame(oracle).to_csv(OUT/'oracle_nll.csv',index=False)
    (OUT/'oracle_table.tex').write_text(tex_table('True density evaluated on the same 20,000 test observations as every fitted model. The standard error concerns only independent test-sample variation. A fitted density can beat the oracle on this finite sample without beating population entropy. No exact 25-dimensional simulated Heston density is available; no oracle value is invented.','tab:oracle',['Law','Oracle NLL','Test SE','Population entropy'],[[a['study'],fmt(a['oracle_nll'],5),fmt(a['iid_test_se'],5),fmt(a['population_entropy'],5)] for a in oracle]))
    # Dataset, not optimization seed, is the independent simulation unit.
    decom=[]
    for law in ['t','metat','heston']:
        records=[]
        for p in RES.glob(law+'_*.json'):
            r=json.loads(p.read_text())
            if r.get('study')==law:records.append(r)
        for name in sorted({r['model'] for r in records}):
            a=[r for r in records if r['model']==name];ds=[];fit=[];sampling=[];total=[]
            for r in a:
                z=next(m for m in r['metrics'] if m['portfolio']=='equal' and m['alpha']==.01)
                fit.append(z['fit_error']);sampling.append(z['sample_error']);total.append(z['total_error'])
                ds.append(dict(dataset=r['dataset'],paired=z['total_error']**2-z['sample_error']**2))
            paired=pd.DataFrame(ds).groupby('dataset').paired.mean();f=np.array(fit);s=np.array(sampling);t=np.array(total)
            decom.append(dict(study=law,model=name,n_datasets=len(paired),fit_mse=float(np.mean(f*f)),sample_mse=float(np.mean(s*s)),cross=float(2*np.mean(f*s)),total_mse=float(np.mean(t*t)),paired_mse=float(paired.mean()),paired_se=float(paired.std(ddof=1)/np.sqrt(len(paired))),rmse_skill=float(1-np.sqrt(np.mean(t*t)/np.mean(s*s)))))
    dec=pd.DataFrame(decom);dec.to_csv(OUT/'synthetic_decomposition.csv',index=False)
    rows=[[a['study'],'Empirical' if a['model']=='Historical' else label(a['model']),fmt(a['fit_mse']),fmt(a['sample_mse']),fmt(a['cross']),fmt(a['paired_mse']),fmt(a['paired_se']),fmt(a['rmse_skill'])] for a in decom]
    (OUT/'decomposition_table.tex').write_text(tex_table('One-percent ES error decomposition. Total MSE equals fit MSE plus sampling MSE plus twice the cross moment. Paired differences subtract the exact empirical-training ES squared error, averaged over seeds within each of three independent datasets; SE is computed across those three paired averages. Skill is one minus the model RMSE divided by the empirical-training RMSE. The empirical resampling baseline adds simulation error, so its RMSE is only approximately the pure sampling component. With three datasets these SEs are descriptive and unstable.','tab:decomposition',['Law','Model','Fit MSE','Samp. MSE','Cross','$\\Delta$ MSE','SE','Skill'],rows,'llrrrrrr'))
    # Concentrated portfolios now contain FHS and every added benchmark.
    rows=[]
    for pn in ['small','large']:
        g=daily[daily.portfolio.eq(pn)&daily.alpha.eq(.01)].groupby('model').agg(score=('fz0','mean'),hits=('hit','mean'))
        for name,r in g.iterrows():rows.append([pn,label(name),fmt(r.score),fmt(100*r.hits,2),fmt(r.score-g.loc['Historical','score'])])
    (OUT/'concentrated_table.tex').write_text(tex_table('Concentrated-portfolio descriptive comparisons at 1\\%, including the FHS reference.','tab:concentrated-new',['Portfolio','Model','FZ0','Hits \\%','$\\Delta$ FHS'],rows,'llrrr'))
    annual=daily[daily.portfolio.eq('equal')&daily.alpha.eq(.01)].assign(year=lambda g:g.date.dt.year).groupby(['year','model']).fz0.mean().unstack()
    fig,axs=plt.subplots(2,1,figsize=(8.5,6.5),sharex=True)
    for i,name in enumerate(['Historical','Meta-t','Flow-meta-t','VAE-G','VAE-H','GARCH-t','GARCH-EVT','EWMAcov-FHS']):
        if name not in annual:continue
        style=dict(marker=['o','s','^','D','v','P','X','*'][i],ms=4,lw=1.2,ls=['-','--',':','-.'][i%4],label=label(name))
        axs[0].plot(annual.index,annual[name],**style)
        axs[1].plot(annual.index,annual[name]-annual.Historical,**style)
    axs[0].set_ylabel('Mean FZ0');axs[1].set_ylabel('FZ0 minus FHS');axs[1].set_xlabel('Out-of-sample year');axs[1].axhline(0,color='black',lw=.7)
    for ax in axs:ax.grid(alpha=.2)
    axs[0].legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/crsp_annual_scores_revised.pdf');plt.close(fig)
    print(summary.to_string(index=False));print('Oracle',oracle)

if __name__=='__main__':main()

"""Integrate the new benchmark without changing archived model forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze import tex_table,fmt
from revision_analysis import inference,mcs,calibration,label,pformat

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'wishart_extension';O=W/'analysis';OLD=ROOT/'referee_round2/analysis'
NAME='Heston-Wishart'


def read_result(study,unit,kind='primary'):
    return json.loads((W/'results'/f'{study}_{unit}_{kind}.json').read_text())


def synthetic():
    records=[read_result('heston',u) for u in range(10)]
    raw=pd.DataFrame([dict(study='heston',dataset=r['dataset'],model=NAME,nll=np.nan,**m) for r in records for m in r['metrics']])
    previous=pd.read_csv(OLD/'synthetic_raw.csv');data=pd.concat([previous,raw],ignore_index=True)
    assert not data.duplicated(['study','dataset','model','alpha']).any()
    data.to_csv(O/'synthetic_raw.csv',index=False)
    summary=[]
    for a,g in raw.groupby('alpha'):
        delta=g.total_error**2-g.sample_error**2;emp=float(np.sqrt(np.mean(g.sample_error**2)));rmse=float(np.sqrt(np.mean(g.total_error**2)))
        summary.append(dict(study='heston',model=NAME,alpha=a,nll=np.nan,rmse=rmse,empirical_rmse=emp,
            delta_mse=float(delta.mean()),paired_se=float(delta.std(ddof=1)/np.sqrt(10)),rmse_skill=1-rmse/emp,
            bias=float(g.total_error.mean()),fit_mse=float(np.mean(g.fit_error**2)),sampling_mse=emp**2,
            cross=float(2*np.mean(g.fit_error*g.sample_error)),total_mse=rmse**2))
    summ=pd.concat([pd.read_csv(OLD/'synthetic_summary.csv'),pd.DataFrame(summary)],ignore_index=True)
    summ.to_csv(O/'synthetic_summary.csv',index=False)
    h=summ[summ.study.eq('heston')];base=h.groupby('alpha').empirical_rmse.first()
    rows=[['Empirical (exact)',fmt(base.loc[.01]),fmt(base.loc[.001]),'0.000','0.000','0.000']];decomp=[]
    for name,z in h.groupby('model'):
        z=z.set_index('alpha');a=z.loc[.01];nm='Empirical (MC)' if name=='Historical' else label(name)
        rows.append([nm,fmt(a.rmse),fmt(z.loc[.001,'rmse']),fmt(a.delta_mse),fmt(a.paired_se),fmt(a.rmse_skill)])
        decomp.append([nm,fmt(a.fit_mse),fmt(a.sampling_mse),fmt(a.cross),fmt(a.delta_mse),fmt(a.paired_se)])
    (O/'heston_risk_table.tex').write_text(tex_table('Correlated scalar-Heston target, ten common estimation datasets and one million scenarios per fitted model. Heston--Wishart uses training-only simulated moments; neural checkpoints use primary one-percent validation FZ0. Paired MSE differences and SE use exact empirical-training ES as reference. The Wishart family is misspecified for this target.','tab:expanded-heston',['Model','RMSE 1\\%','RMSE 0.1\\%','$\\Delta$ MSE','SE','Skill'],rows))
    (O/'heston_decomposition.tex').write_text(tex_table('Heston windows: one-percent ES error decomposition, now including Heston--Wishart. Total MSE equals fit MSE plus sampling MSE plus the cross term.','tab:decomp-heston',['Model','Fit MSE','Samp. MSE','Cross','$\\Delta$ MSE','SE'],decomp))
    a=h[(h.model==NAME)&(h.alpha==.01)].iloc[0]
    text=(f'The added Heston--Wishart fit has one-percent RMSE {a.rmse:.3f}, against {a.empirical_rmse:.3f} for exact empirical ES. Its paired MSE difference is {a.delta_mse:.3f}, with SE {a.paired_se:.3f}. '
          'This is a full stochastic-covariance model under restricted coefficients, fitted by simulated moments; it remains misspecified for the correlated scalar-Heston law. Its addition does not supply a correctly specified Heston oracle. ')
    text+=('The primary empirical estimator still has the lowest observed RMSE.\n' if a.rmse>h[h.model.ne(NAME)].rmse.min() else 'The comparison concerns this fixed calibration design.\n')
    (O/'heston_findings.tex').write_text(text)
    dependence=pd.DataFrame([dict(dataset=r['dataset'],model=NAME,**r['dependence']) for r in records]);dependence.to_csv(O/'heston_dependence.csv',index=False)
    oldtable=(OLD/'heston_dependence_table.tex').read_text();dep=dependence[['kurtosis','cluster','leverage','cross_vol']].mean()
    dep_row=NAME+' & '+' & '.join(fmt(dep[k]) for k in ['kurtosis','cluster','leverage','cross_vol'])+r' \\'+'\n'
    oldtable=oldtable.replace(r'\bottomrule',dep_row+r'\bottomrule')
    oldtable=oldtable.replace('100,000 generated windows per fit.', '100,000 generated windows per retained fit; the Wishart row averages ten such batches per fit.')
    (O/'heston_dependence_table.tex').write_text(oldtable)
    mc=pd.DataFrame([dict(dataset=r['dataset'],**m) for r in records for m in r['metrics']]);mc.to_csv(O/'heston_mc.csv',index=False)
    return summ


def crsp():
    previous=pd.read_csv(OLD/'crsp5_forecasts.csv.gz',parse_dates=['date'])
    extra=pd.concat([pd.read_csv(W/'results'/f'crsp5_{y}_primary.csv.gz',parse_dates=['date']) for y in range(2016,2026)],ignore_index=True)
    assert len(extra)==2514*3*2
    data=pd.concat([previous,extra],ignore_index=True)
    assert not data.duplicated(['date','model','portfolio','alpha','seed']).any()
    data.to_csv(O/'crsp5_forecasts.csv.gz',index=False)
    # Exact same realized losses and dates as the existing forecasts.
    anchor=previous[previous.model.eq('Historical')].groupby(['date','portfolio','alpha']).loss.first()
    observed=extra.set_index(['date','portfolio','alpha']).loss
    assert np.max(abs(anchor.sort_index()-observed.sort_index()))<1e-10
    daily=data.groupby(['date','model','portfolio','alpha'],as_index=False).agg(fz0=('fz0','mean'),hit=('hit','mean'),es=('es','mean'),es_residual=('es_residual','mean'),loss=('loss','mean'))
    summaries=[];infs=[];mcss=[]
    for alpha in [.01,.025]:
        g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(alpha)]
        scores=g.pivot(index='date',columns='model',values='fz0').sort_index();assert scores.shape==(2514,16) and scores.notna().all().all()
        inf=inference(scores,'Historical');inf['alpha']=alpha;infs.append(inf);lookup=inf.set_index('model')
        for block in ([10,20,40] if alpha==.01 else [20]):
            mc=mcs(scores,block);mc['alpha']=alpha;mcss.append(mc)
        mc=next(v for v in mcss if v.alpha.iloc[0]==alpha and v.block.iloc[0]==20).set_index('model');rows=[]
        for name in scores.mean().sort_values().index:
            z=g[g.model.eq(name)];b=lookup.loc[name]
            row=dict(model=name,alpha=alpha,fz0=float(z.fz0.mean()),hits_percent=100*float(z.hit.mean()),es_moment=float(z.es_residual.mean()/z.es.mean()),delta_fhs=b.delta,low=b.low,high=b.high,p=b.p,p_holm=b.p_holm,mcs_p=mc.loc[name,'p_mcs'])
            summaries.append(row)
            rows.append([label(name).replace('Portfolio-EWMA-FHS','Port.-EWMA-FHS'),fmt(row['fz0']),fmt(row['hits_percent'],2),fmt(row['es_moment']),fmt(b.delta),'['+fmt(b.low)+', '+fmt(b.high)+']',pformat(row['mcs_p'])])
        filename='crsp5_primary.tex' if alpha==.01 else 'crsp5_secondary.tex'
        (O/filename).write_text(tex_table(f'Five size baskets, equal-portfolio {100*alpha:g}\\% risk on 2,514 common dates. The added Heston--Wishart benchmark expands the MCS to sixteen models and the Holm family to fifteen FHS contrasts. Paired basic intervals and MCS use 9,999 circular draws and 20-day blocks. Hits are exceedance percentages.','tab:crsp5-'+str(alpha),['Model','FZ0','Hits \\%','ES moment','$\\Delta$ FHS','95\\% interval','MCS $p$'],rows))
    summary=pd.DataFrame(summaries);summary.to_csv(O/'crsp5_summary.csv',index=False);pd.concat(infs).to_csv(O/'crsp5_inference.csv',index=False);pd.concat(mcss).to_csv(O/'crsp5_mcs.csv',index=False)
    cal=[]
    for (model,seed,alpha),z in data[data.portfolio.eq('equal')].groupby(['model','seed','alpha']):cal.append(dict(model=model,seed=seed,alpha=alpha,**calibration(z)))
    cal=pd.DataFrame(cal);cal.to_csv(O/'crsp5_calibration.csv',index=False);rows=[]
    for name,z in cal[cal.alpha.eq(.01)].groupby('model'):
        z=z.sort_values('seed');rows.append([label(name),' / '.join(map(str,z.hits)),' / '.join(pformat(v) for v in z.kupiec_p),' / '.join(pformat(v) for v in z.cc_p)])
    (O/'crsp5_calibration.tex').write_text(tex_table('Five-basket one-percent calibration, including Heston--Wishart. UC is the Kupiec test; CC is the four-moment joint VaR--ES diagnostic with Bartlett bandwidth 20. Slash-separated values identify the two retained neural/baseline seeds; Wishart and portfolio-filter controls have one forecast sequence. These tests condition on fitted forecasts.','tab:crsp5-calibration',['Model','Hit counts','UC $p$','CC $p$'],rows))
    pri=summary[summary.alpha.eq(.01)].set_index('model');best=pri.fz0.idxmin();a=pri.loc[best];w=pri.loc[NAME];c=cal[(cal.model==NAME)&(cal.alpha==.01)].iloc[0]
    neural=pri.loc[[x for x in pri.index if x.startswith(('Flow','VAE','WGAN'))]]
    text='\\subsection{Five-basket findings}\n'
    text+=f'The lowest observed one-percent score is {label(best)} at {a.fz0:.4f}, against {pri.loc["Historical","fz0"]:.4f} for FHS. Its difference is {a.delta_fhs:.4f}, with basic interval $[{a.low:.4f},{a.high:.4f}]$ and fifteen-contrast Holm $p={a.p_holm:.4f}$. '
    text+=('No primary neural improvement over FHS survives this correction. ' if not ((neural.delta_fhs<0)&(neural.p_holm<.05)).any() else 'The updated inference is reported in the accompanying table. ')
    text+='The 100-stock comparison and checkpoint sensitivities retain their original model sets.\n\n'
    text+=f'Heston--Wishart has mean FZ0 {w.fz0:.4f}. Its difference from FHS is {w.delta_fhs:.4f}, with basic interval $[{w.low:.4f},{w.high:.4f}]$ and Holm $p={w.p_holm:.4f}$. Its {int(c.hits)} exceedances imply a {w.hits_percent:.2f}\\% hit rate; the UC and CC $p$-values are {c.kupiec_p:.4f} and {c.cc_p:.4f}. '
    text+='This evaluates its restricted moment calibration and covariance-state filter together.\n\n'
    text+=f'The 95\\% MCS at 20-day blocks retains {int((pri.mcs_p>=.05).sum())} of sixteen models; Heston--Wishart has MCS $p={w.mcs_p:.4f}$. Numerical refinement, particle diagnostics, and crash-day sensitivity are reported in Appendix~\\ref{{sec:wishart-numerics}}.\n'
    (O/'crsp5_findings.tex').write_text(text)
    rows=[]
    for pn in ['small','large']:
        g=daily[daily.portfolio.eq(pn)&daily.alpha.eq(.01)].groupby('model').agg(score=('fz0','mean'),hits=('hit','mean'))
        for name,r in g.iterrows():rows.append([pn,label(name),fmt(r.score),fmt(100*r.hits,2),fmt(r.score-g.loc['Historical','score'])])
    (O/'crsp5_concentrated.tex').write_text(tex_table('Five-basket concentrated portfolios at one percent, including the fitted Heston--Wishart benchmark.','tab:crsp5-concentrated',['Portfolio','Model','FZ0','Hits \\%','$\\Delta$ FHS'],rows,'llrrr'))
    g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(.01)];scores=g.pivot(index='date',columns='model',values='fz0').sort_index();loss=g.groupby('date').loss.first()
    masks={'All':np.ones(len(scores),bool),'No 2020':scores.index.year!=2020}
    for k in [5,10,20]:masks[f'Drop {k} losses']=~scores.index.isin(loss.nlargest(k).index)
    rob=[];rows=[]
    for name in scores.columns:
        row=[label(name)]
        for subset,mask in masks.items():
            delta=float((scores[name]-scores.Historical)[mask].mean());row.append(fmt(delta));rob.append(dict(model=name,subset=subset,n=int(mask.sum()),delta_fhs=delta))
        rows.append(row)
    pd.DataFrame(rob).to_csv(O/'crsp5_robustness.csv',index=False)
    (O/'crsp5_robustness.tex').write_text(tex_table('Five-basket one-percent mean FZ0 differences from FHS after identical ex-post date exclusions. The crash-influence diagnostic now includes Heston--Wishart.','tab:crsp5-robustness',['Model',*masks],rows))
    (O/'robustness.tex').write_text('The full crash-influence comparison in Table~\\ref{tab:crsp5-robustness} applies identical exclusions to every model. Those exclusions do not define an implementable trading rule.\n')
    return summary,daily


def diagnostics():
    fits=[];rows=[]
    for study,units in [('heston',range(10)),('crsp5',range(2016,2026))]:
        for unit in units:
            a=json.loads((W/'fits'/f'{study}_{unit}.json').read_text());p=a['parameters'];best=min(a['fit']['candidates'],key=lambda x:x['objective'])
            fits.append(dict(study=study,unit=unit,beta=p['beta'],kappa=p['kappa'],rho=p['rho'],objective=best['objective'],observed_kurtosis=a['fit']['observed_moments'][0],simulated_kurtosis=best['simulated_moments'][0]))
            rows.append(['Heston '+str(unit) if study=='heston' else str(unit),p['beta'],fmt(p['kappa'],3),fmt(p['rho'],1),fmt(a['fit']['observed_moments'][0],2),fmt(best['simulated_moments'][0],2)])
    pd.DataFrame(fits).to_csv(O/'parameters.csv',index=False)
    (O/'parameters.tex').write_text(tex_table('All twenty training-only Wishart fits. The first ten rows identify independent Heston datasets; calendar years identify CRSP folds. $K_{obs}$ and $K_{sim}$ are the pooled observed and fitted-simulation excess-kurtosis moments. Full mean vectors, covariance matrices, candidate losses and random seeds are supplied in JSON. Boundary choices and moment gaps describe the restricted calibration.','tab:wishart-parameters',['Sample','$b$','$\\kappa$/day','$\\rho$','$K_{obs}$','$K_{sim}$'],rows))
    shape='All twenty fits select the minimum shape $b=6$. On the Heston training datasets, observed excess-kurtosis moments range from 2.93 to 3.56, versus 0.88 to 1.10 for the fitted simulations. Thus the restricted calibration leaves a substantial marginal-moment gap; its error should not be attributed solely to stochastic-correlation architecture.\n'
    (O/'heston_findings.tex').write_text((O/'heston_findings.tex').read_text()+shape)
    checks=json.loads((O/'kernel_checks.json').read_text());assert checks['passed']
    refinements=[];rows=[]
    for unit in [0,1,2]:
        primary=read_result('heston',unit);fine=read_result('heston',unit,'refined')
        for m,n in zip(primary['metrics'],fine['metrics']):
            diff=n['es']-m['es'];se=np.hypot(m['scenario_mc_se'],n['scenario_mc_se'])
            refinements.append(dict(study='heston',unit=unit,alpha=m['alpha'],primary=m['es'],refined=n['es'],difference=diff,mc_se_combined=float(se)))
            rows.append(['Heston '+str(unit),f"{m['alpha']*100:g}\\%",fmt(m['es'],4),fmt(n['es'],4),fmt(diff,4),fmt(se,4)])
    (O/'heston_refinement.tex').write_text(tex_table('Wishart return-integration sensitivity with parameters fixed. Primary ES uses four substeps and one million scenarios; refined ES uses eight substeps and 400,000 scenarios. The last column is the root-sum-square of batch Monte Carlo SEs, a scale diagnostic rather than a paired-test SE.','tab:wishart-heston-refinement',['Sample','Tail','Primary ES','Refined ES','Difference','MC scale'],rows))
    numerical=[];rows=[];comparisons=[]
    for year in range(2016,2026):
        a=read_result('crsp5',year);numerical.append(dict(year=year,median_ess=a['median_test_ess'],min_ess=a['min_test_ess'],fraction_ess_under_10=a['fraction_test_ess_under_10'],median_parent_ess=a['median_test_parent_ess'],min_parent_ess=a['min_test_parent_ess']))
    pd.DataFrame(numerical).to_csv(O/'particle_diagnostics.csv',index=False)
    for year in [2016,2020,2025]:
        a=pd.read_csv(W/'results'/f'crsp5_{year}_primary.csv.gz');b=pd.read_csv(W/'results'/f'crsp5_{year}_refined.csv.gz')
        x=a.merge(b,on=['date','portfolio','alpha'],suffixes=('_primary','_refined'),validate='one_to_one')
        assert np.max(abs(x.loss_primary-x.loss_refined))<1e-12
        for alpha in [.01,.025]:
            z=x[(x.portfolio=='equal')&(x.alpha==alpha)];delta=z.fz0_refined-z.fz0_primary
            item=dict(study='crsp5',unit=year,alpha=alpha,primary=float(z.fz0_primary.mean()),refined=float(z.fz0_refined.mean()),difference=float(delta.mean()),max_abs_es_change=float(np.max(abs(z.es_refined-z.es_primary))),es_relative_rms=float(np.sqrt(np.mean(((z.es_refined-z.es_primary)/z.es_primary)**2))))
            comparisons.append(item);rows.append([year,f'{100*alpha:g}\\%',fmt(item['primary'],4),fmt(item['refined'],4),fmt(item['difference'],4),fmt(100*item['es_relative_rms'],2)])
    pd.DataFrame(refinements+comparisons).to_csv(O/'numerical_refinement.csv',index=False)
    (O/'crsp_refinement.tex').write_text(tex_table('Wishart filter sensitivity with parameters fixed: 512 states, 4,096 predictive paths and four substeps versus 1,024 states, 8,192 paths and eight substeps. FZ0 uses the same realized equal-portfolio losses. RMS is the relative daily ES change in percent. This joint refinement does not identify the contribution of each numerical setting.','tab:wishart-crsp-refinement',['Year','Tail','Primary FZ0','Refined FZ0','Difference','ES RMS \\%'],rows))
    miness=min(z['min_ess'] for z in numerical);meds=[z['median_ess'] for z in numerical];low=[z['fraction_ess_under_10'] for z in numerical]
    text=f'Analytic checks pass for the stationary covariance moments, conditional covariance mean, scalar gamma limit, zero-leverage frozen characteristic function, and Gaussian-mixture risk formula. Forecasts are unchanged through the date of a deliberately altered future observation. The maximum stationary covariance mean error in 100,000 draws is {checks["stationary_mean_max_error"]:.4f}. No covariance eigenvalue projection is used.\n\n'
    text+=f'Across CRSP folds, median importance-correction effective sample sizes range from {min(meds):.1f} to {max(meds):.1f} out of 512 guided proposals; the minimum daily ESS is {miness:.2f}. The fraction of dates with correction ESS below ten ranges from {min(low)*100:.2f}\\% to {max(low)*100:.2f}\\%. The minimum auxiliary-ancestor ESS is {min(z["min_parent_ess"] for z in numerical):.2f}. Low-ESS dates signal approximation difficulty and qualify the score comparison. Resampling duplicates mean that 4,096 predictive paths are not 4,096 independent posterior states.\n\n'
    stress=next(z for z in comparisons if z['unit']==2020 and z['alpha']==.01)
    text+=f'For 2020 at one percent, joint refinement changes mean FZ0 from {stress["primary"]:.4f} to {stress["refined"]:.4f}; the relative daily ES RMS change is {100*stress["es_relative_rms"]:.1f}\\%. Across the three checked years and both levels, the maximum absolute annual score shift is {max(abs(z["difference"]) for z in comparisons):.3f}. These checks do not establish numerical convergence. The bootstrap intervals condition on these approximate forecasts and do not include particle or discretization uncertainty.\n\n'
    finding=O/'crsp5_findings.tex'
    finding.write_text(finding.read_text()+f'In the 2020 numerical check, the one-percent mean score changes by {stress["difference"]:.3f}, while daily ES changes by {100*stress["es_relative_rms"]:.1f}\\% in relative RMS. Thus aggregate-score stability does not imply accurate daily state estimation; the Wishart comparison remains numerically qualified.\n')
    text+='\\input{wishart_extension/analysis/heston_refinement.tex}\n\\input{wishart_extension/analysis/crsp_refinement.tex}\n'
    (O/'numerical_checks.tex').write_text(text)
    return fits,comparisons


def figures(summ,daily):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(7.2,9.8))
    for ax,law in zip(axs,['t','metat','heston']):
        g=summ[(summ.study==law)&(summ.alpha==.01)&(summ.model!='Historical')].sort_values('delta_mse');y=np.arange(len(g))
        ax.errorbar(g.delta_mse,y,xerr=g.paired_se,fmt='o',ms=4,capsize=2);ax.axvline(0,color='black',lw=.7)
        ax.set_yticks(y,[label(v) for v in g.model],fontsize=8);ax.set_title({'t':'Student target: meta-t correctly specified','metat':'Heterogeneous meta-t: correctly specified','heston':'Heston: includes fitted Heston-Wishart'}[law],fontsize=10)
        ax.set_xlabel('Paired MSE difference ± 1 SE',fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/wishart_paired_mse.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(8.5,7.0),sharex=True)
    models=['Historical','Flow-t','Flow-meta-t','VAE-G','VAE-H','GARCH-FHS','GARCH-EVT','Portfolio-EWMA-FHS',NAME]
    markers=['o','s','^','D','v','P','X','h','*']
    for ax,panel in zip(axs,['crsp5','crsp100']):
        a=daily if panel=='crsp5' else pd.read_csv(OLD/'crsp100_forecasts.csv.gz',parse_dates=['date'])
        a=a[(a.portfolio=='equal')&(a.alpha==.01)].groupby(['date','model'],as_index=False).fz0.mean()
        annual=a.assign(year=a.date.dt.year).groupby(['year','model']).fz0.mean().unstack();annual.to_csv(O/(panel+'_annual.csv'))
        for i,name in enumerate(models):
            if name not in annual:continue
            ax.plot(annual.index,annual[name]-annual.Historical,marker=markers[i],ms=5,lw=1.6 if name==NAME else 1.1,ls=['-','--',':','-.'][i%4],label=label(name).replace('Portfolio-EWMA-FHS','Portfolio EWMA-FHS'))
        ax.set_title('Five size baskets' if panel=='crsp5' else '100-stock annual universes (unchanged)');ax.set_ylabel('FZ0 minus FHS');ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,fontsize=7,loc='upper center',bbox_to_anchor=(.52,.995));axs[-1].set_xlabel('Out-of-sample year');fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(ROOT/'figures/wishart_annual.pdf');plt.close(fig)


def main():
    s=synthetic();c,d=crsp();fits,num=diagnostics();figures(s,d)
    record=dict(primary_fits=20,synthetic_scenarios=10000000,crsp_test_dates=2514,crsp_new_forecasts=2514*6,
        old_forecasts_unchanged=True,crsp5_models=16,crsp5_holm_contrasts=15,crsp100_unchanged=True,
        kernel_checks_passed=True,refinement_fits=0,refinement_evaluations=6,
        numerical_fz0_max_abs_shift=max(abs(a['difference']) for a in num),
        boundary_shape_fits=sum(a['beta']==6 for a in fits))
    (O/'verification.json').write_text(json.dumps(record,indent=2))
    print(c[c.model.eq(NAME)].to_string(index=False));print(s[(s.model==NAME)&(s.alpha==.01)].to_string(index=False));print(json.dumps(record,indent=2))

if __name__=='__main__':main()

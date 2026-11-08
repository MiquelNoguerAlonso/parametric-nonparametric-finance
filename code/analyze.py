"""Build all publication tables and figures from completed run records."""
import json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
ORDER=['Historical','Gaussian','Meta-t','Flow-G','Flow-t','Flow-L','Flow-meta-t','VAE-G','VAE-t','VAE-H','WGAN-G','WGAN-t','WGAN-i']
def fmt(x,n=3):return '--' if x is None or not np.isfinite(x) else f'{x:.{n}f}'
def tex_table(caption,label,head,rows,align=None):
    align=align or 'l'+'r'*(len(head)-1)
    return ('\\begin{table}[htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n'+
      '\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(head)+' \\\\\n\\midrule\n'+
      '\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'+
      '\\caption{'+caption+'}\\label{'+label+'}\n\\end{table}\n')

def simulations():
    summaries=[];tables=[];repair=[];heston=[];allrecords=[]
    for study,expected in [('t',60),('metat',66),('heston',54)]:
        records=[json.loads(p.read_text()) for p in RES.glob(study+'_*.json') if p.name!=study+'_truth.json']
        records=[r for r in records if r.get('study')==study]
        assert len(records)==expected,(study,len(records),expected)
        allrecords+=records;names=[n for n in ORDER if n in {r['model'] for r in records}];rows=[]
        for name in names:
            group=[r for r in records if r['model']==name]
            vals=lambda key:[r[key] for r in group]
            metric=lambda alpha:[next(v for v in r['metrics'] if v['portfolio']=='equal' and v['alpha']==alpha) for r in group]
            a,b=metric(.01),metric(.001);err=np.array([m['total_error'] for m in a]);errb=np.array([m['total_error'] for m in b])
            nll=None if group[0]['nll'] is None else float(np.mean(vals('nll')))
            row={'study':study,'model':name,'nll':nll,'es_bias_1':float(err.mean()),'es_rmse_1':float(np.sqrt(np.mean(err**2))),
              'es_rmse_01':float(np.sqrt(np.mean(errb**2))),'w1':float(np.mean([m['w1_mc'] for m in a])),
              'hill1':float(np.mean(vals('hill1'))),'hill01':float(np.mean(vals('hill01'))),'runs':len(group)}
            summaries.append(row);rows.append(['VAE-tlatent' if name=='VAE-t' else 'Empirical' if name=='Historical' else name,fmt(nll),fmt(row['es_bias_1']),fmt(row['es_rmse_1']),fmt(row['w1'])])
            if study!='heston':
                for q in [.025,.05,.1]:
                    rs=[next(v for v in r['transplant'] if v['q']==q) for r in group]
                    ev=[r['total_error'] for r in rs if r['total_error'] is not None]
                    repair.append({'study':study,'model':name,'threshold':q,'valid':len(ev),'rmse':float(np.sqrt(np.mean(np.array(ev)**2))) if ev else None,
                      'before_rmse':row['es_rmse_1'],'min_xi':min(r['xi'] for r in rs),'max_xi':max(r['xi'] for r in rs)})
            else:heston.append({'model':name,**{k:float(np.mean([r['stylised'][k] for r in group])) for k in group[0]['stylised']}})
        title={'t':'Bivariate Student-$t_3$','metat':'Heterogeneous meta-$t$','heston':'Five-asset Heston weekly-window'}[study]
        tables.append(tex_table(title+({'t':': known analytic tail risk. ','metat':': heterogeneous marginal indices. ','heston':': numerical stochastic-volatility target. '}[study])+'Design: three independent estimation datasets and two optimization seeds (six runs). ES bias and RMSE are measured against the analytic or Monte Carlo reference, not training-sample ES. The one-percent level is the primary simulation comparison; noisy 0.1\\% diagnostics remain in the machine-readable results. NLL is exact-density evaluation for flows/baselines and a Monte Carlo IWAE-64 estimate for VAEs; no likelihood is assigned to GANs. $W_1$ is the mean empirical portfolio distance to the Monte Carlo reference. Units are the standardized return units defined in the design.',
           'tab:sim-'+study,['Model','NLL','ES bias 1\\%','RMSE 1\\%','$W_1$'],rows))
    summ=pd.DataFrame(summaries);summ.to_csv(RES/'simulation_summary.csv',index=False)
    rep=pd.DataFrame(repair);rep.to_csv(RES/'transplant_summary.csv',index=False)
    hs=pd.DataFrame(heston);hs.to_csv(RES/'heston_stylised_summary.csv',index=False)
    truth=json.loads((RES/'heston_truth.json').read_text())['stylised']
    rows=[['Reference']+[fmt(truth[k]) for k in ['kurtosis','cluster','leverage','cross_vol']]]
    rows += [[r['model']]+[fmt(r[k]) for k in ['kurtosis','cluster','leverage','cross_vol']] for r in heston]
    tables.append(tex_table('Heston finite-sample diagnostics. Kurtosis is sample excess kurtosis; clustering is the mean adjacent-day absolute-return correlation; leverage is return versus next-day absolute-return correlation; cross-volatility is same-day cross-asset absolute-return correlation. Extreme sample kurtosis for a $t_3$ model is not an estimator of a finite population fourth moment.',
        'tab:heston-dynamics',['Model','Kurtosis','Cluster','Leverage','Cross-vol'],rows))
    rows=[]
    for law in ['t','metat']:
        for name in ['Flow-G','Meta-t','Flow-meta-t','WGAN-G','WGAN-t']:
            a=rep[(rep.study==law)&(rep.model==name)]
            if len(a):rows.append([law,name,fmt(a.iloc[0].before_rmse)]+[fmt(a[a.threshold==q].iloc[0].rmse) for q in [.025,.05,.1]])
    tables.append(tex_table('Radial-splice sensitivity: equal-weight 1\\% ES RMSE before repair and after GPD fits at three tail fractions. The table includes adverse repairs. Shapes and rejected finite-ES fits are recorded in the raw results. Six runs per entry share only three independent tail-estimation samples.',
        'tab:repair',['Law','Model','Before','2.5\\%','5\\%','10\\%'],rows,align='llrrrr'))
    (RES/'simulation_tables.tex').write_text('\n'.join(tables))
    fig,axs=plt.subplots(1,2,figsize=(9.8,3.5))
    for ax,study in zip(axs,['t','metat']):
        ref=np.load(ROOT/'data'/f'{study}_truth.npz')['X'];truthloss=np.sort(-ref.mean(1));grid=np.geomspace(.5,40,200)
        ax.loglog(grid,np.maximum(1-np.searchsorted(truthloss,grid)/len(truthloss),1/len(truthloss)),color='black',lw=2,label='Reference MC')
        for name in ['Flow-G','Meta-t','VAE-G','WGAN-G','WGAN-t']+(['Flow-meta-t'] if study=='metat' else []):
            x=np.load(ROOT/'data'/f'curve_{study}_{name}.npz')['loss'];sf=1-np.searchsorted(x,grid)/len(x)
            ax.loglog(grid,np.where(sf>0,sf,np.nan),label=name,lw=1.2)
        ax.set(xlabel='Portfolio loss',ylabel='Survival probability',title=study,ylim=(1e-5,1));ax.grid(alpha=.2)
    axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/'controlled_survival.pdf');plt.close(fig)
    # Finite-level claims are generated from aggregate metrics only.
    text=['\\subsection{Controlled results and their interpretation}']
    for study in ['t','metat','heston']:
        a=summ[summ.study==study].sort_values('es_rmse_1');best=a.iloc[0];g=a[a.model=='Gaussian'].iloc[0]
        text.append(f"In the {study} study, {best['model']} has the smallest observed 1\\% ES RMSE ({best.es_rmse_1:.3f}), compared with {g.es_rmse_1:.3f} for the Gaussian baseline. This is a ranking within six runs under the stated budget, not a family-level dominance result.")
    text.append('The 0.1\\% results use only about 100 generated tail observations per model and a finite Monte Carlo reference for meta-$t$ and Heston; they are stress diagnostics with substantial simulation uncertainty. The independent-noise GAN is an ablation of its joint input law, not a test of the elliptical-noise theorem. Any recovery of the target at a finite threshold is compatible with a different asymptotic class.')
    text.append('The splice results show why universal removal of architecture error is too strong a claim: its performance depends on the fitted radial shape, threshold, and angular structure. The package retains fit discrepancy and sampling discrepancy separately from the reported total error.')
    text.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=0.98\\linewidth]{figures/controlled_survival.pdf}\\caption{Illustrative survival functions from dataset 0, seed 0. All aggregate conclusions use the complete run panel; this figure is not a selection criterion.}\\end{figure}')
    (RES/'simulation_findings.tex').write_text('\n\n'.join(text))

def bootstrap_means(matrix,block=20,B=1999,seed=20261106):
    n,k=matrix.shape;r=np.random.default_rng(seed);out=[]
    for start in range(0,B,100):
        m=min(100,B-start);beg=r.integers(0,n,size=(m,(n+block-1)//block))
        idx=(beg[:,:,None]+np.arange(block))%n;idx=idx.reshape(m,-1)[:,:n]
        out.append(matrix[idx].mean(axis=1))
    return np.concatenate(out)

def holm(p):
    p=np.array(p);order=np.argsort(p);adj=np.empty(len(p));ranked=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    adj[order]=np.minimum(1,ranked);return adj

def crsp():
    files=sorted(RES.glob('crsp_*.csv'));files=[p for p in files if p.stem.split('_')[1].isdigit()]
    assert len(files)==200,len(files)
    records=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in files],ignore_index=True)
    records.to_csv(RES/'crsp_all_forecasts.csv.gz',index=False,compression='gzip')
    assert not records.duplicated(['date','seed','model','portfolio','alpha']).any()
    fit=[json.loads(p.read_text()) for p in RES.glob('crsp_*.json') if p.stem.split('_')[1].isdigit()]
    assert len(fit)==200
    f=pd.DataFrame(fit);nll={}
    for name,g in f.groupby('model'):
        ok=g.nll_percent_return.notna();nll[name]=float(np.average(g.loc[ok,'nll_percent_return'],weights=g.loc[ok,'n_test'])) if ok.any() else None
    daily=records.groupby(['date','model','portfolio','alpha'],as_index=False).agg(fz0=('fz0','mean'),hit=('hit','mean'),es=('es','mean'),es_residual=('es_residual','mean'),loss=('loss','mean'),var=('var','mean'))
    summary=[];contrasts=[];blockrows=[]
    for (pn,alpha),g in daily.groupby(['portfolio','alpha']):
        scores=g.pivot(index='date',columns='model',values='fz0').sort_index();cols=list(scores.columns)
        assert not scores.isna().any().any();diff=scores.subtract(scores.Gaussian,axis=0)
        boots=bootstrap_means(diff.values);means=diff.mean().values
        lower,upper=np.quantile(boots,[.025,.975],axis=0)
        ps=(1+(np.abs(boots-means)>=np.abs(means)).sum(0))/(len(boots)+1)
        nong=[i for i,k in enumerate(cols) if k!='Gaussian'];adjusted=holm(ps[nong]);adj={cols[i]:v for i,v in zip(nong,adjusted)}
        for i,name in enumerate(cols):
            a=g[g.model==name]
            row={'portfolio':pn,'alpha':alpha,'model':name,'n_dates':len(a),'nll':nll[name],
                 'fz0':float(a.fz0.mean()),'coverage':float(a.hit.mean()),'mean_es':float(a.es.mean()),
                 'normalized_es_residual':float(a.es_residual.mean()/a.es.mean()),'delta_gaussian':float(means[i]),
                 'delta_low':float(lower[i]),'delta_high':float(upper[i]),'p_boot':float(ps[i]),'p_holm':float(adj.get(name,1))}
            summary.append(row)
        if pn=='equal' and alpha==.01:
            for block in [10,20,40]:
                bo=boots if block==20 else bootstrap_means(diff.values,block=block)
                lo,hi=np.quantile(bo,[.025,.975],axis=0)
                blockrows += [{'block':block,'model':name,'difference':float(means[i]),'low':float(lo[i]),'high':float(hi[i])} for i,name in enumerate(cols)]
    summ=pd.DataFrame(summary);summ.to_csv(RES/'crsp_summary.csv',index=False)
    pd.DataFrame(blockrows).to_csv(RES/'crsp_block_sensitivity.csv',index=False)
    # Stress the observed losses with all missing held returns set to -100%,
    # holding the original forecasts fixed; this is not a retrained backtest.
    audit=pd.read_csv(ROOT/'data'/'crsp_holdings_audit.csv',parse_dates=['date'])
    mw=audit.pivot(index='date',columns='bucket',values='missing_weight')
    stress=[]
    for pn in ['equal','small','large']:
        adjustment=100*(mw.mean(1) if pn=='equal' else mw[1] if pn=='small' else mw[5])
        a=records[records.portfolio==pn].copy();a['stress_loss']=a.loss+a.date.map(adjustment)
        a['stress_fz0']=np.log(a.es)+a['var']/a.es-1+np.maximum(a.stress_loss-a['var'],0)/(a.alpha*a.es)
        a['stress_hit']=(a.stress_loss>a['var']).astype(float)
        for (name,alpha),g in a.groupby(['model','alpha']):
            stress.append({'portfolio':pn,'model':name,'alpha':alpha,'base_score':float(g.fz0.mean()),'stress_score':float(g.stress_fz0.mean()),'base_coverage':float(g.hit.mean()),'stress_coverage':float(g.stress_hit.mean())})
    pd.DataFrame(stress).to_csv(RES/'crsp_missing_return_stress.csv',index=False)
    tables=[]
    for alpha in [.01,.025]:
        a=summ[(summ.portfolio=='equal')&(summ.alpha==alpha)].set_index('model');rows=[]
        for name in ORDER:
            if name not in a.index:continue
            r=a.loc[name];ci='['+fmt(r.delta_low)+', '+fmt(r.delta_high)+']'
            rows.append([name,fmt(r.nll),fmt(r.fz0),fmt(100*r.coverage,2),fmt(r.normalized_es_residual),fmt(r.delta_gaussian),ci])
        tables.append(tex_table(f'CRSP equal-weight portfolio, {100*alpha:g}\\% tail probability, 2016--2025. NLL evaluates the joint five-portfolio density in percentage-return units; it is an IWAE-64 estimate for VAEs. Coverage is expressed as a percentage. The ES moment is divided by mean forecast ES. Score differences use Gaussian as the benchmark; negative is better. Intervals are paired 95\\% circular block-bootstrap intervals (20-day blocks) on seed-averaged daily score differences, with {int(a.iloc[0].n_dates)} dates. Secondary-level intervals are descriptive.',
            'tab:crsp-'+str(alpha),['Model','NLL','FZ0','Hits \\%','ES moment','$\\Delta$ score','95\\% interval'],rows))
    a=summ[(summ.portfolio=='equal')&(summ.alpha==.01)].sort_values('fz0');best=a.iloc[0];gauss=a[a.model=='Gaussian'].iloc[0];mt=a[a.model=='Meta-t'].iloc[0]
    rows=[]
    for pn in ['small','large']:
        b=summ[(summ.portfolio==pn)&(summ.alpha==.01)]
        for name in ['Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']:
            r=b[b.model==name].iloc[0];rows.append([pn,name,fmt(r.fz0),fmt(100*r.coverage,2),fmt(r.delta_gaussian)])
    tables.append(tex_table('Concentrated-portfolio robustness at the 1\\% level. Small and large refer to the smallest and largest size baskets, not an ex-post selection of individual surviving stocks. These are descriptive secondary comparisons.',
        'tab:crsp-concentrated',['Portfolio','Model','FZ0','Hits \\%','$\\Delta$ Gaussian'],rows,align='llrrr'))
    (RES/'crsp_tables.tex').write_text('\n'.join(tables))
    # Annual primary scores expose regime dependence rather than only a mean.
    aa=daily[(daily.portfolio=='equal')&(daily.alpha==.01)].copy();aa['year']=aa.date.dt.year
    annual=aa.groupby(['year','model']).fz0.mean().unstack();annual.to_csv(RES/'crsp_annual_scores.csv')
    fig,ax=plt.subplots(figsize=(9,4))
    for name in ['Gaussian','Meta-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']:
        ax.plot(annual.index,annual[name],marker='o',ms=3,label=name)
    ax.set(xlabel='Out-of-sample year',ylabel='Mean FZ0 score (lower is better)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'crsp_annual_scores.pdf');plt.close(fig)
    text=['\\subsection{Out-of-sample findings}']
    text.append(f"For the primary equal-weight 1\\% comparison, the lowest mean score is {best['model']} ({best.fz0:.4f}). The Gaussian score is {gauss.fz0:.4f}, and the plain meta-$t$ score is {mt.fz0:.4f}. The best observed model's score difference from Gaussian is {best.delta_gaussian:.4f}, with a paired 95\\% interval [{best.delta_low:.4f}, {best.delta_high:.4f}]. Its primary-family Holm-adjusted bootstrap $p$-value is {best.p_holm:.4f}. Selection of the best observed score does not turn that ranking into a general architectural theorem.")
    text.append(f"The primary best-score model has an exceedance rate of {100*best.coverage:.2f}\\%, compared with the nominal 1\\%, and a normalized ES moment of {best.normalized_es_residual:.3f}. Score ranking and calibration therefore need to be read jointly. A model can improve the joint score yet retain a systematic calibration error.")
    exact=a[~a.model.str.startswith('VAE')&a.nll.notna()].sort_values('nll').iloc[0]
    residual=a[a.model=='Flow-meta-t'].iloc[0]
    text.append(f"Distributional fit gives a different ranking: {exact['model']} has the smallest average NLL among the exact-density models ({exact.nll:.4f}), compared with {residual.nll:.4f} for Flow-meta-$t$. The VAE columns are IWAE-64 estimates and do not establish an exact-likelihood ordering against that baseline. Flow-meta-$t$ has a normalized ES moment of {residual.normalized_es_residual:.4f}, closer to zero than that of the best-score model. Likelihood, joint risk score, and the ES calibration moment answer different questions.")
    text.append('The separate 2.5\\% and concentrated-portfolio results show the dependence of the comparison on the risk functional and portfolio. Annual scores expose regime dependence. Neither the empirical sequence of losses nor a test-window plug-in ES reveals the true conditional ES; the reported score and moment comparisons avoid calling a noisy realized tail average an oracle.')
    text.append('The complete outputs include the missing-return stress and block-length sensitivities. All model families share the same predictable volatility filter, so these results primarily compare standardized-innovation laws and their finite-budget estimation. They do not establish that a static VAE or GAN has learned a dynamic variance process.')
    text.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=0.98\\linewidth]{figures/crsp_annual_scores.pdf}\\caption{Annual out-of-sample joint VaR--ES scores for the equal-weight portfolio at 1\\%. Scores average the two optimization/scenario seeds.}\\end{figure}')
    (RES/'crsp_findings.tex').write_text('\n\n'.join(text))
    print('CRSP primary:');print(a[['model','fz0','coverage','delta_gaussian','delta_low','delta_high','p_holm']].to_string(index=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['all','simulation','crsp'],default='all');a=p.parse_args()
    if a.part in ['all','simulation']:simulations()
    if a.part in ['all','crsp']:crsp()
if __name__=='__main__':main()

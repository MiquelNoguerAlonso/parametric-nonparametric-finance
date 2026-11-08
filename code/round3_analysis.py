"""Symmetric primary inference and separately labeled retrospective controls."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
from scipy import stats
from analyze import tex_table,fmt,bootstrap_means,holm
from revision_analysis import inference,mcs,label,pformat
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'referee_round2/analysis';O=ROOT/'referee_round3/analysis';R=ROOT/'referee_round3/results'

def scores(panel,alpha=.01,seed=None):
    x=pd.read_csv(OLD/f'{panel}_forecasts.csv.gz',parse_dates=['date'])
    x=x[(x.portfolio=='equal')&(x.alpha==alpha)]
    if seed is not None:x=x[x.seed==seed]
    return x.groupby(['date','model']).fz0.mean().unstack().sort_index(),x

def primary():
    records=[];powers=[];mcsrows=[];pairs=[]
    for panel in ['crsp5','crsp100']:
        for alpha in [.01,.025]:
            matrix,data=scores(panel,alpha);assert matrix.shape==(2514,15)
            inf=inference(matrix,'Historical');mc=mcs(matrix).set_index('model');rows=[]
            for name in matrix.mean().sort_values().index:
                z=data[data.model==name];v=inf.set_index('model').loc[name]
                r=dict(panel=panel,alpha=alpha,model=name,fz0=float(matrix[name].mean()),hits_percent=100*z.hit.mean(),
                    es_moment=z.es_residual.mean()/z.es.mean(),delta_fhs=v.delta,low=v.low,high=v.high,p=v.p,p_holm=v.p_holm,mcs_p=mc.loc[name,'p_mcs'])
                records.append(r);rows.append([label(name).replace('Portfolio-EWMA-FHS','Port.-EWMA-FHS'),fmt(r['fz0']),fmt(r['hits_percent'],2),fmt(r['es_moment']),fmt(v.delta),f'[{v.low:.3f}, {v.high:.3f}]',pformat(v.p_holm),pformat(r['mcs_p'])])
            title='Five-basket' if panel=='crsp5' else '100-stock'
            suffix='primary' if alpha==.01 else 'secondary'
            (O/f'{panel}_{suffix}.tex').write_text(tex_table(f'{title} equal-portfolio {100*alpha:g}\\% risk on 2,514 common dates. Fifteen models and fourteen FHS contrasts; the numerically unresolved Wishart CRSP pilot is excluded. Intervals and MCS use 9,999 circular draws and 20-day blocks; Holm adjusts the within-panel family.','tab:round3-'+panel+'-'+suffix,['Model','FZ0','Hits \\%','ES mom.','$\\Delta$ FHS','95\\% interval','Holm $p$','MCS $p$'],rows))
            if alpha!=.01:continue
            d=matrix[['GARCH-FHS','GARCH-EVT']].subtract(matrix.Historical,axis=0);mean=d.mean().values
            noise=bootstrap_means(d.values,block=20,B=9999)-mean
            for j,name in enumerate(d):
                a=inf.set_index('model').loc[name];n=noise[:,j]
                mdes=[]
                for family in [1,14]:
                    lo,hi=np.quantile(n,[.05/(2*family),1-.05/(2*family)])
                    left,right=0.,float(10*np.std(n))
                    for _ in range(60):
                        mid=(left+right)/2;power=np.mean((n-mid<lo)|(n-mid>hi))
                        if power>=.8:right=mid
                        else:left=mid
                    mdes.append(right)
                powers.append(dict(panel=panel,model=name,delta=a.delta,low=a.low,high=a.high,halfwidth=(a.high-a.low)/2,
                    bootstrap_se=float(n.std(ddof=1)),mde80_unadjusted=mdes[0],mde80_bonferroni14=mdes[1]))
            reduced=matrix.drop(columns=['EWMAcov-FHS','Portfolio-EWMA-FHS']);mr=mcs(reduced).set_index('model')
            for name in matrix:
                mcsrows.append(dict(panel=panel,model=name,mean=matrix[name].mean(),full_p=mc.loc[name,'p_mcs'],reduced_p=mr.loc[name,'p_mcs'] if name in mr.index else np.nan))
            for lhs,rhs in [('Flow-meta-t','Historical'),('Flow-meta-t','Meta-t'),('Flow-meta-t','VAE-G'),('EWMAcov-FHS','Historical')]:
                v=matrix[lhs]-matrix[rhs];noise_pair=bootstrap_means(v.values[:,None],block=20,B=9999)[:,0]-v.mean()
                pi=inference(matrix[[lhs,rhs]],rhs).set_index('model').loc[lhs]
                pairs.append(dict(panel=panel,left=lhs,right=rhs,delta=v.mean(),bootstrap_se=noise_pair.std(ddof=1),low=pi.low,high=pi.high,p=pi.p))
    rec=pd.DataFrame(records);rec.to_csv(O/'primary_summary.csv',index=False)
    pd.DataFrame(powers).to_csv(O/'power_diagnostics.csv',index=False)
    (O/'power_table.tex').write_text(tex_table('Resolution at one percent for the GARCH/FHS contrasts. Half-width is the unadjusted 95\\% interval width divided by two. MDE columns use an empirical location-shift experiment with the centered 20-day bootstrap noise and target 80\\% power: an unadjusted two-sided 5\\% test, or the conservative Bonferroni 14-contrast threshold. These conditional planning diagnostics hold the fitted forecast and dependence structure fixed; they are neither a prospective power guarantee nor a power calculation for the full adaptive Holm procedure.','tab:round3-power',['Panel','Contrast','$\\Delta$ FHS','CI half-width','MDE 80\\%','MDE/14'],[[p['panel'].replace('crsp',''),p['model'],fmt(p['delta']),fmt(p['halfwidth']),fmt(p['mde80_unadjusted']),fmt(p['mde80_bonferroni14'])] for p in powers],'llrrrr'))
    pd.DataFrame(mcsrows).to_csv(O/'mcs_sensitivity.csv',index=False);pd.DataFrame(pairs).to_csv(O/'pairwise_precision.csv',index=False)
    (O/'mcs_table.tex').write_text(tex_table('MCS membership sensitivity at one percent. The reduced set omits EWMAcov-FHS and portfolio-EWMA-FHS, leaving thirteen models. This is a retrospective set-composition diagnostic, not a replacement primary test. Membership does not order retained models by quality. Full results for every model are in the accompanying CSV.','tab:round3-mcs',['Panel','Model','Mean FZ0','Full MCS $p$','Reduced $p$'],[[r['panel'].replace('crsp',''),label(r['model']),fmt(r['mean']),pformat(r['full_p']),pformat(r['reduced_p'])] for r in mcsrows if r['model'] in ['Historical','Flow-meta-t','VAE-G','Gaussian']],'llrrr'))
    (O/'pairwise_table.tex').write_text(tex_table('Pairwise score precision at one percent. SE is the standard deviation of centered circular-bootstrap mean differences. Intervals and $p$-values are unadjusted descriptive diagnostics.','tab:round3-pairs',['Panel','Difference','$\\Delta$','Boot. SE','95\\% interval','$p$'],[[r['panel'].replace('crsp',''),label(r['left'])+' -- '+label(r['right']),fmt(r['delta']),fmt(r['bootstrap_se']),f'[{r["low"]:.3f}, {r["high"]:.3f}]',pformat(r['p'])] for r in pairs],'llrrrr'))
    for panel in ['crsp5','crsp100']:
        p=rec[(rec.panel==panel)&(rec.alpha==.01)].set_index('model');assert not (p.p_holm<.05).any()
        best=p.fz0.idxmin();b=p.loc[best]
        text='\\subsection{'+('Five-basket findings' if panel=='crsp5' else '100-stock findings')+'}\n'
        text+=f'No model, neural or statistical, is distinguishable from FHS at five percent after the fourteen-contrast within-panel Holm correction. The lowest observed mean is {best} at {b.fz0:.4f}, versus FHS {p.loc["Historical","fz0"]:.4f}. '
        for name in ['GARCH-EVT','GARCH-FHS']:
            q=p.loc[name];text+=f'{name} has difference {q.delta_fhs:.4f}, interval $[{q.low:.4f}, {q.high:.4f}]$, and Holm $p={q.p_holm:.4f}$. '
        text+='These non-rejections do not establish equality or an advantage for the statistical family. Table~\\ref{tab:round3-power} quantifies the resolution.\n\n'
        text+=f'The primary 95\\% MCS retains {int((p.mcs_p>=.05).sum())} of fifteen models. Its studentized pairwise statistic measures differences relative to their precision; it does not rank models by mean score. '
        if panel=='crsp100':text+='Flow-meta-$t$ is excluded even though its mean is below that of retained VAE-G. Its most adverse standardized pair is against plain meta-$t$, not FHS; the comparison with VAE-G is imprecise. '
        text+='Set-composition and pairwise-precision diagnostics are in Appendix~\\ref{sec:round3-diagnostics}.\n'
        (O/f'{panel}_findings.tex').write_text(text)
    marg=[];cop=[]
    for path in (ROOT/'referee_round2/checkpoints').glob('heston_*_0_Meta-t.meta.json'):
        x=json.loads(path.read_text());marg.extend(v[0] for v in x['marg'])
    for panel in ['crsp5','crsp100']:
        vals=[json.loads((ROOT/'expansion/checkpoints'/f'{panel}_{year}_0_Meta-t.meta.json').read_text())['nu'] for year in range(2016,2026)]
        cop.append(dict(panel=panel,mean=float(np.mean(vals)),min=float(np.min(vals)),max=float(np.max(vals))))
    (O/'margin_diagnostics.json').write_text(json.dumps(dict(heston_marginal_df_min=min(marg),heston_marginal_df_median=float(np.median(marg)),heston_marginal_df_max=max(marg),heston_margins_fourth_moment_infinite=sum(v<=4 for v in marg),heston_margins=len(marg),copula=cop),indent=2))
    pilot=pd.read_csv(ROOT/'wishart_extension/analysis/crsp5_summary.csv');pilot=pilot[(pilot.alpha==.01)&pilot.model.isin(['Historical','Heston-Wishart'])]
    (O/'wishart_pilot.tex').write_text(tex_table('Descriptive five-basket one-percent CRSP pilot on 2,514 dates. The Wishart approximation is numerically unresolved and is excluded from primary inference; no Holm or MCS result is reported for it. The FHS row supplies scale only. Date-block uncertainty cannot account for the particle/discretization error documented below.','tab:wishart-pilot',['Model','FZ0','Hits \\%','ES moment'],[[label(r.model),fmt(r.fz0),fmt(r.hits_percent,2),fmt(r.es_moment)] for r in pilot.itertuples()],'lrrr'))
    print(rec[(rec.alpha==.01)&(rec.model.isin(['GARCH-EVT','GARCH-FHS']))].to_string(index=False));print(pd.DataFrame(powers).to_string(index=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['primary'],default='primary');a=p.parse_args();primary()

if __name__=='__main__':main()

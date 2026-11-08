"""Paired exploratory analyses; original primary families are untouched."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze import tex_table,fmt,holm,bootstrap_means
from revision_analysis import pformat,inference
from round3_analysis import scores
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'referee_round3/results';O=ROOT/'referee_round3/analysis';S=ROOT/'referee_round2/selection'
RULE={'fz1':'Risk 1\\%','density':'Likelihood','final':'Final'}
def read(p):return json.loads(p.read_text())
def pred(root,name,alpha):
    x=pd.read_csv(root/name,parse_dates=['date']);x=x[(x.alpha==alpha)&(x.portfolio=='equal')].sort_values('date').set_index('date')
    assert x.index.is_unique
    return x
def contrast(series):
    m=series.mean();n=bootstrap_means(series.values[:,None],B=9999)[:,0]-m
    p=min(1.,2*min((1+(n>=m).sum())/10000,(1+(n<=m).sum())/10000))
    return dict(delta=m,low=m-np.quantile(n,.975),high=m-np.quantile(n,.025),p=p)
def adjust(rows):
    for r,p in zip(rows,holm([r['p'] for r in rows])):r['p_holm']=p
def interval(r):return f'[{r["low"]:.3f}, {r["high"]:.3f}]'
def main():
    O.mkdir(exist_ok=True,parents=True);lu=[];syn=[];coord=[];within=[];against=[];nll=[]
    for panel in ['crsp5','crsp100']:
        for alpha in [.01,.025]:
            family=[]
            for name in ['Flow-G','Flow-t']:
                for rule in RULE:
                    new=[];old=[];new_nll=[];old_nll=[];weights=[]
                    for year in range(2016,2026):
                        a=read(R/f'lu_{panel}_{year}_{name}.json')['evaluations'][rule];b=read(S/f'{panel}_{year}_0_{name}.json')['evaluations'][rule]
                        x=pred(R,a['forecasts'],alpha);y=pred(S,b['forecasts'],alpha);assert x.index.equals(y.index)
                        assert np.max(np.abs(x.loss-y.loss))<1e-8
                        new.append(x);old.append(y);new_nll.append(a['nll']);old_nll.append(b['nll']);weights.append(len(x))
                    x=pd.concat(new).sort_index();y=pd.concat(old).sort_index();assert len(x)==2514
                    r=dict(panel=panel,alpha=alpha,model=name,rule=rule,fz0=x.fz0.mean(),old_fz0=y.fz0.mean(),hits_percent=100*x.hit.mean(),nll=np.average(new_nll,weights=weights),old_nll=np.average(old_nll,weights=weights),**contrast(x.fz0-y.fz0));family.append(r)
            adjust(family);lu+=family
            base,raw=scores(panel,alpha,seed=0);family=[];newframes={}
            for kind,name in [('garch',x) for x in ['FHS','Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t']]+[('ewma','FHS')]:
                frames=[];nlls=[];weights=[]
                for year in range(2016,2026):
                    p=R/(f'coord_{panel}_{year}_{name}.json' if name.startswith('Flow') else f'coord_{kind}_{panel}_{year}_{name}.json')
                    a=read(p);ev=a['evaluations']['fz1'] if name.startswith('Flow') else a;x=pred(R,ev['forecasts'],alpha)
                    frames.append(x);nlls.append(ev['nll']);weights.append(len(x))
                x=pd.concat(frames).sort_index();assert x.index.equals(base.index);original='Historical' if name=='FHS' else name
                original_loss=raw[raw.model==original].groupby('date').loss.mean().sort_index();assert np.max(np.abs(x.loss-original_loss))<1e-8
                r=dict(panel=panel,alpha=alpha,filter=kind,model=name,fz0=x.fz0.mean(),old_fz0=base[original].mean(),hits_percent=100*x.hit.mean(),nll=np.average(nlls,weights=weights) if nlls[0] is not None else np.nan,**contrast(x.fz0-base[original]))
                family.append(r);newframes[(kind,name)]=x
            adjust(family);coord+=family
            family=[]
            for name in ['Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t']:
                family.append(dict(panel=panel,alpha=alpha,model=name,**contrast(newframes[('garch',name)].fz0-newframes[('garch','FHS')].fz0)))
            adjust(family);within+=family
            family=[]
            for kind,name in newframes:
                family.append(dict(panel=panel,alpha=alpha,filter=kind,model=name,**contrast(newframes[(kind,name)].fz0-base['GARCH-FHS'])))
            adjust(family);against+=family
    augmented=[]
    for panel in ['crsp5','crsp100']:
        matrix,_=scores(panel,.01)
        for name in ['Flow-G','Flow-t']:
            frames=[pred(R,read(R/f'lu_{panel}_{year}_{name}.json')['evaluations']['fz1']['forecasts'],.01) for year in range(2016,2026)]
            matrix['LU-'+name]=pd.concat(frames).fz0
        for kind,name in [('garch',x) for x in ['FHS','Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t']]+[('ewma','FHS')]:
            frames=[]
            for year in range(2016,2026):
                a=read(R/(f'coord_{panel}_{year}_{name}.json' if name.startswith('Flow') else f'coord_{kind}_{panel}_{year}_{name}.json'))
                ev=a['evaluations']['fz1'] if name.startswith('Flow') else a;frames.append(pred(R,ev['forecasts'],.01))
            matrix['Coord-'+kind+'-'+name]=pd.concat(frames).fz0
        assert matrix.shape==(2514,24) and matrix.notna().all().all()
        z=inference(matrix,'Historical');z['panel']=panel;z['fz0']=z.model.map(matrix.mean());augmented.append(z)
        matrix.to_csv(O/(panel+'_augmented_daily.csv.gz'))
    aug=pd.concat(augmented);aug.to_csv(O/'augmented_24_summary.csv',index=False)
    (O/'augmented_24.tex').write_text(tex_table('Post-result omnibus sensitivity: the fifteen primary models plus all nine new one-percent-risk-selected controls, giving 24 models and 23 FHS contrasts per panel. The original seed convention is retained; new controls use seed zero. Likelihood/final alternatives are not included in this model-set sensitivity. This is a retrospective broad-family check, not a replacement primary design. Only the nine new rows are displayed; every row is archived.','tab:augmented24',['Panel','Added model','FZ0','$\\Delta$ FHS','95\\% interval','Holm/23 $p$'],[[r.panel.replace('crsp',''),r.model.replace('Coord-garch-','GARCH-').replace('Coord-ewma-','Fitted-EWMA-'),fmt(r.fz0),fmt(r.delta),f'[{r.low:.3f}, {r.high:.3f}]',pformat(r.p_holm)] for r in aug.itertuples() if r.model.startswith(('Coord-','LU-'))]))
    for name in ['Flow-G','Flow-t']:
        for rule in RULE:
            for alpha in [.01,.001]:
                errors=[];old_errors=[];empirical=[];nn=[];on=[]
                for unit in range(10):
                    a=read(R/f'lu_heston_{unit}_{name}.json')['evaluations'][rule];b=read(S/f'heston_{unit}_0_{name}.json')['evaluations'][rule]
                    aa=next(z for z in a['metrics'] if z['alpha']==alpha);bb=next(z for z in b['metrics'] if z['alpha']==alpha and z['portfolio']=='equal')
                    assert abs(aa['truth']-bb['truth'])<1e-8
                    errors.append(aa['total_error']);old_errors.append(bb['total_error']);empirical.append(aa['sample_error']);nn.append(a['nll']);on.append(b['nll'])
                diff=np.square(errors)-np.square(old_errors)
                de=np.square(errors)-np.square(empirical)
                syn.append(dict(model=name,rule=rule,alpha=alpha,rmse=np.sqrt(np.mean(np.square(errors))),old_rmse=np.sqrt(np.mean(np.square(old_errors))),delta_mse=diff.mean(),se=diff.std(ddof=1)/np.sqrt(10),delta_empirical_mse=de.mean(),empirical_se=de.std(ddof=1)/np.sqrt(10),nll=np.mean(nn),old_nll=np.mean(on)))
    for name,rows in [('lu_crsp',lu),('lu_heston',syn),('coordinate_filter',coord),('coordinate_within_garch',within),('coordinate_vs_portfolio',against)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
    for panel in ['crsp5','crsp100']:
        rows=[r for r in lu if r['panel']==panel and r['alpha']==.01]
        (O/f'{panel}_lu.tex').write_text(tex_table('Dense LU output controls, '+panel.replace('crsp','')+' dimensions, seed zero and ten matched folds. Differences compare LU against the original diagonal head under the same checkpoint rule. Six contrasts receive Holm adjustment within this panel and tail level. NLL uses exact fitted densities; inference concerns daily FZ0, not NLL.','tab:lu-'+panel,['Model','Rule','NLL diag./LU','FZ0 LU','$\\Delta$ FZ0','95\\% interval','Holm $p$'],[[r['model'],RULE[r['rule']],f'{r["old_nll"]:.2f} / {r["nll"]:.2f}',fmt(r['fz0']),fmt(r['delta']),interval(r),pformat(r['p_holm'])] for r in rows]))
        rows=[r for r in coord if r['panel']==panel and r['alpha']==.01];w={(r['panel'],r['model']):r for r in within if r['alpha']==.01}
        (O/f'{panel}_coordinate.tex').write_text(tex_table('Coordinate-filter refits, '+panel.replace('crsp','')+' dimensions, ten folds and 2,514 dates. GARCH is fitted separately to each coordinate; EWMA estimates each coordinate decay. $\\Delta$ compares the refitted model against its original fixed-decay EWMA version (seed zero). The seven filter/refit contrasts receive a separate Holm correction. Location is also estimated: these are full conditioning/refitting changes, not a variance-only decomposition.','tab:coord-'+panel,['Filter','Innovation','FZ0','Hits \\%','$\\Delta$ original','95\\% interval','Holm $p$'],[[r['filter'].upper(),r['model'],fmt(r['fz0']),fmt(r['hits_percent'],2),fmt(r['delta']),interval(r),pformat(r['p_holm'])] for r in rows]))
    (O/'heston_lu.tex').write_text(tex_table('Heston dense LU controls over ten matched datasets at one percent. Paired MSE differences are LU minus the original diagonal output; SE is across datasets. Risk selection is unchanged. Neither the dense layer nor likelihood selection is assumed to improve risk.','tab:lu-heston',['Model','Rule','NLL diag./LU','RMSE diag./LU','$\\Delta$ MSE','SE'],[[r['model'],RULE[r['rule']],f'{r["old_nll"]:.2f} / {r["nll"]:.2f}',f'{r["old_rmse"]:.3f} / {r["rmse"]:.3f}',fmt(r['delta_mse']),fmt(r['se'])] for r in syn if r['alpha']==.01]))
    (O/'within_garch.tex').write_text(tex_table('Innovation-law contrasts under the common coordinate-GARCH filter at one percent. Differences are relative to coordinate-GARCH-FHS, with five-contrast within-panel Holm correction.','tab:coord-within',['Panel','Innovation','$\\Delta$ GARCH-FHS','95\\% interval','Holm $p$'],[[r['panel'].replace('crsp',''),r['model'],fmt(r['delta']),interval(r),pformat(r['p_holm'])] for r in within if r['alpha']==.01]))
    (O/'coordinate_portfolio.tex').write_text(tex_table('Coordinate versus portfolio conditioning at one percent. Each new coordinate-filter/model combination is compared with portfolio-GARCH-FHS; seven contrasts are Holm-adjusted within each panel. This comparison changes both the filtering level and residual law except in the FHS rows.','tab:coord-portfolio',['Panel','Filter','Innovation','$\\Delta$ port. GARCH-FHS','95\\% interval','Holm $p$'],[[r['panel'].replace('crsp',''),r['filter'].upper(),r['model'],fmt(r['delta']),interval(r),pformat(r['p_holm'])] for r in against if r['alpha']==.01]))
    summaries=[]
    for panel in ['crsp5','crsp100']:
        values=[p for year in range(2016,2026) for p in read(ROOT/f'referee_round3/filters/{panel}_{year}.json')['parameters']]
        lam=np.array([p['ewma_lambda'] for p in values]);persist=np.array([p['arch']+p['beta'] for p in values])
        summaries.append(dict(panel=panel,series_fits=len(values),lambda_min=lam.min(),lambda_median=np.median(lam),lambda_max=lam.max(),persistence_min=persist.min(),persistence_median=np.median(persist),persistence_max=persist.max()))
    pd.DataFrame(summaries).to_csv(O/'filter_parameters_summary.csv',index=False)
    (O/'filter_parameters.tex').write_text(tex_table('Training-only coordinate filter estimates over ten folds. Minima, medians and maxima pool coordinate/fold fits; this does not treat them as independent inferential replications. All estimates and optimizer diagnostics are exported.','tab:coord-parameters',['Panel','Fits per filter','EWMA $\\lambda$ min/med/max','GARCH persistence min/med/max'],[[r['panel'].replace('crsp',''),r['series_fits'],f'{r["lambda_min"]:.3f} / {r["lambda_median"]:.3f} / {r["lambda_max"]:.3f}',f'{r["persistence_min"]:.3f} / {r["persistence_median"]:.3f} / {r["persistence_max"]:.6f}'] for r in summaries],'lrrr'))
    print(pd.DataFrame(lu).query('alpha==.01').to_string(index=False));print(pd.DataFrame(coord).query('alpha==.01').to_string(index=False));print(pd.DataFrame(syn).query('alpha==.01').to_string(index=False))
if __name__=='__main__':main()

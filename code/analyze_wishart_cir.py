"""Paired estimation-dataset errors and separately labelled numerical diagnostics."""
import json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from expanded_experiment import score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wishart_cir_extension';A=OUT/'analysis'
def dump(name,obj):(A/name).write_text(json.dumps(obj,indent=2))
def influence(loc,sd,r,alpha):
    z=(r['var']-loc)/sd
    return r['var']-r['es']+((loc-r['var'])*ndtr(-z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi))/alpha
def synthetic():
    files=sorted((OUT/'results').glob('heston_*.json'))
    if len(files)!=10:return
    rows=[];grids=[];abls=[];pars=[]
    for path in files:
        r=json.loads(path.read_text());unit=r['unit'];old=json.loads((ROOT/'wishart_completion/primary'/f'heston_{unit}.json').read_text())
        z=np.load(path.with_suffix('.npz'));fit=json.loads((OUT/'fits'/f'heston_{unit}.json').read_text())
        pars.append(dict(unit=unit,**{k:v for k,v in r['parameters'].items() if k not in ['C','mu']},observed_kurtosis=fit['fit']['observed_moments'][0],fitted_kurtosis=fit['fit']['simulated_moments'][0],observed_cross_square=fit['fit']['observed_moments'][-1],fitted_cross_square=fit['fit']['simulated_moments'][-1]))
        for alpha in [.01,.001]:
            subset={ (q['steps'],q['ablation']):q for q in r['rows'] if q['alpha']==alpha }
            full=subset[32,'full'];emp=full['empirical'];target=full['truth']
            for name,val,mc in [('Empirical',emp,0),('Restricted Wishart',next(m['es'] for m in old['metrics'] if m['alpha']==alpha),None)]+[(a,subset[32,a]['es'],subset[32,a]['mc_se']) for a in ['full','no_leverage','fixed_component_correlation']]:
                rows.append(dict(unit=unit,alpha=alpha,model=name,es=val,truth=target,error=val-target,fit_error=val-emp,sampling_error=emp-target,mc_se=mc))
            psi=influence(z['32_full_loc'],z['32_full_sd'],full,alpha)
            for s in [4,8,16]:
                q=subset[s,'full'];d=psi-influence(z[f'{s}_full_loc'],z[f'{s}_full_sd'],q,alpha)
                grids.append(dict(unit=unit,alpha=alpha,coarse=s,delta=full['es']-q['es'],paired_mc_se=float(d.std(ddof=1)/np.sqrt(len(d)))))
            for a in ['no_leverage','fixed_component_correlation']:
                q=subset[32,a];d=psi-influence(z[f'32_{a}_loc'],z[f'32_{a}_sd'],q,alpha)
                abls.append(dict(unit=unit,alpha=alpha,ablation=a,full_minus_ablation=full['es']-q['es'],paired_mc_se=float(d.std(ddof=1)/np.sqrt(len(d)))))
    f=pd.DataFrame(rows);f.to_csv(A/'synthetic_errors.csv',index=False)
    pd.DataFrame(pars).to_csv(A/'fitted_parameters.csv',index=False);pd.DataFrame(grids).to_csv(A/'grid_differences.csv',index=False);pd.DataFrame(abls).to_csv(A/'ablation_differences.csv',index=False)
    summary=[]
    for (alpha,model),g in f.groupby(['alpha','model']):
        emp=f[(f.alpha==alpha)&(f.model=='Empirical')].set_index('unit').loc[g.unit]
        de=g.error.to_numpy()**2-emp.error.to_numpy()**2
        summary.append(dict(alpha=alpha,model=model,rmse=float(np.sqrt(np.mean(g.error**2))),mean_error=float(g.error.mean()),delta_mse=float(np.mean(de)),delta_mse_se=float(np.std(de,ddof=1)/np.sqrt(10)),fit_mse=float(np.mean(g.fit_error**2)),sampling_mse=float(np.mean(g.sampling_error**2)),cross_term=float(2*np.mean(g.fit_error*g.sampling_error))))
    dump('synthetic_summary.json',summary)
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{1\% weekly ES} & \multicolumn{2}{c}{0.1\% weekly ES} \\',r'Model/intervention & RMSE & $\Delta$MSE (SE) & RMSE & $\Delta$MSE (SE) \\',r'\midrule']
    labels={'Empirical':'Empirical','Restricted Wishart':'Restricted Wishart','full':'Wishart--CIR','no_leverage':'Same fit, no scalar leverage','fixed_component_correlation':'Same fit, fixed component correlation'}
    for name in labels:
        cells=[]
        for alpha in [.01,.001]:
            s=next(s for s in summary if s['alpha']==alpha and s['model']==name);cells += [f"{s['rmse']:.3f}",f"{s['delta_mse']:+.3f} ({s['delta_mse_se']:.3f})"]
        lines.append(labels[name]+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Retrospective Wishart--CIR extension on the same ten Heston estimation datasets. $\Delta$MSE is paired against the empirical estimator; SE reflects variation across the ten datasets, conditional on the shared reference sample. The two interventions reuse the fitted parameters and paths and are not separately selected models. All stochastic-covariance rows use 32 substeps and 131,072 conditional Gaussian paths per fit.}\label{tab:wc-synthetic}',r'\end{table}']
    (A/'synthetic_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))
def filters():
    configs={};table=[]
    for path in sorted((OUT/'results').glob('crsp5_*.json')):
        r=json.loads(path.read_text());f=pd.read_csv(path.with_suffix('.csv.gz'));f=f[(f.portfolio=='equal')&(f.alpha==.01)].sort_values('date').reset_index(drop=True)
        key=(r['year'],r['particles'],r['steps'],r['predictive'],r['warmup'],r.get('guided',False));configs.setdefault(key,{})[r['bank']]=f
        diag=pd.read_csv(path.parent/(path.stem+'_filter.csv.gz'));diag=diag[diag.test]
        table.append(dict(**{k:r[k] for k in ['year','particles','steps','bank','predictive','warmup']},guided=r.get('guided',False),score=float(f.fz0.mean()),median_ess=float(diag.ess.median()),min_ess=float(diag.ess.min()),fraction_ess_under10=float(np.mean(diag.ess<10))))
    pd.DataFrame(table).to_csv(A/'filter_runs.csv',index=False)
    ensembles={};seeds=[]
    for key,banks in configs.items():
        if not all(b in banks for b in [0,1]):continue
        x,y=banks[0],banks[1];assert x.date.equals(y.date)
        e=x.copy();e['es']=(x.es+y.es)/2;e['var']=(x['var']+y['var'])/2;e['fz0']=score(e.loss,e['var'],e.es,.01);ensembles[key]=e
        seeds.append(dict(config=key,es_relative_rms=float(np.sqrt(np.mean(((x.es-y.es)/e.es)**2))),bank_scores=[float(x.fz0.mean()),float(y.fz0.mean())],ensemble_score=float(e.fz0.mean()),bank_score_se=float(abs(x.fz0.mean()-y.fz0.mean())/2)))
    comparisons=[]
    for k,l in itertools.combinations(sorted(ensembles),2):
        changed=[j for j in range(1,6) if k[j]!=l[j]]
        if k[0]!=l[0] or len(changed)!=1:continue
        x,y=ensembles[k],ensembles[l];assert x.date.equals(y.date)
        rms=float(np.sqrt(np.mean(((x.es-y.es)/y.es)**2)));ds=float(y.fz0.mean()-x.fz0.mean())
        comparisons.append(dict(coarse=k,fine=l,changed=['year','particles','steps','predictive','warmup','proposal'][changed[0]],es_relative_rms=rms,score_change=ds,passes=bool(rms<=.02 and abs(ds)<=.02)))
    dump('filter_audit.json',dict(seed_comparisons=seeds,refinements=comparisons))
    print(json.dumps(dict(seed_comparisons=seeds,refinements=comparisons),indent=2))
if __name__=='__main__':synthetic();filters()

"""Contract checks on completed controls, independently reconstructed scores."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'referee_round3/results';A=ROOT/'referee_round3/analysis'
def main():
    records=[];neural=[];files=set();checks={};series=0;pars=[]
    for study in ['heston','crsp5','crsp100']:
        units=range(10) if study=='heston' else range(2016,2026)
        for unit in units:
            for name in ['Flow-G','Flow-t']:
                x=json.loads((R/f'lu_{study}_{unit}_{name}.json').read_text());records.append(x);neural.append(x)
                assert set(x['evaluations'])=={'fz1','density','final'}
    for panel in ['crsp5','crsp100']:
        for year in range(2016,2026):
            z=json.loads((ROOT/f'referee_round3/filters/{panel}_{year}.json').read_text());series+=len(z['parameters']);pars+=z['parameters'];assert z['train_end']<f'{year-1}-01-01'
            for name in ['Flow-G','Flow-t','Flow-meta-t']:
                x=json.loads((R/f'coord_{panel}_{year}_{name}.json').read_text());records.append(x);neural.append(x)
            for kind,name in [('garch','FHS'),('garch','Gaussian'),('garch','Meta-t'),('ewma','FHS')]:
                x=json.loads((R/f'coord_{kind}_{panel}_{year}_{name}.json').read_text());records.append(x);files.add(x['forecasts'])
    checks['two_hundred_model_fold_records']=len(records)==200
    checks['one_hundred_twenty_neural_runs']=len(neural)==120
    checks['1050_fits_per_scalar_filter']=series==1050
    checks['garch_parameters_admissible']=all(p['omega']>0 and p['arch']>0 and p['beta']>0 and p['arch']+p['beta']<.9999 and 2.05<=p['nu']<=200 for p in pars)
    checks['ewma_training_bounds']=all(.8<=p['ewma_lambda']<=.9995 for p in pars)
    for x in neural:
        assert len(x['trace'])==12 and x['choices']['final']['step']==3000
        for key in ['fz1','density']:
            best=min(x['trace'],key=lambda t:t[key]);assert best['step']==x['choices'][key]['step']
        for ev in x['evaluations'].values():
            assert np.isfinite(ev['nll'])
            if 'forecasts' in ev:files.add(ev['forecasts'])
            for m in ev.get('metrics',[]):assert abs(m['total_error']-m['fit_error']-m['sample_error'])<1e-10
    checks['validation_selection_and_final_grid']=True;checks['finite_nll_and_error_identities']=True
    error=0.;rows=0
    for name in files:
        x=pd.read_csv(R/name);rows+=len(x);assert x[['date','portfolio','alpha']].duplicated().sum()==0
        assert np.isfinite(x[['loss','var','es','fz0']]).all().all() and (x.es>0).all()
        expected=np.log(x.es)+x['var']/x.es-1+np.maximum(x.loss-x['var'],0)/(x.alpha*x.es)
        error=max(error,float(np.max(abs(expected-x.fz0))))
        assert np.array_equal(x.hit.astype(bool),x.loss>x['var'])
        assert np.max(abs(x.es_residual-(x.es-x['var']-np.maximum(x.loss-x['var'],0)/x.alpha)))<1e-9
    checks['independent_fz0_reconstruction']=error<1e-9;checks['hit_and_es_moment_reconstruction']=True
    primary=pd.read_csv(A/'primary_summary.csv');checks['primary_has_no_wishart']=not primary.model.str.contains('Wishart').any()
    checks['primary_two_panels_fifteen_models']=all(len(g)==15 for _,g in primary.groupby(['panel','alpha']))
    checks['symmetric_primary_nonrejection']=not (primary[primary.alpha==.01].p_holm<.05).any()
    for name,n in [('lu_crsp',24),('lu_heston',12),('coordinate_filter',28),('coordinate_within_garch',20),('coordinate_vs_portfolio',28)]:
        x=pd.read_csv(A/(name+'.csv'));checks[name+'_row_count']=len(x)==n
    assert all(checks.values()),checks
    out=dict(checks=checks,passed=sum(checks.values()),model_fold_records=len(records),neural_runs=len(neural),scalar_garch_fits=series,scalar_ewma_fits=series,unique_forecast_files=len(files),forecast_rows=rows,max_score_error=error)
    (A/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

"""Release contracts for replay, forecasts, inference, and added estimators."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from analyze import holm
from round4_analysis import ROOT,OUT,O

def main():
    records=[json.loads(p.read_text()) for p in (OUT/'results').glob('*.json') if '_step' not in p.name]
    neural=[z for z in records if 'evaluations' in z];param=[z for z in records if z.get('model') in ['Student-MLE','Gaussian-LW']]
    assert len(neural)==640,len(neural);assert len(param)==160,len(param)
    checks={};checks['repaired_neural_trajectories']=len(neural);checks['added_joint_model_fits']=len(param)
    values=[];parameter_errors=[];changed={}
    for z in neural:
        for rule,c in z['choices'].items():
            if rule!='final':assert c['value']<=z['old_choices'][rule]['value']+1e-12
            else:assert c['step']==z['old_choices'][rule]['step'] and not c['early_added']
            changed[rule]=changed.get(rule,0)+int(c['early_added'])
            if c['early_added']:assert c['step']<250
            assert (ROOT/c['checkpoint']).exists()
        values+=list(z['replay_criterion_errors'].values())
        if z['replay_parameter_error'] is not None:parameter_errors.append(z['replay_parameter_error'])
    checks['max_replay_criterion_error']=max(values);checks['max_replay_parameter_error']=max(parameter_errors);checks['parameter_comparisons']=len(parameter_errors);checks['changed_choices']=changed
    assert checks['max_replay_parameter_error']<2e-5
    count=0;maximum=0.
    files=set()
    for z in records:
        for e in z.get('evaluations',{'self':z}).values():
            if 'forecasts' in e:files.add(e['forecasts'])
            if e.get('pointwise'):
                a=np.load(ROOT/e['pointwise'])['nll'];assert np.isfinite(a).all();assert abs(a.mean()-e['nll'])<1e-6
    for name in files:
        f=pd.read_csv(ROOT/name);assert np.isfinite(f[['loss','var','es','fz0']].values).all();assert (f.es>0).all();assert (f.es>=f['var']-1e-8).all()
        score=np.log(f.es)+f['var']/f.es-1+np.maximum(f.loss-f['var'],0)/(f.alpha*f.es)
        maximum=max(maximum,float(np.max(abs(score-f.fz0))));count+=len(f)
    checks['forecast_files_checked']=len(files);checks['forecast_rows_checked']=count;checks['maximum_score_reconstruction_error']=maximum;assert maximum<1e-9
    p=pd.read_csv(O/'primary_summary.csv');assert len(p)==68
    for (panel,alpha),g in p.groupby(['panel','alpha']):
        assert len(g)==17;v=g[g.model!='Historical'];assert np.max(abs(holm(v.p.to_numpy())-v.p_holm.to_numpy()))<1e-12
        for row in v.itertuples():
            if row.low>0 or row.high<0:assert row.p<.051
    for panel in ['crsp5','crsp100']:
        f=pd.read_csv(O/f'{panel}_forecasts.csv.gz');dates=pd.to_datetime(f.date);assert dates.nunique()==2514
        assert len(f.model.unique())==17
        for (name,seed,pn,a),g in f.groupby(['model','seed','portfolio','alpha']):assert len(g)==2514,(panel,name,seed,pn,a,len(g))
        x=pd.read_csv(O/f'{panel}_pointwise_nll.csv.gz');assert x.shape==(2514,9)
    d=pd.read_csv(O/'decomposition.csv');assert np.max(abs(d.total_mse-d.fit_mse-d.sampling_mse-d['cross']))<1e-8
    checks['error_decomposition_max_error']=float(np.max(abs(d.total_mse-d.fit_mse-d.sampling_mse-d['cross'])))
    student=[]
    for z in param:
        fit=json.loads((ROOT/z['checkpoint']).read_text())
        if z['model']=='Student-MLE':
            assert 2.05<=fit['nu']<=200.;assert np.min(np.diff(fit['trace']))>=-1e-7
            assert max(q['loglik'] for q in fit['diagnostics'] if q['converged'])>=max(q['loglik'] for q in fit['diagnostics'])-1e-8
            np.linalg.cholesky(fit['scatter']);student.append(fit['nu'])
    checks['student_fits']=len(student);checks['student_df_range']=[min(student),max(student)]
    j=pd.read_csv(O/'joint_summary.csv');assert len(j)==24;assert j.brier.between(0,1).all()
    for panel,g in j.groupby('panel'):
        a=g[g.model!='Historical'];assert np.max(abs(holm(a.p.to_numpy())-a.p_holm.to_numpy()))<1e-12
    s=pd.read_csv(O/'selection_summary.csv');assert len(s)==80
    a=pd.read_csv(O/'augmented28.csv');assert len(a)==56
    checks['all_release_contracts_passed']=True
    (O/'verification.json').write_text(json.dumps(checks,indent=2))
    text=(r'\paragraph{Executed repair and reproducibility.}'+'\n'+f'The current archive retains 1,110 model/fold records, including ten unresolved Wishart CRSP pilot records. The repair replays prefixes of all {len(neural)} retained neural trajectories and adds {len(param)} joint parametric fits; prefix replay is not counted as independent model estimation. The existing 2,100 scalar coordinate-filter fits are separate submodel estimations. Checkpoint alternatives reuse the same fitted trajectory.\n\n'+f'The release checks reconstruct {count:,} forecast rows across {len(files)} unique files, verify common dates and candidate selection, reproduce the Holm arithmetic, and check the paired error identity. Maximum FZ0 reconstruction error is {maximum:.2g}. Every eligible repaired selection is at least as good on its own recorded validation criterion as the old admitted candidate set. Parameter and criterion replay tolerances, added Student optimization diagnostics, and the independent projection/dense-nesting checks are recorded in the numerical audit. These contracts verify implementation consistency, not statistical superiority.\n\n'+r'The source package supplies pointwise likelihoods, all forecast records, checkpoints, protocols and machine-readable sensitivity tables. Raw licensed CRSP extracts are not redistributed. Rebuilding from derived panels requires the main source archive, reference-sample archive, round-three checkpoint archive, and round-four results/checkpoint archive. The Zenodo record contains the manuscript PDF; the separate source archives are distributed by the author.'+'\n')
    (O/'verification.tex').write_text(text);print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

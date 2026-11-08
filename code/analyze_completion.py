"""Close the numerical eligibility audit without changing its acceptance gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'completion_audit';A=OUT/'analysis'


def main():
    rows=[];series={};forecast_rows=0;score_error=0.
    source=pd.read_csv(ROOT/'data/crsp_size_portfolios.csv',index_col=0,parse_dates=True)*100
    parameters=json.loads((ROOT/'wishart_cir_extension/fits/crsp5_2025.json').read_text())['parameters']
    for n in [8192,16384]:
        fs=[];ds=[]
        for b in [0,1]:
            tag=f'move_2025_n{n}_s16_b{b}_p32768_w504_m3'
            meta=json.loads((OUT/'results'/(tag+'.json')).read_text())
            assert meta['parameters']==parameters
            allf=pd.read_csv(OUT/'results'/(tag+'.csv.gz'))
            assert len(allf)==1500 and not allf.duplicated(['date','portfolio','alpha']).any()
            assert np.isfinite(allf[['loss','var','es','fz0']]).all().all() and (allf.es>=allf['var']).all() and (allf.es>0).all()
            reconstructed=np.log(allf.es)+allf['var']/allf.es-1+np.maximum(allf.loss-allf['var'],0)/(allf.alpha*allf.es)
            score_error=max(score_error,float(np.max(abs(reconstructed-allf.fz0))))
            forecast_rows+=len(allf)
            for portfolio,weights in {'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}.items():
                part=allf[allf.portfolio==portfolio]
                expected=-source.loc[pd.to_datetime(part.date)].to_numpy()@weights
                assert np.max(abs(expected-part.loss))<1e-10
            f=allf.query("portfolio=='equal' and alpha==0.01").set_index('date')
            d=pd.read_csv(OUT/'results'/(tag+'_filter.csv.gz')).query('test').set_index('date')
            assert len(f)==250 and len(d)==250 and meta['n_test']==250
            fs.append(f);ds.append(d)
        assert fs[0].index.equals(fs[1].index)
        rel=(fs[1].es-fs[0].es)/((fs[1].es+fs[0].es)/2)
        delta=fs[1].fz0-fs[0].fz0
        rms=float(np.sqrt(np.mean(rel**2)));sc=float(delta.mean())
        # Diagnostics use absolute contributions; no score-driven path removal.
        top=delta.abs().nlargest(3)
        row=dict(particles=n,steps=16,predictive=32768,warmup=504,moves=3,n_test=250,
            es_relative_rms=rms,score_difference=sc,passes=bool(rms<=.02 and abs(sc)<=.02),
            bank_scores=[float(f.fz0.mean()) for f in fs],
            mean_move_acceptance=[float(d.move_acceptance.mean()) for d in ds],
            minimum_ess=[float(d.ess.min()) for d in ds],
            maximum_absolute_relative_es=float(rel.abs().max()),
            top3_absolute_score_share=float(top.sum()/delta.abs().sum()),
            top3_score_dates=top.index.tolist())
        rows.append(row)
        series[n]=fs
    # Coarse/fine ensemble forecasts are averaged BEFORE scoring.
    from expanded_experiment import score
    e=[]
    for n in [8192,16384]:
        f0,f1=series[n];q=(f0['var']+f1['var'])/2;es=(f0.es+f1.es)/2
        e.append((es,score(f0.loss.to_numpy(),q.to_numpy(),es.to_numpy(),.01)))
    rms=float(np.sqrt(np.mean(((e[1][0]-e[0][0])/((e[1][0]+e[0][0])/2))**2)))
    refinement=dict(es_relative_rms=rms,score_difference=float(np.mean(e[1][1]-e[0][1])))
    refinement['passes']=bool(rms<=.02 and abs(refinement['score_difference'])<=.02)
    passed=rows[-1]['passes'] and refinement['passes']
    status=dict(completed_additional_runs=4,total_wishart_cir_filter_runs=30,
        forecast_rows_checked=forecast_rows,score_reconstruction_max_error=score_error,
        independent_banks=rows,particle_refinement=refinement,
        stress_gate_passed=passed,full_panel_completed=False,
        conditional_ranking_eligible=False,
        disposition='Further full-panel validation required' if passed else 'Excluded from conditional ranking: numerical gates failed',
        parameters_unchanged=True,primary_forecasts_and_families_unchanged=True)
    (A/'numerical_decision.json').write_text(json.dumps(status,indent=2))
    pd.DataFrame(rows).to_csv(A/'move_summary.csv',index=False)
    # A factual appendix paragraph; final editorial processor consumes the status.
    f=rows[-1]
    text=(f"The final resample-move audit contains four complete 2025 runs: two independent banks at each of 8,192 and 16,384 particles, all with 16 substeps, 32,768 predictive paths and 504 warmup days. At 16,384 particles the independent-bank relative-ES RMS difference is {100*f['es_relative_rms']:.2f}\\% and the annual FZ0 difference is {f['score_difference']:+.4f}; the respective acceptance tolerances remain 2\\% and 0.02. Mean daily Metropolis acceptance is {100*f['mean_move_acceptance'][0]:.1f}\\% and {100*f['mean_move_acceptance'][1]:.1f}\\% in the two banks. The two-bank particle refinement gives {100*rms:.2f}\\% ES RMS and {refinement['score_difference']:+.4f} FZ0. "
       +f"Three dates account for {100*f['top3_absolute_score_share']:.1f}\\% of the sum of absolute daily score differences between the finest banks; the largest daily relative-ES discrepancy is {100*f['maximum_absolute_relative_es']:.1f}\\%. Annual mean-score agreement and pathwise forecast stability must therefore be checked separately. "
       +("These checks pass, but do not replace a full-panel grid, warmup and predictive-resolution audit." if passed else "The numerical eligibility gate fails. Successful local moves therefore do not establish accuracy of the long sequential posterior approximation. We close this audit with exclusion from conditional ranking, retaining the completed synthetic specification study and all failed numerical runs.")+'\n')
    (A/'filter_findings.tex').write_text(text)
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()

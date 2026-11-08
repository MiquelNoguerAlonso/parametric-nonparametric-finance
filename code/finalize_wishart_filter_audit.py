"""Close the declared stress-period refinement matrix without promoting failures."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from analyze_wishart_completion import records,load,ensemble,summarize
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'wishart_completion';A=W/'analysis'

def main():
 summarize();r=records();r=r[(r.year==2020)&(r.n_forecast_dates==253)]
 configs=[(1024,8),(2048,8),(1024,16)];selected={}
 for n,s in configs:
  g=r[(r.particles==n)&(r.substeps==s)&(r.lag==2)&(r.moves==3)&r.moment_auxiliary&r.gauge_paths&(r.aux_power==.25)&r.antithetic&(~r.canonical)&(r.defensive==0)&(r.warmup==252)&(r.predictive==8192)&r.bank.isin([0,1])]
  if len(g)!=2 or set(g.bank)!={0,1}:raise RuntimeError(f'Incomplete final refinement matrix: N={n}, steps={s}')
  assert (g.n_forecast_dates==253).all()
  selected[n,s]=ensemble(load(g))
 pairs=[]
 for label,ca,cb in [('Particles',(1024,8),(2048,8)),('Time grid',(1024,8),(1024,16))]:
  p=selected[ca].merge(selected[cb],on=['date','fold','portfolio','alpha'],suffixes=('_coarse','_fine'),validate='one_to_one')
  for (pn,a),q in p.groupby(['portfolio','alpha']):
   ds=float((q.fz0_fine-q.fz0_coarse).mean());er=float(np.sqrt(np.mean(((q.es_fine-q.es_coarse)/q.es_fine)**2)))
   pairs.append(dict(refinement=label,portfolio=pn,alpha=a,dates=len(q),coarse_particles=ca[0],coarse_steps=ca[1],fine_particles=cb[0],fine_steps=cb[1],delta_fz0=ds,es_relative_rms=er,annual_score_tolerance=.02,es_relative_rms_tolerance=.02,passes_score=abs(ds)<=.02,passes_es=er<=.02))
 pairs=pd.DataFrame(pairs);pairs.to_csv(A/'final_filter_refinements.csv',index=False)
 primary=pairs[(pairs.portfolio=='equal')&(pairs.alpha==.01)];passes=bool((primary.passes_score&primary.passes_es).all());allpasses=bool((pairs.passes_score&pairs.passes_es).all())
 # Report every completed 2020 configuration. One-bank entries receive no seed-spread estimate.
 configs_fields=['particles','substeps','predictive','warmup','moves','antithetic','defensive','lag','moment_auxiliary','aux_power','canonical','gauge_paths']
 rows=[]
 for cfg,g in r.groupby(configs_fields):
  meta=dict(zip(configs_fields,cfg));f=load(g);raw=f[(f.portfolio=='equal')&(f.alpha==.01)];en=ensemble(raw);bs=raw.groupby('seed').fz0.mean()
  method='Frozen'
  if meta['defensive']:method='Defensive'
  elif meta['moment_auxiliary']:method='Moment'
  if meta['aux_power']!=1:method='Flattened moment'
  if meta['canonical']:method='Canonical states'
  if meta['gauge_paths']:method='Canonical paths'
  if meta['lag']>1:method='Block '+method.lower()
  if meta['antithetic'] and method=='Frozen':method='Frozen, antithetic'
  q=dict(meta,method=method,banks=len(bs),ensemble_fz0=float(en.fz0.mean()),minimum_bank_score=float(bs.min()),maximum_bank_score=float(bs.max()),bank_score_se=float(bs.std()/np.sqrt(len(bs))) if len(bs)>1 else None)
  rows.append(q)
 full=pd.DataFrame(rows);full.to_csv(A/'final_filter_configurations.csv',index=False)
 lines=[r'\begin{table}[htbp]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Proposal / update & $N$ & Steps & Block & Moves & Banks & Ensemble FZ0 & Bank range \\',r'\midrule']
 for q in rows:
  rang=f"{q['minimum_bank_score']:.3f}--{q['maximum_bank_score']:.3f}" if q['banks']>1 else '--'
  lines.append(f"{q['method']} & {q['particles']} & {q['substeps']} & {q['lag']} & {q['moves']} & {q['banks']} & {q['ensemble_fz0']:.3f} & {rang}"+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Wishart numerical stress audit: all completed 2020 configurations, 253 dates, equal portfolio and one-percent loss forecasts. Parameters are fixed. Each bank has 252 prior warmup observations and 8,192 predictive paths. Ensemble FZ0 scores the averaged VaR and ES forecasts; the last column is the range of separately scored banks, not a confidence interval. Moves counts pCN proposals per temperature; each stage also refreshes ancestors. The full diagnostics retain all portfolios and both levels. These are numerical experiments, not model-ranking tests.}\label{tab:wishart-filter-audit}',r'\end{table}']
 (A/'filter_audit.tex').write_text('\n'.join(lines)+'\n')
 p=primary.set_index('refinement');a=p.loc['Particles'];b=p.loc['Time grid']
 text=r'\paragraph{CRSP convergence decision.}'+'\n'
 text+='The earlier ten-fold guided pilot had FZ0 1.520 against FHS 1.370 on 2,514 dates, but its 2025 refinement shifted FZ0 by $-0.097$ and up to 16.6\\% of dates had correction ESS below ten. It remains an archived numerical pilot. The new stress audit uses 253 dates in 2020 and preserves fitted parameters. Normalizer, orthogonal-invariance, Gaussian-risk and forecast-timing checks pass; they validate the kernels, not finite-particle accuracy.\n\n'
 text+='The numerical protocol required annual FZ0 changes no greater than 0.02 and relative daily ES RMS changes no greater than 2\\%, alongside full-panel and seed-error checks. In the final two-day block method, ancestor probabilities use the one-quarter power of the moment-auxiliary likelihood and factors are canonicalized at every integration step. '
 text+=f"Doubling particles from 1,024 to 2,048 at eight steps changes ensemble one-percent FZ0 by {a.delta_fz0:+.4f} and daily ES by {100*a.es_relative_rms:.2f}\\% in relative RMS. Doubling time resolution from eight to sixteen steps at 1,024 particles changes FZ0 by {b.delta_fz0:+.4f} and daily ES by {100*b.es_relative_rms:.2f}\\%. "
 if not passes:text+='The primary stress-period criteria are not met. The CRSP Wishart implementation therefore remains excluded from primary inference; high intermediate ESS cannot replace these convergence checks. No comparative CRSP conclusion about the Wishart model family is drawn.\n'
 else:text+='The primary stress-period criteria are met. Full-panel, longer-history and predictive-path checks remain necessary before adding a CRSP row to primary inference.\n'
 (A/'filter_findings.tex').write_text(text)
 status=dict(audit_complete=True,synthetic_evaluation_finalized=True,synthetic_fits=10,synthetic_paths_per_fit=131072,synthetic_steps=32,crsp_primary_stress_refinement_gate_passed=passes,crsp_all_portfolio_level_refinement_gate_passed=allpasses,crsp_full_panel_validated=False,crsp_accepted_for_primary_inference=False,reason=('The primary stress-period numerical tolerances failed.' if not passes else 'Stress-period checks passed; full-panel acceptance has not been established.'),new_crsp_parameter_fits=0,completed_2020_pilot_runs=len(r),new_stress_dates=253,refinements=primary.to_dict('records'))
 (A/'final_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if __name__=='__main__':
    if (W/'early_stop/decision.json').exists():
        from finalize_wishart_early_audit import main as close_early
        close_early()
    else:main()

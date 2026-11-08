"""Finish the numerical audit after an irreversible tolerance failure."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from analyze_wishart_completion import records,ensemble,summarize
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'wishart_completion';A=W/'analysis';E=W/'early_stop'

def main():
 decision=json.loads((E/'decision.json').read_text());D={}
 for n,s in [(1024,8),(2048,8),(1024,16)]:
  frames=[]
  for b in [0,1]:
   tag=f'y2020_b{b}_n{n}_s{s}_p8192_w252_m3_anti_lag2_moment_ap25_pathgauge';rec=next(v for v in decision['inputs'] if v['tag']==tag);p=ROOT/rec['snapshot'];assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];x=json.loads(p.read_text());frames.append(pd.DataFrame(x['rows']))
  common=set(frames[0].date)&set(frames[1].date);D[n,s]=ensemble(pd.concat([f[f.date.isin(common)] for f in frames]))
 rows=[]
 for label,cfg in [('Particles',(2048,8)),('Time grid',(1024,16))]:
  p=D[1024,8].merge(D[cfg],on=['date','fold','portfolio','alpha'],suffixes=('_coarse','_fine'),validate='one_to_one')
  for (pn,alpha),f in p.groupby(['portfolio','alpha']):
   sq=((f.es_fine-f.es_coarse)/f.es_fine)**2;bound=float(np.sqrt(sq.sum()/253));rows.append(dict(refinement=label,portfolio=pn,alpha=alpha,dates=len(f),first_date=f.date.min(),last_date=f.date.max(),prefix_es_relative_rms=float(np.sqrt(sq.mean())),full_year_es_rms_lower_bound=bound,tolerance=.02,decisively_fails=bool(bound>.02)))
 rows=pd.DataFrame(rows);rows.to_csv(A/'final_filter_refinements.csv',index=False);primary=rows[(rows.portfolio=='equal')&(rows.alpha==.01)].set_index('refinement');a=primary.loc['Particles'];b=primary.loc['Time grid'];assert a.decisively_fails
 summarize();r=records();complete=r[(r.year==2020)&(r.n_forecast_dates==253)];full=r.n_forecast_dates.sum()
 lines=[r'\begin{table}[htbp]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Refinement & Matched days & ES RMS \% & Annual lower bound \% & Decision \\',r'\midrule']
 for label,z in [('Particles: 1,024 to 2,048',a),('Steps: 8 to 16',b)]:
  verdict='Fails 2\\% gate' if z.decisively_fails else 'Incomplete'
  lines.append(f'{label} & {int(z.dates)} & {100*z.prefix_es_relative_rms:.2f} & {100*z.full_year_es_rms_lower_bound:.2f} & {verdict}'+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Final Wishart stress refinements, equal-portfolio one-percent forecasts. Particle refinement fixes eight return steps; time refinement fixes 1,024 states. Each configuration has two independent banks, 252 warmup observations and 8,192 predictive paths per bank. ES forecasts are averaged before comparison. The annual lower bound divides the observed sum of squared relative discrepancies by all 253 dates; uncomputed terms are nonnegative. The particle result already rejects the two-percent annual tolerance. Remaining refinements were stopped and are not represented as full-year scores.}\label{tab:wishart-filter-audit}',r'\end{table}']
 (A/'filter_audit.tex').write_text('\n'.join(lines)+'\n')
 text=r'\paragraph{CRSP convergence decision.}'+'\n'
 text+='The earlier ten-fold guided pilot had FZ0 1.520 against FHS 1.370 on 2,514 dates, but its 2025 refinement shifted FZ0 by $-0.097$ and up to 16.6\\% of dates had one-stage correction ESS below ten. It remains an archived numerical pilot. The new stress audit targets 2020 and preserves every fitted parameter. Analytic normalizer, orthogonal-invariance, Gaussian-risk and forecast-timing checks pass; these establish kernel consistency, not finite-particle accuracy.\n\n'
 text+='The numerical protocol requires annual FZ0 changes no greater than 0.02 and relative daily ES RMS changes no greater than 2\\%, alongside full-panel and seed-error checks. The final method uses two-day blocks, moment-auxiliary ancestor probabilities raised to the one-quarter power, three pCN moves per temperature and canonical factors at every integration step. '
 text+=f'Doubling particles from 1,024 to 2,048 at eight steps produces a {100*a.prefix_es_relative_rms:.2f}\\% relative ES RMS discrepancy over {int(a.dates)} matched dates, {a.first_date} to {a.last_date}. For nonnegative squared discrepancies $d_t^2$,\n'
 text+=r'\[\left(253^{-1}\sum_{t=1}^{253}d_t^2\right)^{1/2}\ge\left(253^{-1}\sum_{t\in\mathcal O}d_t^2\right)^{1/2},\qquad d_t=\frac{\widehat e_t^{\rm fine}-\widehat e_t^{\rm coarse}}{\widehat e_t^{\rm fine}}.\]'+'\n'
 text+=f'The right side is already {100*a.full_year_es_rms_lower_bound:.2f}\\%, above the declared 2\\% limit. Even exact agreement on every remaining date could not reverse this failure. The outstanding runs were therefore stopped; the time-grid comparison had {int(b.dates)} matched dates and does not establish convergence. The stopping rule uses the numerical tolerance, not a favorable score level.\n\n'
 text+=f'The archive retains {len(complete)} completed 2020 pilot runs, other-year pilots, and immutable snapshots of the stopped refinements. No new 2,514-date converged Wishart forecast sequence is claimed. The CRSP implementation remains excluded from primary scores, Holm families and MCS membership. These results identify unresolved numerical error in the tested filtering implementations and do not rank the Wishart model family.\n'
 (A/'filter_findings.tex').write_text(text)
 status=dict(audit_complete=True,synthetic_evaluation_finalized=True,synthetic_fits=10,synthetic_paths_per_fit=131072,synthetic_steps=32,crsp_primary_stress_refinement_gate_passed=False,crsp_full_panel_validated=False,crsp_accepted_for_primary_inference=False,final_refinement_matrix_completed=False,early_stopping_basis='Nonnegative discrepancy terms imply an annual RMS lower bound above the pre-existing tolerance.',reason=f'Observed particle-refinement discrepancies imply annual ES RMS >= {100*a.full_year_es_rms_lower_bound:.4f}%, exceeding 2%.',completed_2020_pilot_runs=len(complete),completed_other_year_pilot_runs=int((r.year!=2020).sum()),independent_estimation_datasets_added=0,new_parameter_fits=0,primary_bounds=primary.reset_index().to_dict('records'))
 (A/'final_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if __name__=='__main__':main()

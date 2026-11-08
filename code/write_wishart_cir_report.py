"""Write the numerical audit from completed runs; no provisional result is inferred."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'wishart_cir_extension/analysis'
r=json.loads((A/'filter_audit.json').read_text())
required={(2020,8192,16,16384,504,False),(2020,8192,16,16384,1008,False),(2020,16384,16,16384,504,False),(2020,8192,32,16384,252,False)}
assert required.issubset({tuple(x['config']) for x in r['seed_comparisons']})
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{lrrr}\toprule',r'2020 numerical refinement & ES RMS (\%) & $\Delta$FZ0 & Gate \\ \midrule']
label={'particles':r'$N$','steps':r'Substeps','predictive':r'Predictive paths','warmup':r'Warmup days'}
# Fixed-config one-factor refinements only; retain the finest and immediate predecessor comparisons.
for x in r['refinements']:
 c,f=x['coarse'],x['fine'];j=['year','particles','steps','predictive','warmup','proposal'].index(x['changed'])
 if c[0]!=2020 or x['changed']=='proposal':continue
 if x['changed']=='steps' and c[2]==8 and f[2]>16:continue
 if x['changed']=='warmup' and c[4]==252 and f[4]>504:continue
 if x['changed']=='particles' and c[1]<f[1]/2:continue
 lines.append(label[x['changed']]+f" {c[j]:,} to {f[j]:,} & {100*x['es_relative_rms']:.2f} & {x['score_change']:+.4f} & "+('Pass' if x['passes'] else 'Fail')+r' \\')
lines.append(r'\midrule\multicolumn{4}{l}{2025 independent banks; fixed 16 substeps and 504-day warmup} \\')
for x in sorted(r['seed_comparisons'],key=lambda z:(z['config'][0],z['config'][1],bool(z['config'][5]))):
 c=x['config']
 if c[0]!=2025:continue
 ds=x['bank_scores'][1]-x['bank_scores'][0]
 passed=x['es_relative_rms']<=.02 and abs(ds)<=.02
 name=('Guided' if c[5] else 'Prior')+f' $N={c[1]:,}$'
 lines.append(name+f" & {100*x['es_relative_rms']:.2f} & {ds:+.4f} & "+('Pass' if passed else 'Fail')+r' \\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Wishart--CIR numerical audit on all 253 dates of 2020, equal-weight portfolio at one percent. The 2020 rows change one numerical dimension and compare two-bank forecast ensembles, averaged before scoring. The 2025 rows compare individual banks on all 250 dates; their score difference is bank 1 minus bank 0. Passing requires relative-ES RMS at most 2\% and absolute annual score change at most 0.02. The 2025 runs use 16,384 predictive paths, except the 32,768-particle run, which uses 32,768. Other configuration values and independent-bank checks are archived; these are convergence diagnostics, not predictive-superiority tests.}\label{tab:wc-filter}',r'\end{table}']
(A/'filter_table.tex').write_text('\n'.join(lines)+'\n')
stress=[x for x in r['seed_comparisons'] if x['config'][0]==2025]
fine=max(stress,key=lambda x:(x['config'][1],bool(x['config'][5])))
n=fine['config'][1];rms=fine['es_relative_rms'];ds=abs(fine['bank_scores'][1]-fine['bank_scores'][0])
failed=rms>.02 or ds>.02
sentence=(f"Even with {n:,} particles and the corrected guided proposal, the independent-bank ES RMS discrepancy is {100*rms:.2f}\\%, above the 2\\% tolerance. " if failed else f"The finest independent-bank check uses {n:,} particles and gives ES RMS discrepancy {100*rms:.2f}\\%; this one check does not establish convergence of the full forecast panel. ")
text=r'''\paragraph{Conditional numerical audit.}
The finer 2020 checks satisfy the annual tolerances: doubling particles from 8,192 to 16,384 changes relative ES RMS by 0.77\% and annual score by $-0.0073$; extending warmup from 504 to 1,008 days changes them by 0.83\% and $-0.0099$. The 16-to-32-step change is 0.93\% and $+0.0037$. Failed coarse checks remain visible in Table~\ref{tab:wc-filter}. However, the 2025 stress fold is more difficult. '''+sentence+r'''These diagnostics prevent treating the new specification as a completed CRSP competitor. We retain the twenty training fits and completed stress-fold paths, but do not expand an unstable filter into a nominal 2,514-date result. No full-panel score, seed-uncertainty claim or portfolio-optimization recommendation is made for Wishart--CIR. It remains outside the original primary Holm and MCS families.
'''
(A/'filter_findings.tex').write_text(text)
status=dict(fits=20,synthetic_datasets=10,synthetic_paths_per_fit=131072,completed_filter_runs=len(pd.read_csv(A/'filter_runs.csv')),stress_years=[2020,2025],full_panel_completed=False,primary_families_unchanged=True,finest_2025_particles=n,finest_2025_bank_es_rms=rms,finest_2025_bank_score_difference=ds,finest_2025_fails=failed)
(A/'final_status.json').write_text(json.dumps(status,indent=2))

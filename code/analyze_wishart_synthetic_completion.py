"""Paired, within-fit synthetic Wishart discretisation audit."""
from pathlib import Path
import json,argparse,numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument("--n",type=int,default=32768);args=p.parse_args();N=args.n
from run_wishart_synthetic_completion import risk
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'wishart_completion/analysis';R=ROOT/'wishart_completion/results'
rows=[];ends=[]
for unit in range(10):
 tag=f'synthetic_d{unit}_b0_n{N}_s32';x=json.loads((R/(tag+'.json')).read_text());ends.append(x['endpoint_max_error'])
 for alpha in [.01,.001]:
  est={};inf={}
  for step in [4,8,16,32]:
   d=np.load(R/f'{tag}_grid{step}.npz');est[step],inf[step]=risk(d['loc'],d['sd'],alpha)
  old=json.loads((ROOT/'wishart_extension/results'/f'heston_{unit}_primary.json').read_text());oldr=next(z for z in old['metrics'] if z['alpha']==alpha)
  for coarse,fine in [(4,8),(8,16),(16,32),(4,32)]:
   delta=est[fine]['es']-est[coarse]['es'];se=np.std(inf[fine]-inf[coarse],ddof=1)/np.sqrt(x['paths'])
   rows.append(dict(dataset=unit,alpha=alpha,coarse=coarse,fine=fine,delta_es=delta,paired_mc_se=se,coarse_es=est[coarse]['es'],fine_es=est[fine]['es'],fine_mc_se=est[fine]['mc_se'],original_primary_es=oldr['es'],target=oldr['truth'],original_mc_se=oldr['scenario_mc_se']))
f=pd.DataFrame(rows);f.to_csv(O/'synthetic_grid_pairs.csv',index=False)
lines=[r'\begin{table}[htbp]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Dataset & $4\to32$, 1\% & $16\to32$, 1\% & $4\to32$, 0.1\% & $16\to32$, 0.1\% \\',r'\midrule']
for unit in range(10):
 vals=[]
 for a,c in [(.01,4),(.01,16),(.001,4),(.001,16)]:
  r=f[(f.dataset==unit)&(f.alpha==a)&(f.coarse==c)&(f.fine==32)].iloc[0];vals.append(f'{r.delta_es:.4f} ({r.paired_mc_se:.4f})')
 lines.append(str(unit)+' & '+' & '.join(vals)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Wishart weekly-ES grid audit on all ten fixed synthetic fits. Entries are fine-minus-coarse ES with paired Monte Carlo SE in parentheses. Each fit uses '+f'{N:,}'+r' common stationary paths and conditional Gaussian integration; exact OU factors coincide at shared grid endpoints. These are numerical refinements, not additional estimation datasets. The original million-scenario four-step evaluations remain separately archived.}\label{tab:wishart-coupled-grid}',r'\end{table}']
(O/'synthetic_grid.tex').write_text('\n'.join(lines)+'\n')
summary={'fits':10,'paths_per_fit':N,'endpoint_max_error':max(ends),'levels':{}}
for a in [.01,.001]:
 g=f[(f.alpha==a)&(f.coarse==16)&(f.fine==32)];h=f[(f.alpha==a)&(f.coarse==4)&(f.fine==32)]
 summary['levels'][str(a)]={'max_abs_16_32':float(g.delta_es.abs().max()),'max_se_16_32':float(g.paired_mc_se.max()),'max_abs_4_32':float(h.delta_es.abs().max()),'mean_4_32':float(h.delta_es.mean()),'max_abs_fine_difference_plus_1_96_se':float((g.delta_es.abs()+1.96*g.paired_mc_se).max())}
(O/'synthetic_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

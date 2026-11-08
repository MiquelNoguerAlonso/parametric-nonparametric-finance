"""Apply the audited Wishart manuscript layer after the round-four writer."""
from pathlib import Path
import json,shutil,re
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'wishart_completion';A=W/'analysis';P=W/'primary'

def apply():
 if not (P/'manifest.json').exists():return
 manifest=json.loads((P/'manifest.json').read_text());assert manifest['complete'] and manifest['fits']==10
 # The filtering table is built separately only after its declared run matrix is complete.
 if not (A/'final_status.json').exists() or not json.loads((A/'final_status.json').read_text())['audit_complete']:raise RuntimeError('The Wishart filtering audit has not been finalized')
 summary=json.loads((A/'synthetic_summary.json').read_text());assert summary['paths_per_fit']==manifest['conditional_paths_per_fit']
 se={a:[] for a in [.01,.001]};short={a:[] for a in se}
 for unit in range(10):
  ev=json.loads((P/f'heston_{unit}.json').read_text())
  for v in ev['metrics']:se[v['alpha']].append(v['scenario_mc_se']);short[v['alpha']].append(v['truth']-v['es'])
 z=pd.read_csv(ROOT/'referee_round4/analysis/synthetic_summary.csv');z=z[(z.study=='heston')&(z.model=='Heston-Wishart')].iloc[0]
 a=summary['levels']['0.01'];b=summary['levels']['0.001']
 text=f"All ten fixed fits are evaluated on the 32-step grid. Common-endpoint factor discrepancies are at most {summary['endpoint_max_error']:.2g}. The largest absolute 16-to-32-step ES change is {a['max_abs_16_32']:.4f} at one percent and {b['max_abs_16_32']:.4f} at 0.1\\%; the largest corresponding paired MC SEs are {a['max_se_16_32']:.4f} and {b['max_se_16_32']:.4f}. The fine-grid ES level MC SE ranges from {min(se[.01]):.3f} to {max(se[.01]):.3f} at one percent and {min(se[.001]):.3f} to {max(se[.001]):.3f} at 0.1\\%. These are numerical uncertainty scales, separate from the variation across estimation datasets.\n\n"
 text+=f"Fine-grid Wishart ES remains below the target for all ten fits; the smallest shortfalls are {min(short[.01]):.3f} and {min(short[.001]):.3f} at the two levels. Its updated RMSEs are {z.rmse:.3f} and {z.rmse_001:.3f}. Thus refining the numerical integration preserves the synthetic underestimation finding. The frozen-mixture dispersion bound diagnoses the fitted restriction; it is not asserted as a finite-horizon leveraged-return kurtosis bound.\n"
 (A/'synthetic_findings.tex').write_text(text)
 for name in ['wishart_methods.tex','wishart_appendix.tex']:shutil.copy2(W/'templates'/name,ROOT/name)
 p=ROOT/'expanded_methods.tex';s=p.read_text();old='Each neural model and scenario-based fitted benchmark generates \\textbf{one million evaluation scenarios}, giving about 10,000 and 1,000 tail draws at 1\\% and 0.1\\%.'
 new='Each neural model and scenario-based fitted benchmark apart from Wishart generates \\textbf{one million evaluation scenarios}, giving about 10,000 and 1,000 tail draws at 1\\% and 0.1\\%. The refined Wishart evaluation instead integrates Gaussian tails analytically over 131,072 stationary latent paths on a 32-step-per-day grid, with paired discretization and Monte Carlo checks in Appendix~\\ref{sec:wishart-numerics}.'
 if old in s:s=s.replace(old,new)
 p.write_text(s)
 p=ROOT/'refs.bib';s=p.read_text();extra=(W/'references.bib').read_text()
 for entry in re.split(r'(?=@article\{)',extra):
  if not entry.strip():continue
  key=re.search(r'@article\{([^,]+)',entry).group(1)
  if '{'+key+',' not in s:s+='\n'+entry
 p.write_text(s)
 p=ROOT/'README.md';s=p.read_text().replace('Extract all four archives','Extract all five archives')
 line='5. `Generative_Models_Finance_Wishart_Numerical_Audit.zip` — fine-grid Wishart simulations, numerical filter audits and restart checkpoints.'
 if line not in s:s=s.replace('Each part has a SHA-256 manifest.',line+'\n\nEach part has a SHA-256 manifest.')
 note='''\n<!-- WISHART-COMPLETION -->
## Wishart numerical revision

The ten synthetic Wishart fits now use 131,072 conditional-Gaussian paths at 32 return steps per day. Parameters, training datasets and target reference are unchanged. Coupled 4/8/16/32-step paths and paired Monte Carlo errors are retained. The CRSP algorithm audit is separate: a high resampling ESS is not a convergence result, and primary inference continues to exclude Wishart until its numerical acceptance gate is met. See `wishart_completion/analysis/final_status.json` for the completed-run decision and `wishart_completion/protocol.json` for the numerical criteria. New numerical evaluations do not add independent estimation datasets.
<!-- /WISHART-COMPLETION -->
'''
 s=re.sub(r'\n<!-- WISHART-COMPLETION -->.*?<!-- /WISHART-COMPLETION -->\n','\n',s,flags=re.S);s=s.replace('## Reproduction',note+'\n## Reproduction');p.write_text(s)
 p=ROOT/'referee_round4/analysis/verification.tex';s=p.read_text();s=s.replace('and round-four results/checkpoint archive.','round-four results/checkpoint archive, and Wishart numerical-audit archive.');p.write_text(s)
 print('Applied fine-grid Wishart manuscript layer')
if __name__=='__main__':apply()

"""Reproduce the second referee revision from retained results or fixed experiments."""
import argparse,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('Running',script,*args,flush=True)
 env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
 subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['outputs','experiments','legacy'],default='outputs');p.add_argument('--jobs',type=int,default=2);p.add_argument('--raw-crsp');p.add_argument('--pdf',action='store_true');a=p.parse_args()
 if a.mode=='legacy':run('run_legacy.py',*(['--pdf'] if a.pdf else []));return
 required=[ROOT/'expansion/data'/f'{law}_reference_losses.npz' for law in ['t','metat']]
 if a.mode=='outputs' and not all(p.exists() for p in required):raise SystemExit('Extract Generative_Models_Finance_Reference_Samples.zip into the same directory as the main source archive before output reconstruction.')
 if a.raw_crsp:
  run('prepare_crsp.py','--source',a.raw_crsp,'--out',ROOT/'data');run('prepare_highdim.py','--source',a.raw_crsp);run('verify_highdim_sources.py','--source',a.raw_crsp)
 if a.mode=='experiments':
  run('run_expanded_archive.py','--mode','experiments','--jobs',a.jobs)
  with ThreadPoolExecutor(max_workers=max(1,a.jobs)) as pool:
   fs=[pool.submit(run,x) for x in ['round2_heston.py','round2_filters.py','round2_leverage.py']]
   for f in fs:f.result()
  jobs=[]
  for study in ['t','metat','heston','crsp5','crsp100']:
   units=list(range(2016,2026)) if study.startswith('crsp') else list(range(10))
   for part in [units[:5],units[5:]]:jobs.append(('round2_selection.py','--study',study,'--units',*part))
  with ThreadPoolExecutor(max_workers=max(1,a.jobs)) as pool:
   fs=[pool.submit(run,*x) for x in jobs]
   for f in fs:f.result()
  wishart_jobs=[('run_wishart.py','--study',study,'--units',unit) for study,units in [('heston',range(10)),('crsp5',range(2016,2026))] for unit in units]
  with ThreadPoolExecutor(max_workers=max(1,a.jobs)) as pool:
   for f in [pool.submit(run,*job) for job in wishart_jobs]:f.result()
  refined_jobs=[('run_wishart.py','--study',study,'--units',unit,'--refined') for study,units in [('heston',[0,1,2]),('crsp5',[2016,2020,2025])] for unit in units]
  with ThreadPoolExecutor(max_workers=max(1,a.jobs)) as pool:
   for f in [pool.submit(run,*job) for job in refined_jobs]:f.result()
  run('run_round3_experiments.py','--jobs',a.jobs)
 for script in ['check_math.py','check_outputs.py','check_revision.py','check_expansion.py','check_expanded_outputs.py']:
  run(script)
 run('expanded_diagnostics.py','--part','analyze')
 run('round2_analysis.py')
 run('round2_figures.py')
 import expanded_sensitivities as missing
 missing.O=ROOT/'referee_round2/analysis';missing.missing()
 run('round2_heston_diagnostics.py')
 run('round2_summaries.py')
 run('check_round2.py')
 run('write_round2_paper.py')
 run('write_round2_notes.py')
 run('polish_final.py')
 run('restore_scope.py')
 run('check_wishart.py')
 run('analyze_wishart.py')
 run('integrate_wishart.py')
 run('check_round3.py')
 run('round3_analysis.py')
 run('round3_control_analysis.py')
 run('check_round3_outputs.py')
 run('round3_findings.py')
 run('integrate_round3.py')
 if a.pdf:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT,check=True)
if __name__=='__main__':main()

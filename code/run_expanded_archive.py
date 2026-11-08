"""Reproduce the complete expanded paper or rerun its fixed experiments."""
import argparse,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('Running',script,*args,flush=True)
 env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('OMP_NUM_THREADS','1')
 subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['outputs','experiments','legacy'],default='outputs');p.add_argument('--jobs',type=int,default=2);p.add_argument('--raw-crsp');p.add_argument('--pdf',action='store_true');a=p.parse_args()
 if a.mode=='legacy':run('run_legacy.py',*(['--pdf'] if a.pdf else []));return
 if a.raw_crsp:
  run('prepare_crsp.py','--source',a.raw_crsp,'--out',ROOT/'data');run('prepare_highdim.py','--source',a.raw_crsp);run('verify_highdim_sources.py','--source',a.raw_crsp)
 if a.mode=='experiments':
  run('expanded_experiment.py','--study','references')
  jobs=[('expanded_experiment.py','--study',law) for law in ['t','metat','heston']]
  for panel in ['crsp5','crsp100']:
   for years in [range(2016,2021),range(2021,2026)]:jobs.append(('expanded_experiment.py','--study',panel,'--years',*map(str,years),'--seeds',*(['0','1'] if panel=='crsp5' else ['0'])))
  jobs += [('expanded_experiment.py','--study','crsp5cov'),('expanded_dynamic.py',)]
  with ThreadPoolExecutor(max_workers=max(1,a.jobs)) as pool:
   futures=[pool.submit(run,*job) for job in jobs]
   for f in futures:f.result()
  for law in ['t','metat','heston']:run('expanded_diagnostics.py','--part',law)
 for script in ['check_math.py','check_outputs.py','check_revision.py','check_expansion.py','analyze_expansion.py','expanded_sensitivities.py']:
  run(script)
 run('expanded_diagnostics.py','--part','analyze')
 run('check_expanded_outputs.py')
 run('write_expanded_findings.py')
 run('write_release_notes.py')
 if a.pdf:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT,check=True)
if __name__=='__main__':main()

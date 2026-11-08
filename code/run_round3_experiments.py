"""Run the fixed control protocol; existing completed units are resumable."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,os,argparse
ROOT=Path(__file__).resolve().parents[1]

def call(task,study,unit):
    log=ROOT/'referee_round3/logs'/f'{task}_{study}_{unit}.log'
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    with log.open('a') as out:
        subprocess.run([sys.executable,str(ROOT/'code/round3_controls.py'),'--task',task,'--study',study,'--units',str(unit)],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)

def job(family,study,unit):
    if family=='pipeline':
        for task in ['filters','baselines','coord']:call(task,study,unit)
    else:call('lu',study,unit)
    print('COMPLETE',family,study,unit,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=3);a=parser.parse_args()
    jobs=[]
    for year in range(2016,2026):
        for study in ['crsp5','crsp100']:jobs.append(('pipeline',study,year))
        jobs.append(('lu','heston',year-2016))
        for study in ['crsp5','crsp100']:jobs.append(('lu',study,year))
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures={pool.submit(job,*j):j for j in jobs}
        failures=[]
        for future in as_completed(futures):
            try:future.result()
            except Exception as e:failures.append((futures[future],repr(e)));print('FAILED',failures[-1],flush=True)
    if failures:raise SystemExit(str(failures))

if __name__=='__main__':main()

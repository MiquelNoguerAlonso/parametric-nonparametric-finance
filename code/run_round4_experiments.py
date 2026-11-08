"""Resumable fixed-protocol computation; runs independent units on CPU groups."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from queue import Queue
import subprocess,os,sys,argparse,json,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'referee_round4'

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--phase',choices=['all','baselines','repair'],default='all');a=p.parse_args()
    available=sorted(os.sched_getaffinity(0));slots=Queue()
    for i in range(a.jobs):slots.put(available[i::a.jobs])
    tasks=[('original',s,u) for s in ['t','metat','heston','crsp5','crsp100','crsp5cov'] for u in (range(2016,2026) if s.startswith('crsp') else range(10))]
    tasks += [('coord',s,u) for s in ['crsp5','crsp100'] for u in range(2016,2026)]
    # High-dimensional controls first; order is fixed independently of outcomes.
    tasks.sort(key=lambda q:(q[1]!='crsp100',q[1]!='heston',q[2],q[0]))
    def job(family,study,unit):
        cpus=slots.get();tag=f'{family}_{study}_{unit}';log=OUT/'logs'/(tag+'.log');env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        def call(script,*args):
            with log.open('a') as f:subprocess.run(['taskset','-c',','.join(map(str,cpus)),sys.executable,str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        try:
            if a.phase!='repair':call('round4_baselines.py','--study',study,'--unit',unit,'--family',family)
            if a.phase!='baselines':
                for seed in ([0,1] if study=='crsp5' and family=='original' else [0]):call('round4_repair.py','--study',study,'--unit',unit,'--family',family,'--seed',seed)
                if family=='original' and study in ['heston','crsp5','crsp100']:call('round4_repair.py','--study',study,'--unit',unit,'--family','lu')
            print('COMPLETE',tag,flush=True)
        finally:slots.put(cpus)
    failed=[];start=time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures={pool.submit(job,*t):t for t in tasks}
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:failed.append(dict(task=futures[f],error=repr(e)));print('FAILED',failed[-1],flush=True)
    (OUT/'run_status.json').write_text(json.dumps(dict(phase=a.phase,units=len(tasks),failures=failed,seconds=time.time()-start),indent=2))
    if failed:raise SystemExit(1)

if __name__=='__main__':main()

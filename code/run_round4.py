"""Current complete build: archived results, repaired candidates, and manuscript."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(script,*args):
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    print('Running',script,*args,flush=True);subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['outputs','experiments','legacy'],default='outputs');p.add_argument('--jobs',type=int,default=3);p.add_argument('--raw-crsp');p.add_argument('--pdf',action='store_true');a=p.parse_args()
    if a.mode=='legacy':run('run_round2.py','--mode','outputs',*(['--pdf'] if a.pdf else []));return
    if a.mode=='experiments':
        run('run_round2.py','--mode','experiments','--jobs',a.jobs,*(['--raw-crsp',a.raw_crsp] if a.raw_crsp else []))
        run('run_round4_experiments.py','--jobs',a.jobs)
    if not (ROOT/'referee_round4/results/original_crsp100_2025_0_WGAN-t.json').exists():raise SystemExit('Extract the round-four results/checkpoint archive alongside the source and earlier companion archives, or run --mode experiments.')
    for script in ['round4_normalize_replay_candidates.py','round4_static.py','check_round4_kernels.py','round4_analysis.py','round4_controls_analysis.py','check_round4_outputs.py','round4_figures.py','write_round4_paper.py','write_round4_notes.py','write_wishart_completion.py']:run(script)
    if a.pdf:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT,check=True)
        run('check_round4_manuscript.py')
if __name__=='__main__':main()

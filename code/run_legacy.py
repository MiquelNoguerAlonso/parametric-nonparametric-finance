"""Reproduce outputs by default; optionally resume all experiments and diagnostics."""
import argparse,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run(script,*args):
    print('Running',script,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/'code'/script),*args],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['outputs','experiments'],default='outputs')
    p.add_argument('--raw-crsp',help='Optional authorized directory of 21 annual parquet extracts')
    p.add_argument('--pdf',action='store_true',help='Build PDF; requires latexmk and a TeX installation')
    a=p.parse_args()
    # Avoid oversubscribed BLAS threads; users can override these values.
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    if a.raw_crsp:run('prepare_crsp.py','--source',a.raw_crsp,'--out',str(ROOT/'data'))
    if a.mode=='experiments':
        for study in ['t','metat','heston','crsp']:run('experiment.py','--study',study)
        for part in ['heston','crsp','support']:run('robustness.py','--part',part)
    run('build_curves.py')
    run('check_math.py')
    run('analyze.py')
    run('check_outputs.py')
    if not (ROOT/'results/revision/dynamic_forecasts.csv.gz').exists():run('dynamic_benchmarks.py')
    run('revision_analysis.py')
    run('check_revision.py')
    run('verification_report.py')
    if a.pdf:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT,check=True)

if __name__=='__main__':main()

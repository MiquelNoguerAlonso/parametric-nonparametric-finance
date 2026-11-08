"""Verify current results without resetting the integrated manuscript.

The older checker writes a legacy verification paragraph, so retain and restore
that paragraph and its audit JSON. New check output is saved separately.
"""
import argparse,hashlib,json,runpy,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pdf',action='store_true');a=ap.parse_args()
    sys.path.insert(0,str(ROOT/'code'))
    baseline=json.loads((ROOT/'final_submission/editorial_numerical_baseline.json').read_text())
    failures=[name for name,sha in baseline.items() if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    assert not failures,failures
    old=[ROOT/'referee_round4/analysis'/n for n in ['verification.tex','verification.json']]
    saved={p:p.read_bytes() for p in old}
    try:
        runpy.run_path(str(ROOT/'code/check_round4_outputs.py'),run_name='__main__')
        (ROOT/'completion_audit/analysis/primary_reconstruction.json').write_bytes(old[1].read_bytes())
    finally:
        for p,data in saved.items():p.write_bytes(data)
    runpy.run_path(str(ROOT/'code/check_wishart_cir_move.py'),run_name='__main__')
    runpy.run_path(str(ROOT/'code/analyze_completion.py'),run_name='__main__')
    if a.pdf:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT,check=True)
        runpy.run_path(str(ROOT/'code/check_round4_manuscript.py'),run_name='__main__')
    print(f'Current release verified; {len(baseline)} baseline numerical files unchanged.')

if __name__=='__main__':main()

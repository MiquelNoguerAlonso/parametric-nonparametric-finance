"""Reconstruct illustrative curves from fitted models; never train a model."""
import json
from pathlib import Path
import numpy as np
import jax
from models import Neural, MetaT
from experiment import sample_law, es

ROOT=Path(__file__).resolve().parents[1]

def main():
    for study in ['t','metat']:
        train=sample_law(study,4000,1000)
        for record in sorted((ROOT/'results').glob(study+'_0_0_*.json')):
            r=json.loads(record.read_text());name=r['model']
            p=ROOT/'data'/f'curve_{study}_{name}.npz'
            if p.exists() and p.stat().st_size:
                with np.load(p) as f: assert np.isfinite(f['loss']).all()
                continue
            checkpoint=ROOT/'checkpoints'/f'{study}_0_0_{name}.npz'
            rng=np.random.default_rng(8000)
            if name=='Historical':s=train[rng.integers(len(train),size=100000)]
            elif name=='Gaussian':
                s=rng.multivariate_normal(train.mean(0),np.cov(train,rowvar=False)+1e-7*np.eye(2),100000)
            elif name=='Meta-t':s=MetaT.load(checkpoint.with_suffix('.meta.json')).sample(seed=8000)
            else:
                s=Neural.load(checkpoint).sample(seed=8000)
                if name=='Flow-meta-t':s=MetaT.load(checkpoint.with_suffix('.meta.json')).from_y(s)
            loss=np.sort(-s@np.ones(2)/2)
            expected=next(m['es'] for m in r['metrics'] if m['portfolio']=='equal' and m['alpha']==.01)
            assert abs(es(loss)-expected)<1e-5,(study,name,es(loss),expected)
            temp=p.with_suffix('.tmp.npz');np.savez_compressed(temp,loss=loss);temp.replace(p)
            print('Rebuilt',p.name,flush=True);jax.clear_caches()

if __name__=='__main__':main()

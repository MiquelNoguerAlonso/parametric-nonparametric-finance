"""Within-window dependence for the three added Heston controls."""
from pathlib import Path
import json,gc
import numpy as np,pandas as pd,jax
from models import Neural,MetaT
from round2_selection import cached_sampler
from experiment import stylised
from analyze import tex_table,fmt
from revision_analysis import label
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT/'referee_round2';D=NEW/'diagnostics';D.mkdir(exist_ok=True)
def main():
 for ds in range(10):
  for name in ['Meta-t','VAE-t','WGAN-i']:
   tag=f'heston_{ds}_0_{name}';p=D/(tag+'.json')
   if p.exists():continue
   if name=='Meta-t':n=MetaT.load(NEW/'checkpoints'/(tag+'.meta.json'));x=n.sample(100000,510000+ds)
   else:n=Neural.load(NEW/'checkpoints'/(tag+'.npz'));x=cached_sampler(n)(100000,510000+ds)
   p.write_text(json.dumps(dict(study='heston',dataset=ds,model=name,n_scenarios=100000,stylised=stylised(x)),indent=2));print('Added dependence',tag,flush=True);del n,x;gc.collect();jax.clear_caches()
 records=[json.loads(p.read_text()) for p in (ROOT/'expansion/diagnostics').glob('heston_*.json')];records=[r for r in records if r.get('study')=='heston'];records += [json.loads(p.read_text()) for p in D.glob('heston*.json')];assert len(records)==120
 ref=json.loads((ROOT/'expansion/diagnostics/heston_reference.json').read_text());keys=['kurtosis','cluster','leverage','cross_vol'];rows=[['Reference']+[fmt(ref[k]) for k in keys]];out=[]
 for name in sorted({r['model'] for r in records}):
  group=[r for r in records if r['model']==name];v={k:float(np.mean([r['stylised'][k] for r in group])) for k in keys};out.append(dict(model=name,**v));rows.append(['Empirical' if name=='Historical' else label(name)]+[fmt(v[k]) for k in keys])
 pd.DataFrame(out).to_csv(NEW/'analysis/heston_dependence.csv',index=False)
 (NEW/'analysis/heston_dependence_table.tex').write_text(tex_table('Within-window Heston dependence, now including Meta-t, VAE-tlatent and WGAN-i. Ten independent fitted datasets and 100,000 generated windows per fit. Leverage is the mean correlation of each return with its next-day absolute return within the same five-day window. A static joint-window generator can represent this statistic; no recursive or state-conditioned dynamics are claimed. Sample kurtosis need not estimate a finite population fourth moment for polynomial-tailed models.','tab:heston-expanded-dependence',['Model','Kurtosis','Cluster','Leverage','Cross-vol'],rows))
if __name__=='__main__':main()

"""Paired Heston leverage ablation under the current joint-bank simulator."""
from pathlib import Path
import json
import numpy as np
from experiment import heston_bank,es,stylised
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round2';bank=heston_bank();cv=np.linalg.cholesky(.4*np.eye(5)+.6*np.ones((5,5)));cp=np.linalg.cholesky(.5*np.eye(5)+.5*np.ones((5,5)));cf=np.linalg.cholesky(.451*np.eye(5)+.549*np.ones((5,5)))

def paths(n,seed):
 r=np.random.default_rng(seed);v=bank[r.integers(len(bank),size=n)].copy();out=np.zeros((3,n,5,5));dt=1/(252*20)
 for day in range(5):
  for k in range(20):
   b=r.standard_normal((n,5))@cv.T;normal=r.standard_normal((n,5));p=normal@cp.T;fixed=normal@cf.T;vp=np.maximum(v,0);sq=np.sqrt(vp*dt)
   out[0,:,:,day]+=-.5*vp*dt+sq*(-.7*b+np.sqrt(.51)*p)
   out[1,:,:,day]+=-.5*vp*dt+sq*fixed
   out[2,:,:,day]+=-.5*vp*dt+sq*p
   v+=3*(.04-vp)*dt+.5*sq*b
 return out.reshape(3,n,25)/np.sqrt(.04/252)

def main():
 loss=[[],[],[]];dep=[]
 for b in range(20):
  p=O/'data'/f'leverage_batch_{b}.npz';j=p.with_suffix('.json')
  if p.exists():a=np.load(p);ll=a['loss'];d=json.loads(j.read_text())
  else:
   x=paths(50000,880000+b);ll=-x.sum(2)/5;d=[stylised(z) for z in x];np.savez_compressed(p,loss=ll);j.write_text(json.dumps(d,indent=2))
  for i in range(3):loss[i].append(ll[i])
  dep.append(d);print('Leverage batch',b,flush=True)
 names=['rho_minus_0p7','rho_zero_fixed_price_covariance','rho_zero_fixed_orthogonal_covariance'];rows=[]
 for i,name in enumerate(names):
  full=np.concatenate(loss[i]);values={}
  for alpha in [.01,.001]:
   base=np.concatenate(loss[0]);diff=np.array([es(z,alpha)-es(b,alpha) for z,b in zip(loss[i],loss[0])]);values[str(alpha)]=dict(es=es(full,alpha),difference_from_leverage=es(full,alpha)-es(base,alpha),paired_batch_se=float(np.std(diff,ddof=1)/np.sqrt(20)),es_batch_se=float(np.std([es(z,alpha) for z in loss[i]],ddof=1)/np.sqrt(20)))
  rows.append(dict(model=name,n=1000000,rho=-.7 if i==0 else 0,price_shock_correlation=.549 if i<2 else .5,metrics=values,dependence={k:float(np.mean([d[i][k] for d in dep])) for k in dep[0][0]}))
 result=dict(paths=1000000,batches=20,conditional_on_joint_initial_bank=True,reference='Current five-asset five-day standardized Heston target; same variance paths and coupled price shocks',rows=rows)
 (O/'analysis/leverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

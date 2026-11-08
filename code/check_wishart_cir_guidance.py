"""Independent proposal-ratio checks, including accessible CIR boundary and negative tilts."""
import json
from pathlib import Path
import numpy as np
from wishart_cir import initial,guided_integrals
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wishart_cir_extension';p=json.loads((OUT/'fits/crsp5_2025.json').read_text())['parameters'];rng=np.random.default_rng(76422);rows=[];n=100000
for v0 in [0.,.01,10.]:
 for tilt in [-1.,0.,1.5]:
  y,v=initial(p,n,rng);v[:]=v0;_,vn,_,corr=guided_integrals(y,v,p,rng,8,np.full(n,tilt));weight=np.exp(corr)
  e=np.exp(-p['kappa_v']);mean=1+(v0-1)*e;var=2*v0*e*(1-e)/p['a']+(1-e)**2/p['a']
  for name,value,target in [('mass',weight,1.),('mean',weight*vn,mean),('second_moment',weight*vn**2,mean**2+var)]:
   se=value.std(ddof=1)/np.sqrt(n);z=abs(value.mean()-target)/se if se>0 else 0.;assert z<6,(v0,tilt,name,z)
   rows.append(dict(v0=v0,tilt=tilt,quantity=name,estimate=float(value.mean()),target=float(target),se=float(se),z=float(z)))
(OUT/'analysis/guided_boundary_checks.json').write_text(json.dumps(rows,indent=2));print('All',len(rows),'proposal checks passed; max absolute standardized deviation',max(x['z'] for x in rows))

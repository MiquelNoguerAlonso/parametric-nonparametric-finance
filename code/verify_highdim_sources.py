"""Independently reconstruct saved universes from source returns and verify exports."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'expansion/data'
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--repair-incomplete',action='store_true');a=p.parse_args()
audit=json.loads((D/'crsp100_audit.json').read_text());ids=sorted({i for u in audit['universes'] for i in u['permnos']});frames=[]
for x in audit['inputs']:
 path=Path(a.source)/x['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==x['sha256']
 f=pq.read_table(path,columns=['permno','dlycaldt','dlyret'],filters=[('permno','in',ids)]).to_pandas();f.dlycaldt=pd.to_datetime(f.dlycaldt);frames.append(f.pivot(index='dlycaldt',columns='permno',values='dlyret'))
raw=pd.concat(frames).sort_index();results=[]
for u in audit['universes']:
 y=u['test_year'];expected=raw.loc[raw.index.year<=y,u['permnos']];path=D/f'crsp100_{y}.csv';saved=pd.read_csv(path,index_col=0,parse_dates=True);saved.columns=saved.columns.astype(int);mask=pd.read_csv(D/f'crsp100_{y}_missing.csv.gz',index_col=0,parse_dates=True);mask.columns=mask.columns.astype(int)
 repaired=False
 if len(saved)!=len(expected):
  assert a.repair_incomplete,(y,len(saved),len(expected))
  assert len(list((ROOT/'expansion/results').glob(f'crsp100_{y}_0_*.json')))==0,'Cannot repair inputs for already completed forecasts'
  tmp=path.with_suffix('.tmp');expected.fillna(0).to_csv(tmp,float_format='%.10g');tmp.replace(path);saved=pd.read_csv(path,index_col=0,parse_dates=True);saved.columns=saved.columns.astype(int);repaired=True
 assert saved.index.equals(expected.index) and saved.columns.equals(expected.columns)
 assert mask.index.equals(expected.index) and np.array_equal(mask.values,expected.isna().values)
 err=float(np.max(abs(saved.values-expected.fillna(0).values)));assert err<1e-9,(y,err)
 results.append(dict(year=y,dates=len(saved),test_dates=int((saved.index.year==y).sum()),max_export_error=err,repaired_incomplete_export=repaired,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()));print(results[-1],flush=True)
(D/'source_reconstruction_checks.json').write_text(json.dumps(results,indent=2))

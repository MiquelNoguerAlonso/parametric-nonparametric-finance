"""Point-in-time 100-stock universes from the authorized CRSP CIZ extracts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from prepare_crsp import eligible,COLS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'expansion/data';OUT.mkdir(exist_ok=True,parents=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();source=Path(a.source)
 archive=json.loads((ROOT/'data/crsp_data_audit.json').read_text());panels=[];ends={};audit=[]
 for row in archive['files']:
  path=source/row['file'];digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert digest==row['sha256']
  f=pq.read_table(path,columns=['permno','dlycaldt','dlyret']).to_pandas();f.dlycaldt=pd.to_datetime(f.dlycaldt)
  wide=f.pivot(index='dlycaldt',columns='permno',values='dlyret');panels.append(wide)
  last=wide.index.max();end=pq.read_table(path,columns=COLS,filters=[('dlycaldt','=',last.to_pydatetime())]).to_pandas()
  ends[row['year']]=end[eligible(end)].sort_values(['dlycap','permno'],ascending=[False,True])
  audit.append(dict(year=row['year'],file=path.name,sha256=digest,dates=len(wide),columns=len(wide.columns)))
  print('Read',row['year'],len(wide),flush=True)
 ret=pd.concat(panels).sort_index();del panels
 universes=[]
 for year in range(2016,2026):
  train=(ret.index.year>=max(2006,year-11))&(ret.index.year<=year-2)
  availability=ret.loc[train].notna().mean()
  prior_count=ret.loc[ret.index<ret.index[train].min()].notna().sum()
  eligible_ids=set(availability[(availability>=.98)&(prior_count>=252)].index)
  chosen=ends[year-2].loc[lambda x:x.permno.isin(eligible_ids)].head(100);assert len(chosen)==100
  ids=chosen.permno.astype(int).tolist();panel=ret.loc[ret.index.year<=year,ids].copy()
  missing=panel.isna();test=panel.index.year==year
  info=dict(test_year=year,universe_date=str(ends[year-2].dlycaldt.max().date()),selection='Top 100 eligible prior-training-end market capitalizations, with >=98% finite training returns and >=252 finite pre-estimation observations; no validation/test screening',permnos=ids,train_min_fraction=float(availability[ids].min()),minimum_pre_estimation_observations=int(prior_count[ids].min()),missing_train_cells=int(missing.loc[train[:len(panel)]].sum().sum()),missing_test_cells=int(missing.loc[test].sum().sum()),test_dates=int(test.sum()),max_missing_equal_weight_test=float(missing.loc[test].mean(1).max()))
  target=OUT/f'crsp100_{year}.csv';tmp=target.with_suffix('.tmp')
  panel.fillna(0).to_csv(tmp,float_format='%.10g')
  reread=pd.read_csv(tmp,index_col=0,parse_dates=True);assert reread.shape==panel.shape and reread.index.equals(panel.index)
  tmp.replace(target);missing.to_csv(OUT/f'crsp100_{year}_missing.csv.gz');universes.append(info)
  print('Universe',year,'missing test',info['missing_test_cells'],flush=True)
 (OUT/'crsp100_audit.json').write_text(json.dumps(dict(source='Same authorized CRSP CIZ extracts as five-basket study; source digests verified',inputs=audit,universes=universes,missing='Zero return for absent/missing observations; missing masks supplied for worst-case sensitivity. Delisting returns are retained when present in CIZ.',units='arithmetic returns, decimal'),indent=2))
if __name__=='__main__':main()

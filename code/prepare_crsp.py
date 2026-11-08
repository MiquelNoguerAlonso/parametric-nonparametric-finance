"""Build lagged-membership, value-weighted size quintiles from CRSP CIZ.

No current-day eligibility filter is applied to realized returns. Missing held
returns use an explicit zero-return convention, with -100% sensitivity retained.
Input annual extracts are proprietary and are not redistributed in the package.
"""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS=['permno','dlycaldt','dlyret','dlycap','primaryexch','conditionaltype',
      'tradingstatusflg','sharetype','securitytype','securitysubtype','usincflg','issuertype']

def eligible(f):
    return ((f.sharetype=='NS')&(f.securitytype=='EQTY')&(f.securitysubtype=='COM')&
            (f.usincflg=='Y')&f.issuertype.isin(['CORP','ACOR'])&f.primaryexch.isin(['N','A','Q'])&
            (f.conditionaltype=='RW')&f.tradingstatusflg.isin(['A','H','S'])&
            np.isfinite(f.dlycap)&(f.dlycap>0))

def main():
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--out',default='data');a=a.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    files={}
    for p in Path(a.source).glob('*.parquet'):
        y=int(re.search(r'crsp_(\d{4})',p.name)[1])
        if y in files:raise ValueError(f'Duplicate annual input {y}')
        files[y]=p
    assert sorted(files)==list(range(2005,2026)),sorted(files)
    prior=None;audits=[];series=[];diagnostics=[];members=set();held=set()
    for year,p in sorted(files.items()):
        f=pq.read_table(p,columns=COLS).to_pandas();f['dlycaldt']=pd.to_datetime(f.dlycaldt)
        assert f.dlycaldt.dt.year.eq(year).all();dup=int(f.duplicated(['permno','dlycaldt']).sum());assert dup==0
        assert not (f.dlyret.dropna() < -1.00000001).any()
        dates=np.sort(f.dlycaldt.unique())
        current=f[['permno','dlycaldt','dlyret','tradingstatusflg']].rename(columns={'dlycaldt':'date','tradingstatusflg':'return_status'})
        form=f if prior is None else pd.concat([prior,f],ignore_index=True)
        alldates=np.sort(form.dlycaldt.unique());nextdate=dict(zip(alldates[:-1],alldates[1:]))
        h=form.loc[eligible(form),['permno','dlycaldt','dlycap']].copy()
        members.update(map(int,h.permno.unique()))
        h=h.sort_values(['dlycaldt','dlycap','permno']);rank=h.groupby('dlycaldt').cumcount()
        count=h.groupby('dlycaldt').permno.transform('size')
        h['bucket']=np.minimum(4,(5*rank//count)).astype(int)+1
        h['date']=h.dlycaldt.map(nextdate);h=h[h.date.notna()&h.date.ge(pd.Timestamp(year,1,1))]
        assert (h.dlycaldt<h.date).all();held.update(map(int,h.permno.unique()))
        h=h.merge(current,on=['date','permno'],how='left',validate='one_to_one',indicator=True)
        h['missing']=~np.isfinite(h.dlyret);h['return']=h.dlyret.fillna(0.)
        h['weighted']=h.dlycap*h['return'];h['missing_cap']=h.dlycap*h['missing']
        h['terminal_cap']=h.dlycap*(~h.return_status.isin(['A','H','S'])&~h['missing'])
        g=h.groupby(['date','bucket']).agg(cap=('dlycap','sum'),weighted=('weighted','sum'),missing_cap=('missing_cap','sum'),
                 terminal_cap=('terminal_cap','sum'),n=('permno','size'),missing_n=('missing','sum'))
        g['ret']=g.weighted/g.cap;g['missing_weight']=g.missing_cap/g.cap;g['terminal_weight']=g.terminal_cap/g.cap
        wide=g['ret'].unstack('bucket');wide.columns=[f'Size{i}' for i in wide.columns]
        assert wide.shape[1]==5 and wide.notna().all().all();series.append(wide)
        diagnostics.append(g.reset_index())
        audits.append({'year':year,'file':p.name,'sha256':hashlib.file_digest(open(p,'rb'),'sha256').hexdigest(),
            'rows':len(f),'duplicates':dup,'min_date':str(pd.Timestamp(dates[0]).date()),'max_date':str(pd.Timestamp(dates[-1]).date()),
            'held_rows':len(h),'held_missing_rows':int(h['missing'].sum()),'unmatched_held_rows':int(h._merge.eq('left_only').sum()),
            'included_nonactive_return_rows':int((~h.return_status.isin(['A','H','S'])&~h['missing']).sum()),
            'max_bucket_missing_weight':float(g.missing_weight.max())})
        prior=f[f.dlycaldt.eq(pd.Timestamp(dates[-1]))].copy()
        print(year,'rows',len(f),'port dates',len(wide),'missing max',g.missing_weight.max(),flush=True)
    panel=pd.concat(series);diag=pd.concat(diagnostics,ignore_index=True)
    panel.to_csv(out/'crsp_size_portfolios.csv',float_format='%.12g')
    diag.to_csv(out/'crsp_holdings_audit.csv',index=False,float_format='%.12g')
    audit={'source':'User-supplied CRSP CIZ daily annual extracts','files':audits,'rows':sum(x['rows'] for x in audits),
           'dates':len(panel),'min_date':str(panel.index.min().date()),'max_date':str(panel.index.max().date()),
           'eligible_permnos':len(members),'held_permnos':len(held),'construction':'Daily size quintiles of eligible prior-close securities, prior-close market-cap weights; deterministic PERMNO tie break.',
           'missing_convention':'Zero return for missing held returns; pessimistic alternative subtracts missing_weight for each bucket/date.',
           'max_missing_weight':float(diag.missing_weight.max()),'mean_missing_weight':float(diag.missing_weight.mean()),
           'included_nonactive_return_rows':sum(x['included_nonactive_return_rows'] for x in audits)}
    (out/'crsp_data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='files'},indent=2))

if __name__=='__main__':main()

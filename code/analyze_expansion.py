"""Publication outputs for the completed fresh experiment; strict fit accounting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from analyze import tex_table,fmt
from revision_analysis import inference,mcs,calibration,label,pformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'expansion';R=BASE/'results';O=BASE/'analysis';O.mkdir(exist_ok=True)
def records(prefix):return [json.loads(p.read_text()) for p in sorted(R.glob(prefix+'_*.json')) if json.loads(p.read_text()).get('study')==prefix]
def synthetic():
 summary=[];decomp=[];tables=[];dec_tables=[];raw=[];oracle=pd.read_csv(ROOT/'results/revision/oracle_nll.csv').set_index('study')
 for law,count in [('t',100),('metat',110),('heston',90)]:
  recs=records(law);assert len(recs)==count,(law,len(recs),count)
  for rec in recs:
   for m in rec['metrics']:raw.append(dict(study=law,dataset=rec['dataset'],model=rec['model'],nll=rec['nll'],**m))
  rows=[];drows=[]
  group=pd.DataFrame([x for x in raw if x['study']==law and x['portfolio']=='equal'])
  # Exact empirical baseline is available from every model's common training sample.
  empirical=group.groupby(['dataset','alpha']).first().reset_index();brows={}
  for alpha in [.01,.001]:
   e=empirical[empirical.alpha.eq(alpha)];brows[alpha]=float(np.sqrt(np.mean(e.sample_error**2)))
  rows.append(['Empirical (exact)','--',fmt(brows[.01]),fmt(brows[.001]),'0.000','0.000','0.000'])
  for name,g in group.groupby('model'):
   stats={}
   for alpha in [.01,.001]:
    a=g[g.alpha.eq(alpha)].sort_values('dataset');assert len(a)==10
    diff=a.total_error.to_numpy()**2-a.sample_error.to_numpy()**2;fit=a.fit_error.to_numpy();samp=a.sample_error.to_numpy();total=a.total_error.to_numpy()
    v=dict(study=law,model=name,alpha=alpha,n_datasets=10,nll=float(a.nll.mean()) if a.nll.notna().any() else np.nan,bias=float(total.mean()),rmse=float(np.sqrt(np.mean(total**2))),empirical_rmse=brows[alpha],delta_mse=float(diff.mean()),paired_se=float(diff.std(ddof=1)/np.sqrt(10)),rmse_skill=float(1-np.sqrt(np.mean(total**2))/brows[alpha]),fit_mse=float(np.mean(fit**2)),sampling_mse=float(np.mean(samp**2)),cross=float(2*np.mean(fit*samp)),total_mse=float(np.mean(total**2)),scenario_mc_se_mean=float(a.scenario_mc_se.mean()),reference_mc_se=0. if law=='t' else float(a.reference_mc_se.iloc[0]))
    summary.append(v);decomp.append(v);stats[alpha]=v
   a,b=stats[.01],stats[.001]
   nm='Empirical (MC)' if name=='Historical' else label(name)
   rows.append([nm,fmt(a['nll']),fmt(a['rmse']),fmt(b['rmse']),fmt(a['delta_mse']),fmt(a['paired_se']),fmt(a['rmse_skill'])])
   drows.append([nm,fmt(a['fit_mse']),fmt(a['sampling_mse']),fmt(a['cross']),fmt(a['delta_mse']),fmt(a['paired_se'])])
  if law in oracle.index:rows.insert(0,['True-density oracle',fmt(oracle.loc[law,'oracle_nll']),'--','--','--','--','--'])
  title={'t':'Bivariate Student-$t_3$: analytic ES reference','metat':'Heterogeneous meta-$t$: four-million-draw reference','heston':'Heston weekly windows: four-million-path reference'}[law]
  tables.append(tex_table(title+'. Ten independent estimation datasets and one optimization seed per dataset, with common one-percent validation FZ0 checkpoint selection. Each fitted generator uses one million evaluation scenarios; the exact empirical row enumerates the training distribution. $\\Delta$ MSE subtracts the paired exact-empirical squared ES error at one percent; SE is across ten datasets. Skill is $1-\\mathrm{RMSE}/\\mathrm{RMSE}_{emp}$. VAE NLL is an IWAE-64 estimate. The oracle uses the same test sample. Lower NLL and RMSE are better.','tab:expanded-'+law,['Model','NLL','RMSE 1\\%','RMSE 0.1\\%','$\\Delta$ MSE','SE','Skill'],rows))
  dec_tables.append(tex_table('Expanded '+law+' study, one-percent ES decomposition. Total MSE equals fit MSE plus sampling MSE plus the cross term $2\\overline{fs}$. The exact empirical estimator has zero fit discrepancy and total MSE equal to the sampling column. Paired differences and SE concern ten independent datasets; the shared reference Monte Carlo error is reported separately.','tab:decomp-'+law,['Model','Fit MSE','Samp. MSE','Cross','$\\Delta$ MSE','SE'],drows))
 pd.DataFrame(raw).to_csv(O/'synthetic_raw.csv',index=False);s=pd.DataFrame(summary);s.to_csv(O/'synthetic_summary.csv',index=False);pd.DataFrame(decomp).to_csv(O/'decomposition.csv',index=False)
 (O/'synthetic_tables.tex').write_text('\n'.join(tables));(O/'decomposition_tables.tex').write_text('\n'.join(dec_tables))
 refrows=[]
 for law in ['t','metat','heston']:
  ref=json.loads((BASE/'data'/f'{law}_reference.json').read_text())
  for a in [.01,.001]:
   z=ref[f'equal_{a}'];g=s[s.study.eq(law)&s.alpha.eq(a)]
   refrows.append([law,f'{100*a:g}\\%',fmt(z['es'],5),'exact' if z['analytic'] else fmt(z['mc_se'],5),fmt(g.scenario_mc_se_mean.min(),4)+'--'+fmt(g.scenario_mc_se_mean.max(),4)])
 (O/'mc_table.tex').write_text(tex_table('Reference ES and evaluation uncertainty. Nonanalytic references use four million scenarios (40 batches); fitted generators use one million scenarios (10 batches). The last column is the range across models of their mean within-fit batch standard-error diagnostic. The Heston reference is conditional on the numerical initial-state bank; these diagnostics exclude bank and time-discretization error.','tab:mc-expanded',['Law','Tail','Reference ES','Reference SE','Generator SE range'],refrows,'llrrr'))
 # Paired differences, with uncertainty across independently estimated datasets.
 fig,axs=plt.subplots(3,1,figsize=(7.0,8.2))
 for ax,law in zip(axs,['t','metat','heston']):
  g=s[s.study.eq(law)&s.alpha.eq(.01)&s.model.ne('Historical')].sort_values('delta_mse');y=np.arange(len(g))
  ax.errorbar(g.delta_mse,y,xerr=g.paired_se,fmt='o',ms=4,capsize=2);ax.axvline(0,color='black',lw=.7);ax.set_yticks(y,[label(x) for x in g.model],fontsize=8);ax.set_title({'t':r'Student $t_3$','metat':'Heterogeneous meta-t','heston':'Heston'}[law]);ax.set_xlabel('Paired MSE difference ± 1 SE',fontsize=8);ax.grid(axis='x',alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'figures/expanded_paired_mse.pdf');plt.close(fig)
 return s

def crsp(panel):
 expected=200 if panel=='crsp5' else 100;fit=records(panel);assert len(fit)==expected,(panel,len(fit),expected)
 files=[p for p in R.glob(panel+'_*.csv.gz') if 'dynamic' not in p.name]
 raw=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in files],ignore_index=True)
 dyn=pd.read_csv(ROOT/'results/revision/dynamic_forecasts.csv.gz',parse_dates=['date']) if panel=='crsp5' else pd.concat([pd.read_csv(p,parse_dates=['date']) for p in R.glob(panel+'_*_dynamic.csv.gz')],ignore_index=True)
 assert len(dyn)==2514*3*3*2,(panel,len(dyn));raw=pd.concat([raw,dyn],ignore_index=True)
 assert not raw.duplicated(['date','model','seed','portfolio','alpha']).any()
 raw.to_csv(O/(panel+'_forecasts.csv.gz'),index=False)
 daily=raw.groupby(['date','model','portfolio','alpha'],as_index=False).agg(fz0=('fz0','mean'),hit=('hit','mean'),es=('es','mean'),es_residual=('es_residual','mean'),loss=('loss','mean'))
 fitdf=pd.DataFrame(fit);nll={}
 for name,g in fitdf.groupby('model'):
  ok=g.nll.notna();nll[name]=float(np.average(g.loc[ok,'nll'],weights=g.loc[ok,'n_test'])) if ok.any() else np.nan
 summaries=[];infs=[];mcsall=[];tables=[]
 for alpha in [.01,.025]:
  g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(alpha)];scores=g.pivot(index='date',columns='model',values='fz0').sort_index();assert scores.shape==(2514,13) and scores.notna().all().all()
  inf=inference(scores,'Historical');inf['alpha']=alpha;infs.append(inf);ri=inf.set_index('model')
  for block in ([10,20,40] if alpha==.01 else [20]):
   m=mcs(scores,block);m['alpha']=alpha;mcsall.append(m)
  rm=mcsall[-2 if alpha==.01 else -1].set_index('model');rows=[]
  for name in scores.mean().sort_values().index:
   a=g[g.model.eq(name)];b=ri.loc[name];v=dict(model=name,alpha=alpha,fz0=float(a.fz0.mean()),hits_percent=float(100*a.hit.mean()),es_moment=float(a.es_residual.mean()/a.es.mean()),nll=nll.get(name,np.nan),delta_fhs=b.delta,low=b.low,high=b.high,p=b.p,p_holm=b.p_holm,mcs_p=rm.loc[name,'p_mcs']);summaries.append(v)
   rows.append([label(name),fmt(v['fz0']),fmt(v['hits_percent'],2),fmt(v['es_moment']),fmt(b.delta),'['+fmt(b.low)+', '+fmt(b.high)+']',pformat(v['mcs_p'])])
  tables.append(tex_table(('Five CRSP size baskets' if panel=='crsp5' else 'Annual 100-stock CRSP universes')+f', equal-weight portfolio, {100*alpha:g}\\% tail probability, 2,514 test dates. Neural checkpoints are selected by common validation FZ0 at one percent. Generators use 100,000 forecast scenarios; FHS enumerates its empirical innovation distribution. Intervals are paired basic circular-bootstrap intervals (9,999 draws; 20-day blocks); MCS reports range-statistic elimination $p$-values. Ranking and calibration are distinct. The '+('two optimization seeds are averaged by date before inference.' if panel=='crsp5' else '100-stock panel uses one optimization seed and no test-survival screen.'),'tab:'+panel+'-'+str(alpha),['Model','FZ0','Hits \\%','ES moment','$\\Delta$ FHS','95\\% interval','MCS $p$'],rows))
 summary=pd.DataFrame(summaries);summary.to_csv(O/(panel+'_summary.csv'),index=False);pd.concat(infs).to_csv(O/(panel+'_inference.csv'),index=False);pd.concat(mcsall).to_csv(O/(panel+'_mcs.csv'),index=False)
 (O/(panel+'_primary.tex')).write_text(tables[0]);(O/(panel+'_secondary.tex')).write_text(tables[1])
 rows=[[label(name),'IWAE-64' if name.startswith('VAE') else 'Exact density',fmt(v,4)] for name,v in sorted(nll.items(),key=lambda item:np.inf if not np.isfinite(item[1]) else item[1]) if np.isfinite(v)]
 (O/(panel+'_density.tex')).write_text(tex_table(('Five-dimensional' if panel=='crsp5' else '100-dimensional')+' joint NLL on percentage returns, weighted by test dates and averaged across the available seeds. VAE estimates have upward NLL bias in expectation and realization noise. Portfolio GARCHs do not supply a comparable joint density.','tab:'+panel+'-density',['Model','Evaluation','Joint NLL'],rows,'llr'))
 cal=[]
 for (name,seed,alpha),g in raw[raw.portfolio.eq('equal')].groupby(['model','seed','alpha']):cal.append(dict(model=name,seed=seed,alpha=alpha,**calibration(g)))
 cal=pd.DataFrame(cal);cal.to_csv(O/(panel+'_calibration.csv'),index=False);rows=[]
 for name,g in cal[cal.alpha.eq(.01)].groupby('model'):
  g=g.sort_values('seed');rows.append([label(name),' / '.join(map(str,g.hits)),' / '.join(pformat(x) for x in g.kupiec_p),' / '.join(pformat(x) for x in g.cc_p)])
 (O/(panel+'_calibration.tex')).write_text(tex_table(panel.replace('crsp','CRSP ')+' one-percent calibration. Slash-separated entries refer to separate optimization/scenario seeds. UC is Kupiec unconditional coverage; CC is the four-moment VaR--ES conditional diagnostic with constant and lagged-hit instruments and Bartlett HAC bandwidth 20. All tests are unadjusted exploratory diagnostics conditional on fitted forecasts.','tab:'+panel+'-calibration',['Model','Hits','UC $p$','CC $p$'],rows))
 g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(.01)];scores=g.pivot(index='date',columns='model',values='fz0').sort_index();loss=g.groupby('date').loss.first();rob=[];rows=[]
 masks={'All':np.ones(len(scores),bool),'No 2020':scores.index.year!=2020}
 for k in [5,10,20]:masks[f'Drop {k}']=~scores.index.isin(loss.nlargest(k).index)
 for name in scores.columns:
  row=[label(name)]
  for key,mask in masks.items():
   delta=float((scores[name]-scores.Historical)[mask].mean());rob.append(dict(model=name,subset=key,n=int(mask.sum()),delta_fhs=delta));row.append(fmt(delta))
  rows.append(row)
 pd.DataFrame(rob).to_csv(O/(panel+'_robustness.csv'),index=False)
 (O/(panel+'_robustness.tex')).write_text(tex_table(panel.replace('crsp','CRSP ')+' mean score differences from FHS after common-date exclusions. Largest losses select dates ex post; these are influence diagnostics, not deployable exclusion rules.','tab:'+panel+'-robustness',['Model',*masks],rows))
 conc=[]
 for (pn,a),g in daily[daily.portfolio.ne('equal')].groupby(['portfolio','alpha']):
  base=g[g.model.eq('Historical')].fz0.mean()
  for name,z in g.groupby('model'):conc.append(dict(portfolio=pn,alpha=a,model=name,fz0=float(z.fz0.mean()),hits_percent=float(100*z.hit.mean()),delta_fhs=float(z.fz0.mean()-base)))
 pd.DataFrame(conc).to_csv(O/(panel+'_concentrated.csv'),index=False)
 rows=[[a['portfolio'],label(a['model']),fmt(a['fz0']),fmt(a['hits_percent'],2),fmt(a['delta_fhs'])] for a in conc if a['alpha']==.01]
 (O/(panel+'_concentrated.tex')).write_text(tex_table(panel.replace('crsp','CRSP ')+' concentrated portfolios at one percent, including FHS. These are secondary comparisons without a separate superiority claim.','tab:'+panel+'-concentrated',['Portfolio','Model','FZ0','Hits \\%','$\\Delta$ FHS'],rows,'llrrr'))
 annual=daily[daily.portfolio.eq('equal')&daily.alpha.eq(.01)].assign(year=lambda z:z.date.dt.year).groupby(['year','model']).fz0.mean().unstack()
 annual.to_csv(O/(panel+'_annual.csv'))
 return summary,raw

def parameters():
 rows=[];full=[]
 for panel in ['crsp5','crsp100']:
  for year in range(2016,2026):
   p=BASE/'checkpoints'/f'{panel}_{year}_0_Meta-t.meta.json';z=json.loads(p.read_text());full.append(dict(panel=panel,year=year,**z));nu=np.array(z['marg'])[:,0]
   rows.append([panel.replace('crsp',''),year,fmt(nu.min(),2),fmt(np.median(nu),2),fmt(nu.max(),2),fmt(z['nu'],2)])
 (O/'parameters_full.json').write_text(json.dumps(full,indent=2))
 (O/'parameter_table.tex').write_text(tex_table('New constrained meta-$t$ fits: minimum, median, and maximum marginal degrees of freedom, plus the copula parameter. Complete locations, scales, marginal degrees of freedom, and correlation matrices are exported. WGAN-t keeps $\\nu=3$.','tab:expanded-parameters',['Dimension','Year','$\\nu_{min}$','$\\nu_{med}$','$\\nu_{max}$','$\\nu_c$'],rows,'llrrrr'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['synthetic','crsp5','crsp100','all'],default='all');a=p.parse_args()
 if a.part in ['synthetic','all']:synthetic()
 for panel in ['crsp5','crsp100']:
  if a.part in [panel,'all']:crsp(panel)
 if a.part=='all':
  parameters();fig,axs=plt.subplots(2,1,figsize=(8.5,6),sharex=True)
  for ax,panel in zip(axs,['crsp5','crsp100']):
   annual=pd.read_csv(O/(panel+'_annual.csv'),index_col=0)
   for i,name in enumerate(['Historical','Meta-t','Flow-t','Flow-meta-t','VAE-G','VAE-H','GARCH-t','GARCH-EVT','EWMAcov-FHS']):
    ax.plot(annual.index,annual[name]-annual.Historical,marker=['o','s','^','D','v','P','X','*','h'][i],ms=4,lw=1.2,ls=['-','--',':','-.'][i%4],label=label(name))
   ax.axhline(0,color='black',lw=.6);ax.set_ylabel('FZ0 minus FHS');ax.set_title('Five size baskets' if panel=='crsp5' else '100-stock annual universes');ax.grid(alpha=.2)
  axs[0].legend(ncol=3,fontsize=7);axs[-1].set_xlabel('Out-of-sample year');fig.tight_layout();fig.savefig(ROOT/'figures/expanded_annual.pdf');plt.close(fig)
 print('Analyzed',a.part,flush=True)
if __name__=='__main__':main()

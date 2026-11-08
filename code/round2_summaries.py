"""Matched selection, synthetic model completion, and density reporting."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy import stats
from analyze import tex_table,fmt
from revision_analysis import inference,label
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'expansion';NEW=ROOT/'referee_round2';O=NEW/'analysis';S=NEW/'selection'
TITLES={'t':'Student target','metat':'Heterogeneous meta-t target','heston':'Heston windows','crsp5':'Five size baskets','crsp100':'100-stock panel'}
RULES=['fz1','fz2p5','fz5','pooled1','final','density'];NAMES={'fz1':'FZ 1\\%','fz2p5':'FZ 2.5\\%','fz5':'FZ 5\\%','pooled1':'Pooled 1\\%','final':'Final','density':'Likelihood'}
def selected_records(study):
 out=[]
 for p in S.glob(study+'_*.json'):
  if '_choices' not in p.name and '_step' not in p.name:out.append(json.loads(p.read_text()))
 return out

def synthetic():
 records=[]
 for law,n in [('t',100),('metat',110),('heston',120)]:
  a=[json.loads(p.read_text()) for p in (OLD/'results').glob(law+'_*.json')]
  if law=='heston':a += [json.loads(p.read_text()) for p in (NEW/'results').glob(law+'_*.json')]
  assert len(a)==n,(law,len(a));records+=a
 raw=pd.DataFrame([dict(study=r['study'],model=r['model'],dataset=r['dataset'],nll=r['nll'],**m) for r in records for m in r['metrics'] if m['portfolio']=='equal']);raw.to_csv(O/'synthetic_raw.csv',index=False)
 summary=[]
 for (law,name,a),g in raw.groupby(['study','model','alpha']):
  assert len(g)==10;diff=g.total_error**2-g.sample_error**2;emp=float(np.sqrt(np.mean(g.sample_error**2)))
  summary.append(dict(study=law,model=name,alpha=a,nll=float(g.nll.mean()) if g.nll.notna().any() else np.nan,rmse=float(np.sqrt(np.mean(g.total_error**2))),empirical_rmse=emp,delta_mse=float(diff.mean()),paired_se=float(diff.std(ddof=1)/np.sqrt(10)),rmse_skill=float(1-np.sqrt(np.mean(g.total_error**2))/emp),bias=float(g.total_error.mean()),fit_mse=float(np.mean(g.fit_error**2)),sampling_mse=emp**2,cross=float(2*np.mean(g.fit_error*g.sample_error)),total_mse=float(np.mean(g.total_error**2))))
 summary=pd.DataFrame(summary);summary.to_csv(O/'synthetic_summary.csv',index=False)
 for law in ['t','metat','heston']:
  g=summary[summary.study.eq(law)];b=g.groupby('alpha').empirical_rmse.first();rows=[['Empirical (exact)',fmt(b.loc[.01]),fmt(b.loc[.001]),'0.000','0.000','0.000']];drows=[]
  for name,z in g.groupby('model'):
   z=z.set_index('alpha');a=z.loc[.01];nm='Empirical (MC)' if name=='Historical' else label(name)
   rows.append([nm,fmt(a.rmse),fmt(z.loc[.001,'rmse']),fmt(a.delta_mse),fmt(a.paired_se),fmt(a.rmse_skill)])
   drows.append([nm,fmt(a.fit_mse),fmt(a.sampling_mse),fmt(a.cross),fmt(a.delta_mse),fmt(a.paired_se)])
  caption={'t':'Student target, correctly specified for the meta-t family.','metat':'Heterogeneous meta-t target, correctly specified for plain meta-t and its identity residual extension.','heston':'Heston target, outside the fitted meta-t family; the added Meta-t, VAE-tlatent and WGAN-i controls complete the requested model comparison.'}[law]
  caption+=' Ten independent estimation datasets, one seed each, common one-percent FZ0 selection, and one million evaluation scenarios per fit. Paired MSE differences and SE compare each model with exact empirical-training ES at one percent. NLL is reported separately by checkpoint criterion; lower RMSE is better.'
  (O/(law+'_risk_table.tex')).write_text(tex_table(caption,'tab:expanded-'+law,['Model','RMSE 1\\%','RMSE 0.1\\%','$\\Delta$ MSE','SE','Skill'],rows))
  (O/(law+'_decomposition.tex')).write_text(tex_table(TITLES[law]+' one-percent ES error decomposition. Total MSE is fit MSE plus sampling MSE plus the cross term. All models share the same ten estimation datasets and references.','tab:decomp-'+law,['Model','Fit MSE','Samp. MSE','Cross','$\\Delta$ MSE','SE'],drows))
 h=raw[(raw.study=='heston')&(raw.alpha==.01)].pivot(index='dataset',columns='model',values='total_error');delta=h['VAE-H']**2-h['VAE-G']**2
 paired=dict(n=10,delta_mse=float(delta.mean()),paired_se=float(delta.std(ddof=1)/np.sqrt(10)),t_stat=float(stats.ttest_1samp(delta,0).statistic),exploratory_t_p=float(stats.ttest_1samp(delta,0).pvalue),differences=delta.tolist());(O/'vae_h_vs_g.json').write_text(json.dumps(paired,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,1,figsize=(7.2,9.1))
 for ax,law in zip(axs,['t','metat','heston']):
  g=summary[(summary.study==law)&(summary.alpha==.01)&(summary.model!='Historical')].sort_values('delta_mse');y=np.arange(len(g));ax.errorbar(g.delta_mse,y,xerr=g.paired_se,fmt='o',ms=4,capsize=2);ax.axvline(0,color='black',lw=.7);ax.set_yticks(y,[label(x) for x in g.model],fontsize=8);ax.set_title({'t':r'Student $t_3$: meta-t correctly specified','metat':'Heterogeneous meta-t: correctly specified','heston':'Heston: no assumed correctly specified family'}[law],fontsize=10);ax.set_xlabel('Paired MSE difference ± 1 SE',fontsize=8);ax.grid(axis='x',alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'figures/round2_paired_mse.pdf');plt.close(fig)
 mc=[]
 for law in ['t','metat','heston']:
  ref=json.loads((OLD/'data'/f'{law}_reference.json').read_text())
  for a in [.01,.001]:
   z=ref[f'equal_{a}'];g=raw[(raw.study==law)&(raw.alpha==a)].groupby('model').scenario_mc_se.mean();mc.append([law,f'{100*a:g}\\%',fmt(z['es'],5),'exact' if z['analytic'] else fmt(z['mc_se'],5),fmt(g.min(),4)+'--'+fmt(g.max(),4)])
 (O/'mc_table.tex').write_text(tex_table('Reference and evaluation precision for the completed model matrix. Nonanalytic references use four million draws; each primary fitted generator uses one million. The final column spans model-specific mean batch-SE diagnostics, including all three added Heston controls. Bank and discretization uncertainty are separate.','tab:mc-expanded',['Law','Tail','Reference ES','Reference SE','Generator SE range'],mc,'llrrr'))
 print('Synthetic completion tables written')

def selection():
 rows=[];changes=[];density=[];infer=[]
 for study,n in [('t',70),('metat',80),('heston',90),('crsp5',70),('crsp100',70)]:
  rec=selected_records(study);assert len(rec)==n,(study,len(rec),n)
  scores=[]
  for r in rec:
   for rule,v in r['evaluations'].items():
    changes.append(dict(study=study,unit=r['unit'],model=r['model'],criterion=rule,step=v['step'],primary_step=r['primary_step'],changed=v['step']!=r['primary_step']))
    base=dict(study=study,unit=r['unit'],model=r['model'],criterion=rule,step=v['step'],nll=v['nll'])
    if study.startswith('crsp'):
     f=pd.read_csv(S/v['forecasts'],parse_dates=['date']);f=f[f.alpha.eq(.01)];assert len(f) in [250,251,252,253]
     rows.append(dict(**base,n_test=len(f),fz0=float(f.fz0.mean()),hits_percent=float(100*f.hit.mean()),es_moment=float(f.es_residual.mean()/f.es.mean())))
     f['spec']=r['model']+'::'+rule;scores.append(f[['date','spec','fz0']])
    else:
     for m in v['metrics']:rows.append(dict(**base,alpha=m['alpha'],total_error=m['total_error'],sample_error=m['sample_error']))
  if study.startswith('crsp'):
   score=pd.concat(scores).pivot(index='date',columns='spec',values='fz0').sort_index();assert score.notna().all().all() and len(score)==2514
   fhs=pd.read_csv(OLD/'analysis'/(study+'_forecasts.csv.gz'),parse_dates=['date']);fhs=fhs[(fhs.model=='Historical')&(fhs.portfolio=='equal')&(fhs.alpha==.01)].groupby('date').fz0.mean();score['FHS']=fhs
   inf=inference(score,'FHS');inf['study']=study;infer.append(inf);score.to_csv(O/(study+'_selection_daily.csv.gz'))
  # Matched NLL: risk, native validation, and final states share training samples.
  for name in sorted({r['model'] for r in rec}):
   group=[r for r in rec if r['model']==name]
   if name.startswith('WGAN'):continue
   v={}
   for rule in ['fz1','density','final']:
    vals=[r['evaluations'][rule]['nll'] for r in group];weights=[]
    for r in group:
     if study.startswith('crsp'):weights.append(len(pd.read_csv(S/r['evaluations'][rule]['forecasts']))/2)
     else:weights.append(20000)
    v[rule]=float(np.average(vals,weights=weights))
   density.append(dict(study=study,model=name,**v))
 data=pd.DataFrame(rows);data.to_csv(O/'selection_raw.csv',index=False);ch=pd.DataFrame(changes);ch.to_csv(O/'checkpoint_changes.csv',index=False);den=pd.DataFrame(density);den.to_csv(O/'matched_density.csv',index=False);pd.concat(infer).to_csv(O/'selection_inference.csv',index=False)
 summaries=[];tablefiles=[];dentables=[]
 oldsynthetic=pd.read_csv(ROOT/'results/simulation_summary.csv');oldcrsp=pd.read_csv(ROOT/'results/crsp_summary.csv')
 primarysynthetic=pd.read_csv(O/'synthetic_summary.csv')
 for study in ['t','metat','heston','crsp5','crsp100']:
  g=data[data.study.eq(study)];rs=[]
  for name,z in g.groupby('model'):
   row=[label(name)];primary=z[z.criterion.eq('fz1')].sort_values('unit')
   for rule in RULES:
    a=z[z.criterion.eq(rule)].sort_values('unit')
    if len(a)==0:row.append('--');continue
    if study.startswith('crsp'):
     val=float(np.average(a.fz0,weights=a.n_test));v=dict(study=study,model=name,criterion=rule,fz0=val)
    else:
     a=a[a.alpha.eq(.01)];b=primary[primary.alpha.eq(.01)];delta=a.total_error.to_numpy()**2-b.total_error.to_numpy()**2;emp=a.total_error.to_numpy()**2-a.sample_error.to_numpy()**2;val=float(np.sqrt(np.mean(a.total_error**2)));v=dict(study=study,model=name,criterion=rule,rmse=val,paired_mse_from_primary=float(delta.mean()),paired_se=float(delta.std(ddof=1)/np.sqrt(10)),paired_mse_from_empirical=float(emp.mean()),paired_empirical_se=float(emp.std(ddof=1)/np.sqrt(10)))
    summaries.append(v);row.append(fmt(val))
   rs.append(row)
  if study.startswith('crsp'):
   base=pd.read_csv(O/(study+'_summary.csv'));value=base[(base.model=='Historical')&(base.alpha==.01)].fz0.iloc[0];rs.insert(0,['FHS']+[fmt(value)]*6)
  else:value=primarysynthetic[(primarysynthetic.study==study)&(primarysynthetic.alpha==.01)].empirical_rmse.iloc[0];rs.insert(0,['Empirical']+[fmt(value)]*6)
  caption=TITLES[study]+(': one-percent equal-portfolio test FZ0, seed zero and 2,514 dates.' if study.startswith('crsp') else ': one-percent ES RMSE across ten independent estimation datasets.')+' Columns change only checkpoint selection along identical training trajectories. Final uses the last checkpoint. Likelihood selects exact validation NLL for flows or IWAE-8 for VAEs; its table entries still report test risk performance. Test scenarios use common seeds. Rankings are retrospective sensitivity diagnostics; all paired statistics and the CRSP adjustment across forty neural rule/model contrasts are exported.'
  path=study+'_selection.tex';(O/path).write_text(tex_table(caption,'tab:'+study+'-selection',['Model']+[NAMES[x] for x in RULES],rs));tablefiles.append(path)
  dr=[]
  for name in sorted(set(den[den.study.eq(study)].model)|({'Gaussian','Meta-t'} if study in ['t','metat','heston','crsp5','crsp100'] else set())):
   if name in ['Gaussian','Meta-t']:
    if study.startswith('crsp'):a=pd.read_csv(OLD/'analysis'/(study+'_summary.csv'));v=float(a[(a.model==name)&(a.alpha==.01)].nll.iloc[0])
    else:v=float(primarysynthetic[(primarysynthetic.study==study)&(primarysynthetic.model==name)&(primarysynthetic.alpha==.01)].nll.iloc[0])
    risk=native=final=v
   else:a=den[(den.study==study)&(den.model==name)].iloc[0];risk,native,final=a.fz1,a.density,a.final
   archive=np.nan
   if study in ['t','metat','heston']:
    a=oldsynthetic[(oldsynthetic.study==study)&(oldsynthetic.model==name)]
    if len(a):archive=a.nll.iloc[0]
   elif study=='crsp5':
    a=oldcrsp[(oldcrsp.model==name)&(oldcrsp.portfolio=='equal')&(oldcrsp.alpha==.01)]
    if len(a):archive=a.nll.iloc[0]
   dr.append([label(name),fmt(risk),fmt(native),fmt(final),fmt(archive)])
  caption={'t':'Student target: NLL on the common 20,000 test observations, with the true-density oracle.','metat':'Heterogeneous meta-t target: NLL on the common 20,000 test observations, including its true density and the residual flow.','heston':'Heston windows: NLL on 20,000 common test observations, including the added Meta-t and VAE-tlatent controls. No exact Heston oracle is available.','crsp5':'Five-basket joint NLL on percentage-point returns, weighted by test-fold length; matched neural columns use seed zero.','crsp100':'100-stock joint NLL on percentage-point returns, weighted by test-fold length. The earlier density-selected archive has no 100-stock panel.'}[study]
  caption+=' VAE values are IWAE-64 estimates; other entries evaluate exact fitted densities. Common column definitions are given above.'
  if study in ['t','metat']:
   oracle=pd.read_csv(ROOT/'results/revision/oracle_nll.csv').set_index('study').loc[study,'oracle_nll'];dr.insert(0,['True oracle']+[fmt(oracle)]*4)
  path=study+'_matched_density.tex';(O/path).write_text(tex_table(caption,'tab:'+study+'-matched-density',['Model','Risk-selected','Native-selected','Final','Archive'],dr));dentables.append(path)
 pd.DataFrame(summaries).to_csv(O/'selection_summary.csv',index=False)
 (O/'selection_tables.tex').write_text('\n'.join('\\input{referee_round2/analysis/'+x+'}' for x in tablefiles))
 intro=r'''The following five density tables separate checkpoint criteria. Risk-selected, native-selected, and final neural states share training trajectories and test observations. Native means validation likelihood selection within the fixed budget, not a global optimum; statistical laws have no checkpoint choice. The archive uses the earlier density-selected design with different training datasets/seeds and is unpaired. VAE test values use IWAE-64, with the expectation-bias and Monte Carlo qualifications in Section~\ref{sec:objectives}. Units and dimensions differ across studies, so NLL levels are compared only within a table.

'''
 (O/'density_tables.tex').write_text(intro+'\n'.join('\\input{referee_round2/analysis/'+x+'}' for x in dentables))
 print('Matched selection and density tables written')

def leverage():
 x=json.loads((O/'leverage.json').read_text());rows=[]
 for r in x['rows']:
  a=r['metrics']['0.01'];name={'rho_minus_0p7':'Leverage $\\rho=-0.7$','rho_zero_fixed_price_covariance':'No leverage, price corr. fixed','rho_zero_fixed_orthogonal_covariance':'No leverage, orthogonal corr. fixed'}[r['model']]
  rows.append([name,fmt(r['price_shock_correlation']),fmt(a['es']),fmt(a['difference_from_leverage']),fmt(a['paired_batch_se']),fmt(r['dependence']['leverage'])])
 (O/'leverage_table.tex').write_text(tex_table('Paired Heston leverage ablation: one million five-day paths, the same joint initial-variance bank, and common variance paths/random inputs. The second row removes price/variance leverage while holding cross-asset price-shock correlation at 0.549. The third keeps the orthogonal-shock correlation at 0.5, so it also changes total price-shock dependence. Differences are relative to the first row; SE uses twenty paired 50,000-path batches. The final column is return/next-day absolute-return correlation.','tab:leverage-ablation',['Specification','Price corr.','ES 1\\%','$\\Delta$ ES','Paired SE','Leverage'],rows,'lrrrrr'))
 print('Leverage table written')
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['synthetic','selection','leverage','all'],default='all');a=p.parse_args()
 if a.part in ['synthetic','all']:synthetic()
 if a.part in ['selection','all']:selection()
 if a.part in ['leverage','all']:leverage()
if __name__=='__main__':main()

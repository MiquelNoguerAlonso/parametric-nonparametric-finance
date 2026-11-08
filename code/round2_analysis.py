"""Referee-round additions: matched filter attribution and augmented comparisons."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from analyze import tex_table,fmt
from revision_analysis import inference,mcs,calibration,label,pformat
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'expansion';NEW=ROOT/'referee_round2';O=NEW/'analysis'
def filters():
 decompositions=[]
 for panel in ['crsp5','crsp100']:
  display='Five-basket panel' if panel=='crsp5' else '100-stock panel'
  raw=pd.read_csv(OLD/'analysis'/(panel+'_forecasts.csv.gz'),parse_dates=['date']);extra=pd.read_csv(NEW/'results'/(panel+'_matched_filters.csv.gz'),parse_dates=['date'])
  allraw=pd.concat([raw,extra[extra.model.ne('EWMA-FHS-matched')]],ignore_index=True);assert not allraw.duplicated(['date','model','portfolio','alpha','seed']).any()
  allraw.to_csv(O/(panel+'_forecasts.csv.gz'),index=False)
  daily=allraw.groupby(['date','model','portfolio','alpha'],as_index=False).agg(fz0=('fz0','mean'),hit=('hit','mean'),es=('es','mean'),es_residual=('es_residual','mean'),loss=('loss','mean'))
  summaries=[];infs=[];mcss=[]
  for a in [.01,.025]:
   g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(a)];scores=g.pivot(index='date',columns='model',values='fz0').sort_index();assert scores.shape==(2514,15)
   inf=inference(scores,'Historical');inf['alpha']=a;infs.append(inf);ri=inf.set_index('model')
   for block in ([10,20,40] if a==.01 else [20]):
    mc=mcs(scores,block);mc['alpha']=a;mcss.append(mc)
   mc=next(x for x in mcss if x.alpha.iloc[0]==a and x.block.iloc[0]==20).set_index('model');rows=[]
   for name in scores.mean().sort_values().index:
    z=g[g.model.eq(name)];b=ri.loc[name];row=dict(model=name,alpha=a,fz0=float(z.fz0.mean()),hits_percent=float(100*z.hit.mean()),es_moment=float(z.es_residual.mean()/z.es.mean()),delta_fhs=b.delta,low=b.low,high=b.high,p=b.p,p_holm=b.p_holm,mcs_p=mc.loc[name,'p_mcs']);summaries.append(row)
    rows.append([label(name).replace('Portfolio-EWMA-FHS','Port.-EWMA-FHS'),fmt(row['fz0']),fmt(row['hits_percent'],2),fmt(row['es_moment']),fmt(b.delta),'['+fmt(b.low)+', '+fmt(b.high)+']',pformat(row['mcs_p'])])
   caption=('Five size baskets' if panel=='crsp5' else 'Annual 100-stock universes')+f', equal-portfolio {100*a:g}\\% risk over 2,514 dates. The augmented set has fifteen specifications, including GARCH-FHS and portfolio-EWMA-FHS. Paired basic intervals and range-statistic MCS use 9,999 circular draws and 20-day blocks. Holm tests in the text now cover fourteen FHS contrasts; the earlier twelve-contrast results are archived. All scores use percentage-point returns; common rescaling adds the same constant to every score. Hits are exceedance percentages.'
   (O/(panel+('_primary.tex' if a==.01 else '_secondary.tex'))).write_text(tex_table(caption,'tab:'+panel+'-'+str(a),['Model','FZ0','Hits \\%','ES moment','$\\Delta$ FHS','95\\% interval','MCS $p$'],rows))
  pd.DataFrame(summaries).to_csv(O/(panel+'_summary.csv'),index=False);pd.concat(infs).to_csv(O/(panel+'_inference.csv'),index=False);pd.concat(mcss).to_csv(O/(panel+'_mcs.csv'),index=False)
  cal=[]
  for (model,seed,a),z in allraw[allraw.portfolio.eq('equal')].groupby(['model','seed','alpha']):cal.append(dict(model=model,seed=seed,alpha=a,**calibration(z)))
  cal=pd.DataFrame(cal);cal.to_csv(O/(panel+'_calibration.csv'),index=False);rows=[]
  for name,z in cal[cal.alpha.eq(.01)].groupby('model'):
   z=z.sort_values('seed');rows.append([label(name),' / '.join(map(str,z.hits)),' / '.join(pformat(x) for x in z.kupiec_p),' / '.join(pformat(x) for x in z.cc_p)])
  (O/(panel+'_calibration.tex')).write_text(tex_table(display+' one-percent calibration in the augmented model set. UC is Kupiec coverage; CC is the unchanged finite-instrument conditional-moment diagnostic. Slash-separated entries are distinct seeds. These exploratory tests are unadjusted and condition on fitted forecasts.','tab:'+panel+'-calibration',['Model','Hits','UC $p$','CC $p$'],rows))
  g=daily[daily.portfolio.eq('equal')&daily.alpha.eq(.01)];scores=g.pivot(index='date',columns='model',values='fz0').sort_index();loss=g.groupby('date').loss.first();masks={'All':np.ones(len(scores),bool),'No 2020':scores.index.year!=2020}
  for k in [5,10,20]:masks[f'Drop {k}']=~scores.index.isin(loss.nlargest(k).index)
  rows=[];rr=[]
  for name in scores.columns:
   row=[label(name)]
   for key,mask in masks.items():
    v=float((scores[name]-scores.Historical)[mask].mean());row.append(fmt(v));rr.append(dict(model=name,subset=key,n=int(mask.sum()),delta_fhs=v))
   rows.append(row)
  pd.DataFrame(rr).to_csv(O/(panel+'_robustness.csv'),index=False);(O/(panel+'_robustness.tex')).write_text(tex_table(display+' score differences from FHS after common-date exclusions. All fifteen specifications are evaluated on identical retained dates. Exclusions are descriptive and selected ex post.','tab:'+panel+'-robustness',['Model',*masks],rows))
  conc=[]
  for (pn,a),z in daily[daily.portfolio.ne('equal')].groupby(['portfolio','alpha']):
   b=z[z.model.eq('Historical')].fz0.mean()
   for name,x in z.groupby('model'):conc.append(dict(portfolio=pn,alpha=a,model=name,fz0=float(x.fz0.mean()),hits_percent=float(100*x.hit.mean()),delta_fhs=float(x.fz0.mean()-b)))
  pd.DataFrame(conc).to_csv(O/(panel+'_concentrated.csv'),index=False);rows=[[x['portfolio'],label(x['model']),fmt(x['fz0']),fmt(x['hits_percent'],2),fmt(x['delta_fhs'])] for x in conc if x['alpha']==.01]
  (O/(panel+'_concentrated.tex')).write_text(tex_table(display+' concentrated-portfolio diagnostics at one percent, including the two new matched-filter controls.','tab:'+panel+'-concentrated',['Portfolio','Model','FZ0','Hits \\%','$\\Delta$ FHS'],rows,'llrrr'))
  # Four-step decomposition, including matched GARCH location to isolate variance filtering.
  matched=extra[(extra.model=='EWMA-FHS-matched')&(extra.portfolio=='equal')&(extra.alpha==.01)].set_index('date').fz0
  scores['EWMA-FHS-matched']=matched
  pairs=[('Aggregation','Portfolio-EWMA-FHS','Historical'),('Location','EWMA-FHS-matched','Portfolio-EWMA-FHS'),('Variance filter','GARCH-FHS','EWMA-FHS-matched'),('EVT tail','GARCH-EVT','GARCH-FHS'),('Total','GARCH-EVT','Historical'),('Student vs empirical','GARCH-t','GARCH-FHS')]
  differences=pd.DataFrame({desc:scores[a]-scores[b] for desc,a,b in pairs});differences['Zero']=0.;inf=inference(differences,'Zero').set_index('model');rows=[]
  identity=float(abs(differences[['Aggregation','Location','Variance filter','EVT tail']].sum(1)-differences.Total).max());assert identity<1e-10
  for desc,a,b in pairs:
   q=inf.loc[desc];decompositions.append(dict(panel=panel,contrast=desc,left=a,right=b,delta=q.delta,low=q.low,high=q.high,p=q.p,p_holm=q.p_holm,identity_max_error=identity));rows.append([desc,fmt(q.delta),'['+fmt(q.low)+', '+fmt(q.high)+']',pformat(q.p),pformat(q.p_holm)])
  (O/(panel+'_filter_decomposition.tex')).write_text(tex_table(display+' matched-filter score attribution at one percent. Aggregation changes multivariate diagonal-EWMA FHS to zero-mean portfolio-EWMA FHS; Location introduces the fitted GARCH mean into portfolio EWMA; Variance filter switches to the fitted GARCH recursion with that mean unchanged; EVT tail replaces only its empirical residual law. The first four differences sum exactly to Total on every date. Student vs empirical also holds the fitted GARCH mean and variance fixed. The six contrasts have their own exploratory Holm adjustment.','tab:'+panel+'-filter-decomposition',['Component','$\\Delta$ FZ0','95\\% interval','$p$','Holm $p$'],rows,'lrrrr'))
 pd.DataFrame(decompositions).to_csv(O/'filter_decomposition.csv',index=False);print('Augmented filter analysis complete')

def main():
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['filters'],default='filters');p.parse_args();filters()
if __name__=='__main__':main()

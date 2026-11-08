"""Matched controls and explicitly scoped retrospective comparison families."""
import json
import numpy as np,pandas as pd
from round4_analysis import ROOT,OUT,O,R,read,record,evaluation,pred,matrix,contrast,adjust,lab,interval
from round4_repair import names
from analyze import tex_table,fmt
from revision_analysis import inference,pformat

def joined(panel,name,rule='fz1',family='original',seed=0,alpha=.01):
    return pd.concat([pred(evaluation(panel,y,name,rule,family,seed),alpha) for y in range(2016,2026)]).sort_index()
def coordinate(panel,name,alpha=.01):
    if name.startswith('Flow') or name in ['Student-MLE','Gaussian-LW']:return joined(panel,name,family='coord',alpha=alpha)
    frames=[]
    for y in range(2016,2026):
        kind='ewma' if name=='Fitted-EWMA-FHS' else 'garch';oldname='FHS' if name in ['Historical','Fitted-EWMA-FHS'] else name
        ev=read(ROOT/'referee_round3/results'/f'coord_{kind}_{panel}_{y}_{oldname}.json');ev['forecasts']='referee_round3/results/'+ev['forecasts'];frames.append(pred(ev,alpha))
    return pd.concat(frames).sort_index()

def main():
    selections=[];selection_inf=[];own_inf=[];changes=[];selection_syn=[];lu=[];lusyn=[];filters=[];within=[];against=[];augmented=[];cov=[]
    for panel in ['crsp5','crsp100']:
        scores,raw=matrix(panel,.01,seed=0);columns={'FHS':scores.Historical};table=[];own=[]
        for name in names(panel,'original'):
            rules=list(record(panel,2016,name)['evaluations']);means={};early={}
            primary=joined(panel,name)
            for rule in rules:
                f=joined(panel,name,rule);value=float(f.fz0.mean());means[rule]=value;columns[name+'/'+rule]=f.fz0
                early[rule]=sum(record(panel,y,name)['choices'][rule]['early_added'] for y in range(2016,2026))
                selections.append(dict(panel=panel,model=name,rule=rule,fz0=value,early_folds=early[rule]))
                if rule!='fz1':own.append(dict(panel=panel,model=name,rule=rule,**contrast(f.fz0-primary.fz0)))
            for y in range(2016,2026):
                z=record(panel,y,name)
                for rule,c in z['choices'].items():changes.append(dict(study=panel,unit=y,model=name,rule=rule,new_step=c['step'],old_step=z['old_choices'][rule]['step'],early=c['early_added']))
            old=pd.read_csv(ROOT/'referee_round2/analysis'/f'{panel}_forecasts.csv.gz');old=old[(old.model==name)&(old.seed==0)&(old.alpha==.01)&(old.portfolio=='equal')]
            table.append([panel.replace('crsp',''),lab(name),fmt(means['fz1']),fmt(means.get('density')),fmt(means['final']),fmt(means['fz1']-old.fz0.mean()),f'{early["fz1"]}/10; '+(f'{early["density"]}/10' if 'density' in early else '--')])
        s=pd.DataFrame(columns);assert s.shape==(2514,41);z=inference(s,'FHS');z['panel']=panel;z['fz0']=z.model.map(s.mean());selection_inf+=z.to_dict('records');adjust(own);own_inf+=own
        (O/f'{panel}_selection.tex').write_text(tex_table('Matched repaired checkpoint rules, seed zero. Values are equal-portfolio one-percent FZ0. The final state is unchanged; likelihood uses validation NLL (VAE IWAE-8). $\\Delta$ risk compares repaired with original one-percent selection. Added choices count initialization, identity, or updates below 250. The full six-rule record and forty FHS contrasts are exported; comparisons with each model\'s primary rule form a separate 33-contrast family.','tab:r4-selection-'+panel,['Panel','Model','Risk','Likelihood','Final','$\\Delta$ risk','Added R/L'],table,'llrrrrl'))
        for alpha in [.01,.025]:
            base,_=matrix(panel,alpha,seed=0);local=[]
            for name in ['Flow-G','Flow-t']:
                for rule in ['fz1','density','final']:
                    a=joined(panel,name,rule,'lu',alpha=alpha);b=joined(panel,name,rule,alpha=alpha)
                    nll={}
                    for fam in ['original','lu']:
                        parts=[np.load(ROOT/evaluation(panel,y,name,rule,fam)['pointwise'])['nll'] for y in range(2016,2026)]
                        nll[fam]=float(np.concatenate(parts).mean())
                    local.append(dict(panel=panel,alpha=alpha,model=name,rule=rule,dense_fz0=float(a.fz0.mean()),diagonal_fz0=float(b.fz0.mean()),dense_nll=nll['lu'],diagonal_nll=nll['original'],**contrast(a.fz0-b.fz0)))
            adjust(local);lu+=local
            models=['Historical','Gaussian','Gaussian-LW','Student-MLE','Meta-t','Flow-G','Flow-t','Flow-meta-t','Fitted-EWMA-FHS'];frames={name:coordinate(panel,name,alpha) for name in models};local=[];wlocal=[];plocal=[]
            for name,f in frames.items():
                ref='Historical' if name=='Fitted-EWMA-FHS' else name
                local.append(dict(panel=panel,alpha=alpha,model=name,fz0=float(f.fz0.mean()),**contrast(f.fz0-base[ref])))
                plocal.append(dict(panel=panel,alpha=alpha,model=name,**contrast(f.fz0-base['GARCH-FHS'])))
                if name not in ['Historical','Fitted-EWMA-FHS']:wlocal.append(dict(panel=panel,alpha=alpha,model=name,**contrast(f.fz0-frames['Historical'].fz0)))
            adjust(local);adjust(wlocal);adjust(plocal);filters+=local;within+=wlocal;against+=plocal
            if alpha==.01:
                broad,_=matrix(panel,alpha)
                for name in ['Flow-G','Flow-t']:broad['LU-'+name]=joined(panel,name,family='lu').fz0
                for name,f in frames.items():broad['Coord-'+name]=f.fz0
                assert broad.shape==(2514,28) and broad.notna().all().all();z=inference(broad,'Historical');z['panel']=panel;z['fz0']=z.model.map(broad.mean());augmented+=z.to_dict('records');broad.to_csv(O/f'{panel}_augmented_daily.csv.gz')
        # Covariance-filter check remains paired with seed-zero diagonal filtering.
        if panel=='crsp5':
            base,_=matrix(panel,.01,seed=0);local=[]
            for name in ['Historical','Gaussian','Gaussian-LW','Student-MLE','Meta-t']+names(panel,'original'):
                if name.startswith(('Flow','VAE','WGAN')) or name in ['Student-MLE','Gaussian-LW']:f=joined('crsp5cov',name)
                else:
                    parts=[pred({'forecasts':f'expansion/results/crsp5cov_{y}_0_{name}.csv.gz'}) for y in range(2016,2026)];f=pd.concat(parts).sort_index()
                local.append(dict(model=name,fz0=float(f.fz0.mean()),**contrast(f.fz0-base[name])))
            adjust(local);cov+=local
    for study in ['t','metat','heston']:
        for name in names(study,'original'):
            rules=list(record(study,0,name)['evaluations'])
            for rule in rules:
                a=[];b=[];emp=[]
                for u in range(10):
                    z=record(study,u,name);v=next(x for x in z['evaluations'][rule]['metrics'] if x['alpha']==.01 and x.get('portfolio','equal')=='equal');p=next(x for x in z['evaluations']['fz1']['metrics'] if x['alpha']==.01 and x.get('portfolio','equal')=='equal')
                    a.append(v['total_error']);b.append(p['total_error']);emp.append(v['sample_error']);c=z['choices'][rule]
                    changes.append(dict(study=study,unit=u,model=name,rule=rule,new_step=c['step'],old_step=z['old_choices'][rule]['step'],early=c['early_added']))
                d=np.square(a)-np.square(b);de=np.square(a)-np.square(emp)
                selection_syn.append(dict(study=study,model=name,rule=rule,rmse=float(np.sqrt(np.mean(np.square(a)))),paired_mse=float(np.mean(d)),paired_se=float(np.std(d,ddof=1)/np.sqrt(10)),delta_empirical_mse=float(np.mean(de)),empirical_se=float(np.std(de,ddof=1)/np.sqrt(10))))
    for name in ['Flow-G','Flow-t']:
        for rule in ['fz1','density','final']:
            aa=[];bb=[];emp=[];nll=[];oldnll=[]
            for u in range(10):
                a=evaluation('heston',u,name,rule,'lu');b=evaluation('heston',u,name,rule);x=next(v for v in a['metrics'] if v['alpha']==.01);y=next(v for v in b['metrics'] if v['alpha']==.01 and v.get('portfolio','equal')=='equal')
                aa.append(x['total_error']);bb.append(y['total_error']);emp.append(x['sample_error']);nll.append(a['nll']);oldnll.append(b['nll'])
            d=np.square(aa)-np.square(bb);de=np.square(aa)-np.square(emp)
            lusyn.append(dict(model=name,rule=rule,dense_rmse=float(np.sqrt(np.mean(np.square(aa)))),diagonal_rmse=float(np.sqrt(np.mean(np.square(bb)))),delta_mse=float(d.mean()),se=float(d.std(ddof=1)/np.sqrt(10)),delta_empirical_mse=float(de.mean()),empirical_se=float(de.std(ddof=1)/np.sqrt(10)),dense_nll=float(np.mean(nll)),diagonal_nll=float(np.mean(oldnll))))
    outputs={'selection_summary':selections,'selection_inference':selection_inf,'selection_vs_primary':own_inf,'checkpoint_changes':changes,'synthetic_selection':selection_syn,'dense_crsp':lu,'dense_heston':lusyn,'coordinate_filter':filters,'within_garch':within,'coordinate_vs_portfolio':against,'augmented28':augmented,'covariance_filter':cov}
    for name,rows in outputs.items():pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
    (O/'dense_crsp.tex').write_text(tex_table('Dense versus diagonal output after repairing both candidate grids. FZ0 is at one percent; six head/rule risk contrasts are Holm-adjusted within each panel. NLL columns are descriptive same-date means. Dense models additionally admit their exact fitted Gaussian or multivariate Student nested state; capacity, initialization and candidate availability therefore change together.','tab:r4-dense',['Panel','Model','Rule','FZ0 diag./dense','$\\Delta$','Holm $p$','NLL diag./dense'],[[r['panel'].replace('crsp',''),r['model'],{'fz1':'Risk','density':'Likelihood','final':'Final'}[r['rule']],f'{r["diagonal_fz0"]:.3f}/{r["dense_fz0"]:.3f}',fmt(r['delta']),pformat(r['p_holm']),f'{r["diagonal_nll"]:.2f}/{r["dense_nll"]:.2f}'] for r in lu if r['alpha']==.01],'lllrrrr'))
    (O/'dense_heston.tex').write_text(tex_table('Heston dense-output controls after candidate repair. RMSE and paired MSE dispersion use ten matched estimation datasets at one percent. Neither the density criterion nor dense mixing is assumed to improve risk.','tab:r4-dense-heston',['Model','Rule','RMSE diagonal/dense','$\\Delta$ MSE','SE','NLL diagonal/dense'],[[r['model'],{'fz1':'Risk','density':'Likelihood','final':'Final'}[r['rule']],f'{r["diagonal_rmse"]:.3f}/{r["dense_rmse"]:.3f}',fmt(r['delta_mse']),fmt(r['se']),f'{r["diagonal_nll"]:.2f}/{r["dense_nll"]:.2f}'] for r in lusyn],'llrrrr'))
    for panel in ['crsp5','crsp100']:
        a=pd.DataFrame(filters);a=a[(a.panel==panel)&(a.alpha==.01)];w=pd.DataFrame(within);w=w[(w.panel==panel)&(w.alpha==.01)].set_index('model')
        rows=[]
        for r in a.itertuples():rows.append([lab(r.model),fmt(r.fz0),fmt(r.delta),pformat(r.p_holm),fmt(w.loc[r.model,'delta']) if r.model in w.index else '--',pformat(w.loc[r.model,'p_holm']) if r.model in w.index else '--'])
        (O/f'{panel}_coordinate.tex').write_text(tex_table(f'{panel.replace("crsp","")} coordinates: GARCH-filter refits and fitted-decay EWMA-FHS. The original-filter contrast changes location and residual fit as well as variance, with nine-contrast Holm correction. Within-GARCH comparisons hold the coordinate filter fixed and use seven-contrast Holm correction. All runs use seed zero.','tab:r4-coord-'+panel,['Innovation','FZ0','$\\Delta$ original','Holm/9','$\\Delta$ GARCH-FHS','Holm/7'],rows))
    fd=pd.read_csv(ROOT/'referee_round2/analysis/filter_decomposition.csv')
    (O/'filter_decomposition.tex').write_text(tex_table('Unchanged portfolio-filter attribution at one percent. The first four components sum exactly to Total on every date. Each panel has six exploratory contrasts. Student versus empirical holds the same fitted portfolio mean and GARCH variance fixed.','tab:r4-filter',['Panel','Component','$\\Delta$ FZ0','95\\% interval','Holm/6 $p$'],[[r.panel.replace('crsp',''),r.contrast,fmt(r.delta),interval(r),pformat(r.p_holm)] for r in fd.itertuples()],'llrrr'))
    print('CONTROL ANALYSES COMPLETE', {k:len(v) for k,v in outputs.items()})

if __name__=='__main__':main()

"""Integrate the completed repair; every numerical claim comes from retained output."""
from pathlib import Path
import json,re
import numpy as np,pandas as pd
from analyze import tex_table,fmt
from revision_analysis import pformat
from round4_analysis import ROOT,OUT,O,lab,evaluation
B=OUT/'baseline'

def combined_tables():
    den=pd.read_csv(O/'density_crsp.csv');rows=[]
    for panel in ['crsp5','crsp100']:
        for r in den[den.panel==panel].sort_values('nll').itertuples():rows.append([panel.replace('crsp',''),lab(r.model),fmt(r.nll),fmt(r.delta),f'[{r.low:.3f}, {r.high:.3f}]',pformat(r.p_holm)])
    (O/'density_combined.tex').write_text(tex_table('Joint density comparison on common CRSP dates after repaired likelihood selection, seed zero. Differences are from the full-covariance Gaussian. Basic intervals use 20-day paired blocks; eight comparisons with Gaussian are Holm-adjusted within each panel. VAE values are IWAE-64 estimates. These conditional intervals do not include retraining uncertainty.','tab:r4-density',['Panel','Model','NLL','$\\Delta$ Gaussian','95\\% interval','Holm $p$'],rows,'llrrrr'))
    joint=pd.read_csv(O/'joint_summary.csv');rows=[]
    for panel in ['crsp5','crsp100']:
        for r in joint[joint.panel==panel].sort_values('brier').itertuples():rows.append([panel.replace('crsp',''),lab(r.model),fmt(r.brier,4),fmt(r.delta,4),f'[{r.low:.4f}, {r.high:.4f}]',pformat(r.p_holm)])
    (O/'joint_combined.tex').write_text(tex_table('Joint downside-event Brier scores under the common diagonal-EWMA filter: at least two of five, or five of 100, innovations below coordinate training fifth percentiles. Eleven paired FHS contrasts are Holm-adjusted per panel. Event frequencies and predicted probabilities are exported. This retrospective joint-tail diagnostic has a separate comparison family from portfolio risk.','tab:r4-joint',['Panel','Model','Brier','$\\Delta$ FHS','95\\% interval','Holm $p$'],rows,'llrrrr'))
    selections=pd.read_csv(O/'selection_summary.csv');rows=[]
    for (panel,name),g in selections.groupby(['panel','model'],sort=False):
        v=g.set_index('rule');old=pd.read_csv(ROOT/'referee_round2/analysis'/f'{panel}_forecasts.csv.gz');old=old[(old.model==name)&(old.seed==0)&(old.alpha==.01)&(old.portfolio=='equal')]
        rows.append([panel.replace('crsp',''),lab(name),fmt(v.loc['fz1','fz0']),fmt(v.loc['density','fz0']) if 'density' in v.index else '--',fmt(v.loc['final','fz0']),fmt(v.loc['fz1','fz0']-old.fz0.mean()),str(int(v.loc['fz1','early_folds']))+'/'+(str(int(v.loc['density','early_folds'])) if 'density' in v.index else '--')])
    (O/'selection_combined.tex').write_text(tex_table('One-percent equal-portfolio FZ0 under repaired risk, likelihood and final-state rules, seed zero. $\\Delta$ risk is repaired minus original risk selection. Added R/L counts early or identity choices among ten folds under risk/likelihood selection. All six rules, forty FHS contrasts and 33 paired alternatives versus each model\'s primary rule are exported.','tab:r4-selection',['Panel','Model','Risk','Likelihood','Final','$\\Delta$ risk','Added R/L'],rows,'llrrrrl'))
    filters=pd.read_csv(O/'coordinate_filter.csv');within=pd.read_csv(O/'within_garch.csv');rows=[]
    for panel in ['crsp5','crsp100']:
        w=within[(within.panel==panel)&(within.alpha==.01)].set_index('model')
        for r in filters[(filters.panel==panel)&(filters.alpha==.01)].itertuples():rows.append([panel.replace('crsp',''),lab(r.model),fmt(r.fz0),fmt(r.delta),pformat(r.p_holm),pformat(w.loc[r.model,'p_holm']) if r.model in w.index else '--'])
    (O/'coordinate_combined.tex').write_text(tex_table('Coordinate-GARCH refits and fitted-EWMA-FHS after candidate repair, seed zero. $\\Delta$ compares the corresponding fixed-decay EWMA model, with nine-contrast Holm correction. The last column tests the innovation law against coordinate-GARCH-FHS under a common filter, with seven contrasts. Filtering, location and innovation refitting change together in the original-filter comparison.','tab:r4-coordinate',['Panel','Innovation','FZ0','$\\Delta$ original','Holm/9','Within Holm/7'],rows,'llrrrr'))
    cal=pd.read_csv(O/'calibration.csv');rows=[]
    for (panel,name),g in cal.groupby(['panel','model'],sort=False):
        g=g.sort_values('seed');rows.append([panel.replace('crsp',''),lab(name),'/'.join(map(str,g.hits.astype(int))),'/'.join(pformat(p) for p in g.kupiec_p),'/'.join(pformat(p) for p in g.cc_p)])
    (O/'calibration.tex').write_text(tex_table('One-percent calibration diagnostics for the repaired primary forecasts. Slash-separated values identify optimization seeds. UC is Kupiec coverage; CM is the four-moment joint VaR--ES Wald diagnostic with lagged-hit instruments and Bartlett HAC bandwidth 20. Rare exceptions and conditioning on fitted forecasts limit the chi-square approximation. These are unadjusted diagnostics.','tab:r4-calibration',['Panel','Model','Hits','UC $p$','CM $p$'],rows,'llrrr'))
    fam=[['Primary FHS comparisons','16 per panel','1\\% portfolio FZ0'],['Secondary level','16 per panel','2.5\\% portfolio FZ0'],['Checkpoint forecasts / FHS','40 per panel','1\\% FZ0, seed zero'],['Alternative rule / primary rule','33 per panel','Paired 1\\% FZ0'],['Portfolio-filter attribution','6 per panel','1\\% FZ0'],['Dense head / diagonal head','6 per panel and level','Paired FZ0'],['Coordinate filter / original','9 per panel and level','Paired FZ0'],['Innovation / coordinate-GARCH-FHS','7 per panel and level','Paired FZ0'],['Broad retrospective set / FHS','27 per panel','1\\% FZ0'],['Joint NLL / Gaussian','8 per panel','Paired log density'],['Joint downside event / FHS','11 per panel','Brier score'],['Full-covariance filter / diagonal','12, five baskets','1\\% FZ0']]
    (O/'families.tex').write_text(tex_table('Declared multiplicity families. Holm controls each displayed family, conditional on the test construction; the study makes no global error-control claim across its retrospective analyses.','tab:r4-families',['Comparison','Contrasts','Target'],fam,'lll'))
    rows=[]
    for panel in ['crsp5','crsp100']:
        for year in range(2016,2026):
            mt=json.loads((ROOT/'expansion/checkpoints'/f'{panel}_{year}_0_Meta-t.meta.json').read_text());nu=np.array(mt['marg'])[:,0]
            st=json.loads((OUT/'checkpoints'/f'original_{panel}_{year}_Student-MLE.json').read_text());lw=json.loads((OUT/'checkpoints'/f'original_{panel}_{year}_Gaussian-LW.json').read_text())
            rows.append([panel.replace('crsp',''),str(year),f'{nu.min():.2f}/{np.median(nu):.2f}/{nu.max():.2f}',f'{mt["nu"]:.2f}',f'{st["nu"]:.2f}',f'{lw["shrinkage"]:.4f}'])
    (O/'parameters.tex').write_text(tex_table('Training-only primary-panel parameter summaries. Meta-t uses staged marginal and copula estimation; Student MLE estimates one common degree of freedom jointly with location and scatter. Gaussian-LW estimates shrinkage intensity. Full locations, scales, scatter/correlation matrices and optimization diagnostics are archived. WGAN-t and WGAN-i keep input degrees of freedom three.','tab:r4-parameters',['Panel','Year','Meta marginal min/med/max','$\\nu_c$','Student $\\nu$','LW intensity'],rows,'llrrrr'))
    dec=pd.read_csv(O/'decomposition.csv')
    for study in ['t','metat','heston']:
        g=dec[dec.study==study];rows=[[('Empirical ES' if r.model=='Historical' else lab(r.model)),fmt(r.total_mse),fmt(r.fit_mse),fmt(r.sampling_mse),fmt(r.cross)] for r in g.itertuples()]
        (O/f'{study}_decomposition.tex').write_text(tex_table('One-percent paired ES error accounting for '+{'t':'the Student target','metat':'the heterogeneous meta-t target','heston':'the Heston target'}[study]+'. Total MSE equals fit MSE plus sampling MSE plus the signed cross term. The exact empirical estimator has zero fit error. Scenario noise is included in fitted estimates and separately retained in the numerical records.','tab:r4-decomp-'+study,['Model','Total MSE','Fit MSE','Sampling MSE','Cross term'],rows))

def revise_methods():
    s=(B/'expanded_methods.tex').read_text()
    s=s.replace('The expanded experiment uses fresh estimation datasets and newly trained neural models. It does not relabel the earlier selected checkpoints as risk-selected fits. A machine-readable protocol records seeds, training budgets, validation and test dates, and scenario counts. The original experiment is retained for audit, while the primary tables below concern the expanded design.','The current experiment retains the expanded training datasets and trajectories, repairs the checkpoint candidate set, and adds two training-only joint statistical benchmarks. A machine-readable protocol records seeds, budgets, dates and scenario counts. Earlier forecasts remain archived; the main tables use the repaired selection rule.')
    target='The Gaussian baseline uses a full sample covariance with a $10^{-7}I$ ridge.'
    addition=r'''The Gaussian baseline uses a full sample covariance with a $10^{-7}I$ ridge. Student-$t$ MLE estimates the full location $\mu$, scatter $\Sigma$ and common degrees of freedom $\nu\in[2.05,200]$. Three starts, $\nu=3,8,30$, use weighted normal-mixture location/scatter updates and conditional observed-likelihood maximization over $\nu$; the highest converged training likelihood is retained \citep{liu1995t}. This is numerical constrained likelihood estimation, with convergence and boundary diagnostics retained. Scatter differs from covariance, which is $\nu\Sigma/(\nu-2)$. For a linear loss, put $s_w^2=w^\top\Sigma w$ and $z_\alpha=t_\nu^{-1}(1-\alpha)$; then
\[
 q_\alpha=-w^\top\mu+s_wz_\alpha,\qquad
 e_\alpha=-w^\top\mu+s_w\frac{\nu+z_\alpha^2}{\nu-1}\frac{f_\nu(z_\alpha)}{\alpha}.
\]
Gaussian-LW uses the training mean and Ledoit--Wolf shrinkage, $(1-\widehat\lambda)S+\widehat\lambda\operatorname{tr}(S)I/d$, with $S$ normalized by the training sample size \citep{ledoit2004}. It is a regularization benchmark; no quadratic-risk optimality is asserted for the heavy-tailed targets. Both new laws have analytic portfolio VaR and ES.'''
    s=s.replace(target,addition)
    s=s.replace('Every 250 steps, \\emph{all neural families} are selected','At initialization, updates 10, 25, 50 and 100, and every 250 steps thereafter, \\emph{all neural families} are selected')
    s=s.replace('Thus the selection target is common although the estimation objectives and architectures differ.','Admissible identity and nested parametric flow candidates are included as detailed in Section~\\ref{sec:checkpoint-repair}. The selection target is common although estimation objectives and architectures differ.')
    s=s.replace('The retained design has 100 Student, 110 heterogeneous meta-$t$, and 120 Heston model evaluations; ten additional Heston--Wishart fits expand the Heston comparison to 130.','The current synthetic comparison has 120 Student, 130 heterogeneous meta-$t$, and 170 Heston model/dataset records, including the added Student and shrinkage-Gaussian laws, twenty dense Heston flows and ten Wishart fits. Exact empirical ES is computed from the same training samples; it requires no extra fit.')
    s=s.replace('Every fitted model generates \\textbf{one million evaluation scenarios}', 'Each neural model and scenario-based fitted benchmark generates \\textbf{one million evaluation scenarios}')
    s=s.replace('The Student ES reference is analytic.','The added Student and shrinkage-Gaussian benchmarks use analytic linear-projection risk, removing their scenario Monte Carlo component. The Student ES reference is analytic.')
    (ROOT/'expanded_methods.tex').write_text(s)
    s=(B/'expanded_crsp_methods.tex').read_text()
    s=s.replace('The new studies contain 200 five-basket and 100 100-stock model evaluations.','The added joint Student and shrinkage-Gaussian laws contribute twenty model/fold records to each primary panel. The reproducibility ledger distinguishes records, optimizer seeds and replayed trajectories.')
    s=s.replace('Ten within-ablation Holm contrasts','Twelve within-ablation Holm contrasts')
    s=s.replace('An additional five-basket ablation refits all ten original specifications after full-covariance whitening, then recolors scenarios with each predictable forecast-date Cholesky factor. It uses seed zero, the same training budgets, common validation FZ0, and 100,000 scenarios, adding 100 model evaluations.','An additional five-basket ablation refits the ten original specifications and the two added joint parametric benchmarks after full-covariance whitening, then recolors scenarios with each predictable forecast-date Cholesky factor. It uses seed zero, the same training budgets, repaired validation FZ0, and 100,000 neural scenarios, giving 120 model/fold records.')
    s=s.replace('There are fourteen non-FHS contrasts in each panel.','The repaired primary set has seventeen models and sixteen contrasts against diagonal-EWMA FHS in each panel.')
    s=s.replace('over fifteen specifications per panel','over seventeen specifications per panel')
    (ROOT/'expanded_crsp_methods.tex').write_text(s)
    s=(B/'round2_filter_methods.tex').read_text().replace('These add two specifications to each primary comparison, yielding fifteen models and fourteen within-panel FHS contrasts.','Together with the two added joint parametric benchmarks, the repaired primary comparison contains seventeen models and sixteen FHS contrasts.')
    (ROOT/'round2_filter_methods.tex').write_text(s)

def main():
    combined_tables();revise_methods()
    primary=pd.read_csv(O/'primary_summary.csv');p=primary[primary.alpha==.01];syn=pd.read_csv(O/'synthetic_summary.csv').set_index(['study','model']);den=pd.read_csv(O/'density_crsp.csv');joint=pd.read_csv(O/'joint_summary.csv');aug=pd.read_csv(O/'augmented28.csv')
    controls=pd.read_csv(O/'coordinate_filter.csv');within=pd.read_csv(O/'within_garch.csv');sel=pd.read_csv(O/'selection_summary.csv');changes=pd.read_csv(O/'checkpoint_changes.csv');lu=pd.read_csv(O/'dense_heston.csv')
    winners=p[(p.delta<0)&(p.p_holm<.05)];broad=aug[(aug.delta<0)&(aug.p_holm<.05)]
    if winners.empty:headline='No tested primary forecast improves detectably on filtered historical simulation (FHS) after within-panel multiplicity correction on 2,514 CRSP dates.'
    elif set(winners.model)=={'Student-MLE'} and set(winners.panel)=={'crsp100'}:headline='On 2,514 CRSP dates, fitted multivariate Student risk forecasts improve on filtered historical simulation (FHS) in 100 stocks under the revised primary multiplicity correction.'
    else:headline='The repaired CRSP comparison detects primary improvements over filtered historical simulation (FHS) for '+', '.join(lab(r.model)+' in '+r.panel.replace('crsp','')+' dimensions' for r in winners.itertuples())+'.'
    neural_winners=winners[winners.model.str.startswith(('Flow','VAE','WGAN'))]
    abstract='We compare parametric distributions, empirical nonparametric resampling, and neural generative models of joint asset returns to determine when additional flexibility improves density estimation and portfolio tail-risk forecasting. Parametric benchmarks comprise Gaussian, shrinkage-Gaussian, Student and meta-t distributions, plus a restricted Heston--Wishart model with evolving variances and correlations. Meta-t allows asset-specific Student tails joined by a t copula. Neural alternatives comprise normalizing flows, which learn invertible transformations of a base distribution, variational autoencoders, and Wasserstein generative adversarial networks. The empirical benchmark is filtered historical simulation (FHS), which rescales resampled standardized historical innovations by forecast volatility. We use ten independent estimation datasets for each of three synthetic laws---Student, heterogeneous meta-t, and correlated Heston---and 2,514 out-of-sample CRSP dates for five-basket and 100-stock panels. We evaluate joint density fit, simultaneous downside events, and portfolio Value-at-Risk and expected shortfall. Matched volatility-filter controls and alternative training-checkpoint rules help distinguish the effects of conditioning, model specification, and selection. Correctly specified parametric laws perform strongly in the synthetic studies. On CRSP, no neural improvement over FHS survives primary multiplicity correction under the common risk-based selection rule. A fitted Student distribution improves in the primary 100-stock comparison but not the broader retrospective comparison. GARCH point-score gains mainly reflect variance conditioning. The restricted Wishart fit understates synthetic tail risk; its unconverged CRSP filter is excluded from primary inference. The results show why density fit, dependence, and portfolio risk must be assessed separately.'
    (O/'abstract.tex').write_text(abstract+'\n')
    sn=r'\subsection{Controlled findings: specification before architecture}'+'\n'
    sn+=f'The homogeneous Student target belongs to both fitted Student and meta-$t$ families. Its one-percent RMSEs are {syn.loc[("t","Student-MLE"),"rmse"]:.3f} for Student MLE, {syn.loc[("t","Meta-t"),"rmse"]:.3f} for meta-$t$, and {syn.loc[("t","Historical"),"rmse"]:.3f} for exact empirical ES. The heterogeneous target belongs to meta-$t$ but generally not to the common-degree-of-freedom Student family. Their RMSEs there are {syn.loc[("metat","Meta-t"),"rmse"]:.3f} and {syn.loc[("metat","Student-MLE"),"rmse"]:.3f}, against empirical ES {syn.loc[("metat","Historical"),"rmse"]:.3f}. Favorable results therefore combine specification and estimation effects.\n'
    sn+=r'\input{referee_round4/analysis/t_risk.tex}'+'\n'+r'\input{referee_round4/analysis/metat_risk.tex}'+'\n'+r'\FloatBarrier'+'\n'
    sn+=r'\paragraph{Heston misspecification.}'+'\n'
    sn+=f'No fitted family is assumed to equal the correlated scalar-Heston target. Empirical ES has RMSE {syn.loc[("heston","Historical"),"rmse"]:.3f}; meta-$t$ has {syn.loc[("heston","Meta-t"),"rmse"]:.3f}, Student MLE {syn.loc[("heston","Student-MLE"),"rmse"]:.3f}, and dense Flow-G {syn.loc[("heston","LU-Flow-G"),"rmse"]:.3f}. The complete repaired comparison is in Table~\\ref{{tab:r4-syn-heston}}.\n'
    hz=syn.loc['heston'].sort_values('rmse');hn=hz.index[0];hr=hz.iloc[0]
    sn+=f'The lowest observed Heston RMSE after repair is {lab(hn)} at {hr.rmse:.3f}. Its paired MSE difference from empirical ES is {hr.delta_mse:.4f}, with dataset dispersion SE {hr.se:.4f}. This changes the point ranking without establishing a resolved advantage on ten datasets.\n'
    # Direct paired VAE contrast from retained errors.
    pairs=[]
    for u in range(10):
        es=[]
        for name in ['VAE-H','VAE-G']:
            ev=evaluation('heston',u,name);es.append(next(v['total_error'] for v in ev['metrics'] if v['alpha']==.01 and v.get('portfolio','equal')=='equal'))
        pairs.append(es[0]**2-es[1]**2)
    dm=float(np.mean(pairs));se=float(np.std(pairs,ddof=1)/np.sqrt(10))
    sn+=f'VAE-H minus VAE-G has paired MSE difference {dm:.4f}, with dataset SE {se:.4f}. '+('These ten datasets do not resolve a difference. ' if abs(dm)<2*se else 'This is a dataset-paired descriptive comparison under the repaired rule. ')
    sn+=f'WGAN-i changes from RMSE {syn.loc[("heston","WGAN-i"),"rmse"]:.3f} at one percent to {syn.loc[("heston","WGAN-i"),"rmse_001"]:.3f} at 0.1 percent, illustrating that selection at one level need not transfer to another. The tail theorems constrain asymptotic representation; they do not predict these finite-level errors.\n'
    sn+=r'\input{referee_round4/analysis/heston_risk.tex}'+'\n'
    sn+=r'''\paragraph{Central fit and extreme moments.}
Appendix~\ref{sec:heston} gives a frozen stationary variance-gamma snapshot with shape $a=0.96$, close to the Laplace case. It suggests a central-peak versus tail compromise for symmetric Student likelihoods, without being the exact dynamic daily law. The unchanged meta-$t$ Heston fits have all 250 marginal degrees of freedom below four, with median 3.15. Their population fourth moments are infinite; generated sample excess kurtosis 153.7 is a finite-sample diagnostic. Its NLL 28.165 and ES RMSE 1.126 show that central density fit can coexist with poor evaluated tail risk. They do not identify the peak as the unique cause.

A centrally symmetric copula with symmetric margins gives $R\overset d=-R$. Hence $\E[R_tg(R_{t+1})]=0$ for centered returns and even $g$, when the expectation exists. This constrains plain Student and meta-$t$ window laws, while a nonlinear residual flow can break the symmetry. A joint distribution on the ordered 25-dimensional window can encode leverage; it need not be a recursive state model.

\paragraph{Restricted stochastic covariance.}
The fitted Wishart specification is misspecified for this target. Its frozen stationary driftless mixture has excess kurtosis $6/b\le1$ for $b\ge6$, whereas the scalar-Heston stationary gamma mixture gives $3/0.96=3.125$. Matching dispersion would require $b=1.92$ outside the fitted class. Fitted reversion is $0.05$--$0.2$ per day, against the target's $3/252\simeq0.012$. These restrictions prevent attributing its error to stochastic correlation alone. The full fitted SDE, estimator and numerical checks are in Appendix~\ref{sec:wishart-fit-details}.

The paired target-law leverage ablation holds total price-shock correlation at 0.549. Removing leverage lowers one-percent weekly ES from 6.176 to 5.785, a 6.33\% reduction with paired-difference SE 0.019. This target-law result is unaffected by checkpoint repair; Table~\ref{tab:leverage-ablation} separates it from the alternative ablation that also changes price correlation.
\begin{figure}[htbp]\centering\includegraphics[width=0.98\linewidth]{figures/wishart_paired_mse.pdf}\caption{Repaired one-percent ES MSE differences from exact empirical-training ES, with one dataset-level dispersion SE. Structured parametric laws, flows and other neural generators share estimation datasets.}\label{fig:r4-synthetic}\end{figure}
\FloatBarrier
'''
    nus=[json.loads((OUT/'checkpoints'/f'original_heston_{u}_Student-MLE.json').read_text())['nu'] for u in range(10)]
    sd=pd.read_csv(O/'density_synthetic.csv');stnll=sd[(sd.study=='heston')&(sd.model=='Student-MLE')].nll.iloc[0]
    explanation=f'The added joint Student likelihood fits have $\\nu$ between {min(nus):.2f} and {max(nus):.2f}, also implying infinite fourth moments. Their mean NLL is {stnll:.3f}, versus Gaussian 31.500, while analytic one-percent ES RMSE is {syn.loc[("heston","Student-MLE"),"rmse"]:.3f}. This density--risk discrepancy persists for a training-likelihood estimator with analytic risk evaluation.\n\n'
    sn=sn.replace(r'\paragraph{Restricted stochastic covariance.}',explanation+r'\paragraph{Restricted stochastic covariance.}')
    (O/'synthetic_narrative.tex').write_text(sn)
    findings=''
    for panel in ['crsp5','crsp100']:
        z=p[p.panel==panel].set_index('model');best=z.fz0.idxmin();sig=z[(z.delta<0)&(z.p_holm<.05)]
        findings+='\\subsection{'+('Five-basket' if panel=='crsp5' else '100-stock')+' results}\n'
        findings+=f'The lowest observed mean is {lab(best)} at {z.loc[best,"fz0"]:.4f}, versus FHS {z.loc["Historical","fz0"]:.4f}. '
        findings+=('None of the sixteen FHS contrasts establishes improvement after within-panel Holm adjustment. ' if sig.empty else 'Within-panel Holm adjustment detects an improvement only for '+', '.join(lab(n) for n in sig.index)+'. ')
        for name in ['GARCH-EVT','GARCH-FHS','Student-MLE']:
            q=z.loc[name];findings+=f'{lab(name)} has difference {q.delta:.4f}, interval $[{q.low:.4f},{q.high:.4f}]$, and Holm $p={q.p_holm:.4f}$. '
        findings+=f'The 95\\% MCS retains {int((z.mcs_p>=.05).sum())} of seventeen models. Membership uses studentized pairwise differences and does not order retained forecasts by their sample means.\n'
        if not sig.empty:findings+='The MCS uses an all-pairs statistic and sequential elimination rule, so it can retain FHS despite a significant contrast in the separately defined Holm family against that benchmark.\n'
        findings+='\\input{referee_round4/analysis/'+panel+'_primary.tex}\n'
    findings+=r'''\paragraph{Conditioning and resolution.}
The portfolio-level decomposition holds location and residual scenarios fixed when changing the variance recursion. Its components still add exactly on each date. The incremental EVT effect is level-specific: the one-percent quantile lies inside an approximately 2,500-point innovation sample, where empirical interpolation remains possible. Undetected incremental EVT value here does not assess its extrapolation value at more extreme levels.
\input{referee_round4/analysis/filter_decomposition.tex}
The unadjusted interval half-widths for the GARCH/FHS contrasts are about 0.05 in five baskets and 0.12 in 100 stocks. Table~\ref{tab:r4-power} distinguishes this precision from conditional 80\%-power effects. Economically relevant differences can remain unresolved. The 2.5\% results are secondary; a future co-primary design requires advance specification.
\input{referee_round4/analysis/power.tex}
\begin{figure}[htbp]\centering\includegraphics[width=0.98\linewidth]{figures/round2_annual.pdf}\caption{Annual one-percent score differences from FHS under repaired selection. FHS is the zero reference. All model histories and primary seed conventions are retained in the accompanying records.}\label{fig:r4-annual}\end{figure}
\FloatBarrier
'''
    (O/'crsp_findings.tex').write_text(findings)
    densitytext='\\paragraph{Density results.}\n'
    for panel in ['crsp5','crsp100']:
        z=den[den.panel==panel];best=z.loc[z.nll.idxmin()];exact=z[~z.model.str.startswith('VAE')];be=exact.loc[exact.nll.idxmin()]
        densitytext+=f'In {panel.replace("crsp","")} dimensions, the lowest observed exact NLL is {lab(be.model)} at {be.nll:.3f}. '
        if best.model.startswith('VAE'):densitytext+=f'{lab(best.model)} has a lower realized IWAE estimate, {best.nll:.3f}; its expected variational bias works against this ordering, while Monte Carlo and test-sample uncertainty remain. '
    densitytext+='The paired intervals quantify test-period precision conditional on fitted states. A low NLL does not select a calibrated portfolio forecast automatically. The identity and nested candidates make a failure to improve the base visible as a selection outcome.\n'
    nidentity=sum(evaluation('crsp100',y,'Flow-meta-t','density')['step']==-1 for y in range(2016,2026))
    densitytext+=f'In the 100-stock panel, likelihood validation selects the exact meta-$t$ identity residual in {nidentity} of ten folds. Its NLL therefore reflects the fitted parametric base, not a learned residual improvement. The original Flow-t base has identity scatter; its repaired NLL is {den[(den.panel=="crsp100")&(den.model=="Flow-t")].nll.iloc[0]:.3f}, versus Gaussian {den[(den.panel=="crsp100")&(den.model=="Gaussian")].nll.iloc[0]:.3f}. Table~\\ref{{tab:r4-dense}} separately reports dense-head likelihood levels and risk scores.\n'
    (O/'density_findings.tex').write_text(densitytext)
    selectiontext='\\paragraph{Selection fragility after repair.}\n'
    for panel in ['crsp5','crsp100']:
        z=sel[sel.panel==panel].pivot(index='model',columns='rule',values='fz0');dif=z['final']-z['fz1']
        selectiontext+=f'Final states have lower observed one-percent scores than risk-selected states for {int((dif<0).sum())} of seven models in {panel.replace("crsp","")} dimensions; final-minus-selected differences range from {dif.min():.3f} to {dif.max():.3f}. '
    z=sel[(sel.panel=='crsp100')&(sel.model=='Flow-meta-t')].set_index('rule')
    selectiontext+=f'For 100-stock Flow-meta-$t$, the mean is {z.loc["fz1","fz0"]:.4f} under risk selection and {z.loc["density","fz0"]:.4f} under likelihood selection, versus FHS {p[(p.panel=="crsp100")&(p.model=="Historical")].fz0.iloc[0]:.4f}. The forty-contrast FHS and 33-contrast within-model adjustments are retained in the sensitivity records. These test-period comparisons diagnose rule sensitivity; they cannot choose a new primary rule without another evaluation sample.\n'
    (O/'selection_findings.tex').write_text(selectiontext)
    ctext='\\paragraph{Repaired controls.}\n'
    for panel in ['crsp5','crsp100']:
        c=controls[(controls.panel==panel)&(controls.alpha==.01)];w=within[(within.panel==panel)&(within.alpha==.01)];b=aug[aug.panel==panel]
        sig=c[(c.delta<0)&(c.p_holm<.05)];ws=w[(w.delta<0)&(w.p_holm<.05)];bs=b[(b.delta<0)&(b.p_holm<.05)]
        ctext+=f'In {panel.replace("crsp","")} dimensions, the targeted adjustments detect {len(sig)} improvements among nine filter/refit contrasts and {len(ws)} among seven innovation contrasts against common coordinate-GARCH-FHS. The broader 27-contrast comparison establishes {len(bs)} improvements over the original FHS. '
    ctext+='These separately adjusted comparisons answer different questions; targeted significance does not transfer automatically to the broader family. Full-covariance conditioning, alternative rules and comparisons with portfolio-GARCH-FHS remain in the machine-readable sensitivity records.\n'
    (O/'control_findings.tex').write_text(ctext)
    jtext='\\paragraph{Joint-event results.}\n'
    for panel in ['crsp5','crsp100']:
        z=joint[joint.panel==panel];b=z.loc[z.brier.idxmin()];sig=z[(z.delta<0)&(z.p_holm<.05)]
        jtext+=f'The observed event frequency is {z.observed_frequency.iloc[0]:.3f} in {panel.replace("crsp","")} dimensions. {lab(b.model)} has the lowest mean Brier score ({b.brier:.4f}); adjustment detects {len(sig)} improvements among eleven FHS contrasts. '
        worse=z[(z.delta>0)&(z.p_holm<.05)]
        if not worse.empty:jtext+='Adjusted deterioration is detected for '+', '.join(lab(r.model)+' ($p='+f'{r.p_holm:.4f}'+'$)' for r in worse.itertuples())+'. '
    jtext+='These results concern the declared co-exceedance event under the common filter, complementing density and portfolio-risk comparisons.\n'
    (O/'joint_findings.tex').write_text(jtext)
    base=(B/'main.tex').read_text();prefix=base[:base.index('\\input{expanded_methods.tex}')]
    prefix=prefix.replace('\\input{referee_round2/analysis/abstract.tex}','\\input{referee_round4/analysis/abstract.tex}')
    prefix=prefix.replace('The fitted joint parametric baselines are Gaussian and meta-$t$ laws, supplemented in the five-asset studies by a restricted Heston--Wishart stochastic-covariance model. The multivariate Student law supplies a controlled target and neural base distributions; GARCH-t supplies a portfolio-level conditional benchmark.','The fitted joint parametric baselines are Gaussian, shrinkage-Gaussian, multivariate Student and meta-$t$ laws, supplemented in the five-asset studies by a restricted Heston--Wishart stochastic-covariance model. GARCH-t supplies a portfolio-level conditional benchmark.')
    prefix=prefix.replace('and no primary CRSP improvement over FHS, neural or statistical, survives within-panel multiplicity correction.','and the CRSP conclusions concern the declared validation design and evaluated portfolio exposures.')
    prefix=prefix.replace('A Gaussian-order tail means', 'The CRSP application uses arithmetic returns: a long-only one-day loss is bounded by 100\\%, whereas the architectural tail statements concern idealized unbounded laws. The empirical task evaluates finite probability levels, not an infinite-loss tail index. A Gaussian-order tail means')
    conclusion=headline+' The main risk comparison uses the repaired selection rule and includes fitted multivariate Student and shrinkage-Gaussian benchmarks. '
    conclusion+=('The broader retrospective 28-model comparison establishes no FHS improvement after its 27-contrast correction. ' if broad.empty else 'The broader retrospective comparison and targeted control families have separately reported conclusions. ')
    if neural_winners.empty:conclusion+='No neural improvement survives the primary correction under the common one-percent validation rule. '
    if set(winners.model)=={'Student-MLE'} and set(winners.panel)=={'crsp100'}:
        st=p[(p.panel=='crsp100')&(p.model=='Student-MLE')].iloc[0]
        st_b=aug[(aug.panel=='crsp100')&(aug.model=='Student-MLE')].iloc[0]
        conclusion+=f'The Student contrast has adjusted $p={st.p_holm:.4f}$ in the sixteen-contrast family and $p={st_b.p_holm:.4f}$ in the broader family. '
    conclusion+='The portfolio decomposition associates the GARCH point-score change with variance conditioning; the primary adjustments do not resolve the GARCH improvements over FHS.\n\n'
    conclusion+='The checkpoint audit yields a practical requirement: include the structured base or identity transformation when validating a flexible residual model, and inspect the early training path. Otherwise, a reported likelihood-selected network can be worse on validation data than an available untrained or parametric state. The repaired results distinguish that failure from a lack of representational capacity.\n\n'
    zz=sel[(sel.panel=='crsp100')&(sel.model=='Flow-meta-t')].set_index('rule')
    conclusion+=f'For example, 100-stock Flow-meta-$t$ has mean FZ0 {zz.loc["fz1","fz0"]:.4f} under the primary risk rule and {zz.loc["density","fz0"]:.4f} under likelihood selection; the latter chooses the exact parametric base in all ten folds. Validation design therefore affects both the observed ranking and what part of a model receives credit.\n\n'
    wstatus=ROOT/'wishart_completion/analysis/final_status.json'
    if wstatus.exists():
        ws=json.loads(wstatus.read_text());assert ws['audit_complete']
        wr=syn.loc[('heston','Heston-Wishart')];er=syn.loc[('heston','Historical')]
        conclusion+=f'The Wishart evidence separates specification from computation. The restricted fit underestimates synthetic ES after refinement, with one-percent RMSE {wr.rmse:.3f} versus {er.rmse:.3f} for empirical ES. Its CRSP filter fails the declared numerical tolerance and supplies no admissible primary forecast comparison. These findings neither reject the Heston--Wishart family nor identify the incremental value of stochastic correlation. A broader specification and a converged filter would require a new evaluation.\n\n'
    conclusion+='The study compares parametric structure, empirical innovations and neural flexibility through distinct objectives. Correct specification, dependence representation, volatility conditioning and model selection all matter. Joint NLL, the declared downside-event score and the evaluated portfolio losses provide complementary evidence; none alone establishes a general ordering of multivariate model families.\n'
    (O/'conclusion.tex').write_text(conclusion)
    body=r'''\input{expanded_methods.tex}
\input{referee_round4/analysis/synthetic_narrative.tex}
\input{expanded_crsp_methods.tex}
\input{referee_round4/analysis/crsp_findings.tex}
\section{Density, selection, and multivariate controls}\label{sec:checkpoint-results}
\input{round4_methods.tex}
\input{referee_round4/analysis/selection_combined.tex}
\input{referee_round4/analysis/selection_findings.tex}
\input{referee_round4/analysis/density_combined.tex}
\input{referee_round4/analysis/density_findings.tex}
\FloatBarrier
\input{referee_round4/analysis/dense_crsp.tex}
\input{referee_round4/analysis/dense_heston.tex}
\input{referee_round4/analysis/coordinate_combined.tex}
\input{referee_round4/analysis/control_findings.tex}
\FloatBarrier
\input{referee_round4/analysis/joint_combined.tex}
\input{referee_round4/analysis/joint_findings.tex}
\FloatBarrier
\section{Implications and limitations}\label{sec:limitations}
The comparison concerns fixed architectures, budgets, annual universes and declared financial exposures. The repaired candidate set improves the implementation of validation, but does not optimize over every training procedure or demonstrate a population ranking of model families. The sparse one-percent validation tail remains a source of selection variation. Alternative levels, pooled portfolios and final states quantify sensitivity; a rule chosen after viewing their test performance requires independent confirmation.

Inference conditions on retained forecasts. The date bootstrap does not re-estimate models, regenerate scenario samples, remove nonstationarity, or validate every conditional-calibration restriction. Close observed rankings can depend on the finite scenario sample. The named Holm families in Table~\ref{tab:r4-families} have separate scopes. Non-rejection does not establish equivalence, and MCS retention does not mean that a model is better than every excluded model. Seed-score averaging measures typical fitted forecasts; a practitioner ensemble would require separate forecast aggregation and evaluation.

The 100-stock panels contain established large-cap universes, one optimization seed, and zero-filled missing allocations. This convention can suppress downside variation under distress-related missingness; the retained stress and complete-date checks cannot recover the absent returns. Equal-weight and concentrated portfolios cover a small set of projections. The new joint-event diagnostic broadens the dependence evidence but still measures one feature of the joint law.

Synthetic conclusions condition on ten estimation datasets and, for Heston, a finite jointly evolved variance bank and time discretization. Squared-error and synthetic log-density SEs are descriptive where heavy tails prevent finite-variance guarantees. The restricted Wishart fit and the CRSP numerical failure have different implications: the former evaluates a misspecified distribution, while the latter prevents a reliable forecasting comparison. Appendix~\ref{sec:wishart-numerics} reports the completed audit and stopping criterion. The CRSP implementation is excluded from primary scores, Holm families and MCS membership.

\section{Conclusion}
\input{referee_round4/analysis/conclusion.tex}
\appendix
\input{heston_appendix.tex}
\section{Inference, construction, and multiplicity audit}
\input{expanded_audit.tex}
\input{referee_round4/analysis/families.tex}
\FloatBarrier
\section{Synthetic density uncertainty}
The following tables separate fitting-dataset dispersion from uncertainty in the shared test sample. Their construction and heavy-tail qualification are in Section~\ref{sec:density-uncertainty}. Same-sample oracle densities remain distinct from population entropy; the Student target has population entropy 3.36070 and test-sample oracle NLL 3.35305.
\input{referee_round4/analysis/t_density.tex}
\input{referee_round4/analysis/metat_density.tex}
\input{referee_round4/analysis/heston_density.tex}
\FloatBarrier
\section{Paired error accounting and fitted parameters}
\input{referee_round4/analysis/t_decomposition.tex}
\input{referee_round4/analysis/metat_decomposition.tex}
\input{referee_round4/analysis/heston_decomposition.tex}
\FloatBarrier
\input{referee_round4/analysis/parameters.tex}
The meta-$t$ copula degrees of freedom average 7.35 in five dimensions and 29.27 in 100 dimensions. The latter comes from a staged profile with marginal fits and Kendall-based dependence held fixed; it should not be interpreted as a dimension-invariant structural tail parameter. Flow-t's original base scatter is identity, with learned degrees of freedom supplied in each selected checkpoint. Dense Flow-t additionally admits the fitted Student benchmark.
\FloatBarrier
\section{Secondary portfolio and calibration results}
\input{referee_round4/analysis/crsp5_secondary.tex}
\input{referee_round4/analysis/crsp100_secondary.tex}
\input{referee_round4/analysis/calibration.tex}
The source archive retains the full concentrated-portfolio, crash-exclusion, covariance-filter, checkpoint-rule and broad-family tables, as well as every underlying date-level record. The missing-return stress and complete-date calculations use unchanged portfolios and source masks; they qualify the construction rather than imputing unobserved returns.
\input{referee_round2/analysis/missing_table.tex}
\FloatBarrier
\section{Fitted Wishart specification}\label{sec:wishart-fit-details}
\input{wishart_methods.tex}
\input{wishart_appendix.tex}
\FloatBarrier
\section{Target-law leverage and reproducibility}
\input{referee_round2/analysis/leverage_table.tex}
\input{referee_round4/analysis/verification.tex}
\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
'''
    (ROOT/'main.tex').write_text(prefix+body)
    # Keep the historical provenance audit, update only its parameter description.
    audit=(B/'expanded_audit.tex').read_text();audit=audit.replace('Complete new parameters are supplied.','Complete new parameters are supplied.')
    (ROOT/'expanded_audit.tex').write_text(audit)
    refs=(B/'refs.bib').read_text()
    refs+=r'''
@article{liu1995t,
 author={Liu, Chuanhai and Rubin, Donald B.},
 title={{ML} estimation of the {$t$} distribution using {EM} and its extensions, {ECM} and {ECME}},
 journal={Statistica Sinica}, volume={5}, number={1}, pages={19--39}, year={1995},
 url={https://www3.stat.sinica.edu.tw/statistica/j5n1/j5n12/j5n12.htm}}
@article{ledoit2004,
 author={Ledoit, Olivier and Wolf, Michael},
 title={A well-conditioned estimator for large-dimensional covariance matrices},
 journal={Journal of Multivariate Analysis}, volume={88}, number={2}, pages={365--411}, year={2004},
 doi={10.1016/S0047-259X(03)00096-4}}
'''
    (ROOT/'refs.bib').write_text(refs)
    print('INTEGRATED repaired manuscript; abstract words',len(abstract.split()))

if __name__=='__main__':main()

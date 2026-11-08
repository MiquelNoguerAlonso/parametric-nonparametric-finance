"""Write the referee-round narrative from completed, checked outputs."""
from pathlib import Path
import json,re
import pandas as pd,numpy as np
from revision_analysis import label
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round2/analysis'
def nm(x):return label(x).replace('meta-t','meta-$t$')
def criterion(x):return {'fz1':'one-percent FZ0','fz2p5':'2.5-percent FZ0','fz5':'five-percent FZ0','pooled1':'pooled one-percent FZ0','final':'the final checkpoint','density':'native likelihood validation'}[x]
def synthetic():
 s=pd.read_csv(O/'synthetic_summary.csv');h=s[(s.study=='heston')&(s.alpha==.01)].set_index('model');pair=json.loads((O/'vae_h_vs_g.json').read_text());lev=json.loads((O/'leverage.json').read_text());b=lev['rows'][0]['metrics']['0.01'];z=lev['rows'][1]['metrics']['0.01'];c=lev['rows'][2]['metrics']['0.01']
 text=r'''\subsection{Controlled findings: specification before architecture}
Both heavy-tailed targets belong to the fitted meta-$t$ family: the bivariate $t_3$ is its homogeneous-margin special case. Their favorable parametric results partly reflect correct specification and do not establish robustness to an unknown law. The residual meta-$t$ flow also contains the target at its identity residual map, so its experiment is not an out-of-family validation. These are controlled estimation comparisons within a favorable model family.

'''
 for law in ['t','metat']:
  g=s[(s.study==law)&(s.alpha==.01)].set_index('model');m=g.loc['Meta-t'];text+=f"On the {'Student' if law=='t' else 'heterogeneous meta-$t$'} target, plain meta-$t$ has one-percent RMSE {m.rmse:.3f}, versus {m.empirical_rmse:.3f} for exact empirical-training ES. The paired MSE difference is {m.delta_mse:.3f}, with dataset-level SE {m.paired_se:.3f}. "
  if law=='metat':
   f=g.loc['Flow-meta-t'];text+=f"Flow-meta-$t$ has RMSE {f.rmse:.3f}; its paired difference {f.delta_mse:.3f} has SE {f.paired_se:.3f}. These point estimates support the construction in its correctly specified setting, without certifying a general flexible-model advantage. "
  text+='\n\\input{referee_round2/analysis/'+law+'_risk_table.tex}\n\n'
 text+=r'\FloatBarrier'+'\n'+r'\paragraph{Misspecified Heston controls.}'+'\n'
 best=h.rmse.idxmin();emp=h.empirical_rmse.iloc[0];best_fit=h.drop('Historical').rmse.idxmin()
 text+=f"We now fit plain meta-$t$ to Heston as well, giving the same parametric family a role in every controlled law. It is misspecified here and has RMSE {h.loc['Meta-t','rmse']:.3f}. VAE-tlatent and WGAN-i are also added; their earlier omission was a design choice, not a mathematical or computational restriction. Their RMSEs are {h.loc['VAE-t','rmse']:.3f} and {h.loc['WGAN-i','rmse']:.3f}. "
 text+=(f"Under the primary risk-selection rule, the exact empirical estimator has RMSE {emp:.3f}, below every parametric or neural model in this experiment. " if emp<h.drop('Historical').rmse.min() else f"The lowest fitted-model RMSE is {h.loc[best_fit,'rmse']:.3f} for {nm(best_fit)}, compared with {emp:.3f} for the exact empirical estimator. ")
 text+="No correctly specified Heston likelihood or particle-filter benchmark is fitted; adding misspecified meta-$t$ supplies a common family across targets, not an oracle.\n\n"
 text+=f"VAE-H and VAE-G have observed RMSEs {h.loc['VAE-H','rmse']:.3f} and {h.loc['VAE-G','rmse']:.3f}. The direct paired H-minus-G MSE difference is {pair['delta_mse']:.4f}, with SE {pair['paired_se']:.4f}; there is no detectable difference in these ten datasets. Neither a benefit nor equivalence of the decoder tail choices is established at one percent. WGAN-t has RMSE {h.loc['WGAN-t','rmse']:.3f}; its polynomial asymptote differs from the ideal Heston exponential tail, but that theorem alone does not predict a fixed-level RMSE or establish the cause of this error.\n"
 text+=r'\input{referee_round2/analysis/heston_risk_table.tex}'+'\n\n'
 text+=r'\paragraph{Leverage within a joint weekly window.}'+'\n'
 text+=f"A static distribution on the ordered 25-dimensional window can encode dependence between today's return and tomorrow's volatility, including leverage. It does not by itself specify a recursive state-conditioned process or prove consistency of generated paths beyond that window. The within-window leverage statistic is explicitly reported for every Heston model in Table~\\ref{{tab:heston-expanded-dependence}}.\n\nA new paired simulation uses one million paths from the current joint-bank target. With cross-asset price-shock correlation held at 0.549, removing leverage lowers one-percent weekly ES from {b['es']:.3f} to {z['es']:.3f}, a reduction of {100*(b['es']-z['es'])/b['es']:.2f}\\% of the leverage-model ES; the paired difference is {z['difference_from_leverage']:.3f}, with SE {z['paired_batch_se']:.3f}. Holding the orthogonal-shock correlation at 0.5 instead gives ES {c['es']:.3f}, but also changes total price-shock correlation and therefore mixes two effects. Table~\\ref{{tab:leverage-ablation}} makes that distinction explicit. The earlier 5.36 versus 4.78 figures are not reused as estimates for this different, jointly initialized numerical target.\n\n"
 text+=r'''The 0.1\% columns use one million generated scenarios per fit; nonanalytic references use four million. Batch diagnostics quantify evaluation noise separately from estimation-dataset uncertainty. Density values are separated from risk tables and compared under matched checkpoint criteria in Section~\ref{sec:checkpoint-results}. A best observed row selected from these summaries is not a multiplicity-adjusted winner certificate.
\begin{figure}[htbp]\centering\includegraphics[width=0.96\linewidth]{figures/round2_paired_mse.pdf}\caption{Paired one-percent ES MSE differences from exact empirical-training ES, with one dataset-level SE. The two heavy-tailed targets are correctly specified for meta-$t$; no fitted family is assumed correctly specified for Heston. The Heston panel includes the three added controls.}\label{fig:paired-expanded}\end{figure}
'''
 (O/'synthetic_narrative.tex').write_text(text)

def crsp():
 for panel in ['crsp5','crsp100']:
  s=pd.read_csv(O/(panel+'_summary.csv'));s=s[s.alpha.eq(.01)].set_index('model');b=s.fz0.idxmin();f=s.loc['Historical'];n=s[s.index.str.startswith(('Flow','VAE','WGAN'))];best=n.fz0.idxmin();mc=pd.read_csv(O/(panel+'_mcs.csv'));mc=mc[(mc.alpha==.01)&(mc.block==20)];excluded=mc[mc.p_mcs<.05].model.tolist()
  t=r'\subsection{'+('Five-basket findings' if panel=='crsp5' else '100-stock findings')+'}\n'
  t+=f"The lowest observed one-percent score is {nm(b)} at {s.loc[b,'fz0']:.4f}, against {f.fz0:.4f} for FHS. Its difference is {s.loc[b,'delta_fhs']:.4f}, with basic interval $[{s.loc[b,'low']:.4f},{s.loc[b,'high']:.4f}]$ and fourteen-contrast Holm $p={s.loc[b,'p_holm']:.4f}$. The best primary neural mean is {nm(best)} at {n.loc[best,'fz0']:.4f}; no primary neural improvement over FHS survives the augmented within-panel correction. These comparisons concern the frozen one-percent selection rule. Alternative-rule results are reported separately.\n\n"
  if panel=='crsp5':
   a=s.loc['Flow-t'];v=s.loc['Flow-meta-t'];t+=f"Flow-t and Flow-meta-$t$ retain unadjusted FHS improvements, with $p={a.p:.4f}$ and {v.p:.4f}, but their augmented Holm values are {a.p_holm:.4f} and {v.p_holm:.4f}.\n\n"
  cal=pd.read_csv(O/(panel+'_calibration.csv'));c=cal[(cal.model=='Meta-t')&(cal.alpha==.01)].sort_values('seed');m=s.loc['Meta-t'];t+=f"Plain meta-$t$ produces {' and '.join(str(int(x)) for x in c.hits)} one-percent exceedances against 25.14 expected, with Kupiec $p$-values {' and '.join(f'{x:.4f}' for x in c.kupiec_p)} and normalized ES moment {m.es_moment:.4f}. Its fitted likelihood is assessed separately against matched likelihood-selected neural states; a ranking among risk-selected checkpoints is not presented as a family-level density comparison.\n\n"
  t+=f"The 95\\% 20-day-block MCS retains {15-len(excluded)} of fifteen specifications. "+('It excludes '+', '.join(nm(x) for x in excluded)+'. ' if excluded else 'All fifteen remain. ')+r'Membership does not certify calibration; 10- and 40-day blocks and all seed-specific calibration diagnostics are supplied.'+'\n'
  if panel=='crsp100':t+=r'\par Missing slots have mean weight 1.549\% and maximum weight 7\%. Zero imputation treats these slots as riskless for that day and can suppress losses and estimated volatility when missingness reflects distress. This is a downward-risk concern, not evidence that missing outcomes were benign. The severe fixed-forecast stress and 1,003 complete-date diagnostic are retained; the convention does not identify the sign or magnitude of every possible missing-data bias.'+'\n'
  (O/(panel+'_findings.tex')).write_text(t)
 d=pd.read_csv(O/'filter_decomposition.csv');t=r'\subsection{Matched portfolio filters and residual laws}'+'\n'
 for panel in ['crsp5','crsp100']:
  z=d[d.panel.eq(panel)].set_index('contrast');a=z.loc['Variance filter'];e=z.loc['EVT tail'];t+=f"For {'five baskets' if panel=='crsp5' else '100 stocks'}, replacing the matched-location portfolio EWMA variance recursion by GARCH changes mean FZ0 by {a.delta:.4f}, with interval $[{a.low:.4f},{a.high:.4f}]$. Replacing only the GARCH empirical residual law by EVT changes it by {e.delta:.4f}, with interval $[{e.low:.4f},{e.high:.4f}]$. Thus most of the observed total improvement is associated with variance filtering, and no incremental EVT advantage over GARCH-FHS is detected here. The variance-filter contrasts are significant only unadjusted; their six-contrast Holm values are {a.p_holm:.3f}.\n\\input{{referee_round2/analysis/{panel}_filter_decomposition.tex}}\n\n"
 t+=r'The decomposition is exact for the saved daily scores. It does not show that a particular volatility estimator is universally superior, and it does not attribute the whole GARCH-EVT/FHS difference to extreme-value modeling. In the augmented 100-stock comparison GARCH-FHS, rather than GARCH-EVT, has the lowest observed mean.'+'\n';(O/'matched_filter_findings.tex').write_text(t)

def sensitivity():
 s=pd.read_csv(O/'selection_summary.csv');d=pd.read_csv(O/'matched_density.csv');ch=pd.read_csv(O/'checkpoint_changes.csv');inf=pd.read_csv(O/'selection_inference.csv');text=''
 for study in ['crsp5','crsp100']:
  g=s[s.study.eq(study)].dropna(subset=['fz0']);best=g.loc[g.fz0.idxmin()];orig=g[g.criterion.eq('fz1')].set_index('model');final=g[g.criterion.eq('final')].set_index('model');change=final.fz0-orig.fz0
  c=ch[ch.study.eq(study)&ch.criterion.isin(['fz2p5','fz5','pooled1'])];sig=inf[(inf.study==study)&(inf.delta<0)&(inf.p_holm<.05)]
  text+=f"In {'five baskets' if study=='crsp5' else '100 stocks'}, {100*c.changed.mean():.1f}\\% of model/fold/alternative-risk-rule choices change the selected checkpoint relative to one-percent FZ0. Final-minus-selected mean-score changes range from {change.min():.4f} to {change.max():.4f} across neural specifications. The lowest observed rule/model mean is {nm(best.model)} under {criterion(best.criterion)} at {best.fz0:.4f}. "+('No improvement survives the exploratory forty-contrast rule/model Holm correction.' if len(sig)==0 else 'The following exploratory rule/model contrasts survive their forty-contrast Holm correction: '+', '.join(nm(x.split('::')[0])+' / '+criterion(x.split('::')[1]) for x in sig.model)+'.')+' The sensitivity table reports every tested rule, including adverse changes.\n\n'
 h=d[d.study.eq('heston')].set_index('model');syn=pd.read_csv(O/'synthetic_summary.csv');gauss=syn[(syn.study=='heston')&(syn.model=='Gaussian')&(syn.alpha==.01)].nll.iloc[0]
 text+=f"For Heston, Flow-G NLL is {h.loc['Flow-G','fz1']:.3f} under risk selection and {h.loc['Flow-G','density']:.3f} under native likelihood-validation selection; Flow-L changes from {h.loc['Flow-L','fz1']:.3f} to {h.loc['Flow-L','density']:.3f}. The fitted full-covariance Gaussian is {gauss:.3f}. This matched comparison directly exposes selection effects and avoids treating risk-selected values as likelihood-optimal family comparisons. The native criterion still searches only the saved training trajectory, at the stated checkpoint spacing and capacity.\n\n"
 hs=s[s.study.eq('heston')].dropna(subset=['rmse']);hb=hs.loc[hs.rmse.idxmin()];mt=syn[(syn.study=='heston')&(syn.model=='Meta-t')&(syn.alpha==.01)].iloc[0]
 text+=f"Checkpoint sensitivity also changes the Heston risk comparison. The lowest observed secondary RMSE is {hb.rmse:.3f} for {nm(hb.model)} under {criterion(hb.criterion)}, versus exact empirical ES {mt.empirical_rmse:.3f}. Its paired MSE difference from empirical ES is {hb.paired_mse_from_empirical:.3f}, with dataset-level SE {hb.paired_empirical_se:.3f}. Thus the primary empirical-estimator lead is not invariant to selection. This is an exploratory choice among {len(hs)} rule/model combinations, not a confirmed winner.\n\nThe added plain meta-$t$ fit has exact Heston NLL {mt.nll:.3f}, lower than the likelihood-selected flows in this implementation, yet one-percent ES RMSE {mt.rmse:.3f}. This is a concrete density/risk discrepancy within fitted specifications. Higher VAE IWAE-NLL estimates do not establish their true NLL ordering against meta-$t$, because the expected importance-sampling gap works against the VAE.\n\n"
 text+=r'Across synthetic laws, Tables~\ref{tab:t-selection}--\ref{tab:heston-selection} report the same criterion sensitivity with paired MSE changes and SE in the supplied CSV. The empirical baseline and fitted statistical laws do not change when a neural checkpoint rule changes. The alternative rules were specified retrospectively, and the lowest observed secondary result is not a newly validated deployment policy. Joint density, portfolio score, and conditional calibration must be evaluated on their own targets; the paper makes no empirical claim about an unattained likelihood optimum.'+'\n'
 (O/'selection_findings.tex').write_text(text)

def abstract():
 s=pd.read_csv(O/'synthetic_summary.csv');h=s[(s.study=='heston')&(s.alpha==.01)];emp=h.empirical_rmse.iloc[0];best=h[h.model!='Historical'].rmse.min()
 text='''No tested neural model improves on filtered historical simulation after within-panel multiplicity correction under the primary validation rule in five- and 100-dimensional CRSP forecasts. GARCH-filtered statistical benchmarks have the lowest observed scores. Matched residual-law controls associate most of their advantage with volatility filtering and detect no incremental EVT gain over empirical residuals. We distinguish architectural tail restrictions from finite-sample risk performance. Shared elliptical Student noise imposes a common nondegenerate loss-tail index; independent coordinates permit different directional structures. A residual meta-t flow preserves prescribed marginal indices while changing dependence. Ten independent datasets per synthetic law show favorable meta-t estimation when the true law belongs to that family; these wins partly reflect correct specification. '''
 text+=('For primary Heston checkpoints, empirical ES leads; alternative selection can reverse that ranking. ' if emp<best else 'Heston supplies a misspecified-family comparison. ')
 text+='''Matched training trajectories compare alternative risk-selection rules, likelihood selection and final checkpoints. A paired Heston ablation isolates leverage at fixed price-shock correlation. The evidence favors strong conditional benchmarks and explicit validation uncertainty; it does not establish neural or tail-family dominance.'''
 (O/'abstract.tex').write_text(text+'\n');print('Abstract words',len(text.split()))

def verification():
 files=[ROOT/'results/mathematical_checks.json',ROOT/'results/output_checks.json',ROOT/'results/revision/revision_checks.json',ROOT/'expansion/checks_preflight.json',ROOT/'expansion/checks_outputs.json',ROOT/'expansion/analysis/diagnostic_checks.json',O/'round2_checks.json'];counts=[len(json.loads(p.read_text())) for p in files];assert all(all(x['passed'] for x in json.loads(p.read_text())) for p in files)
 text=r'\section{Reproducibility and executed verification}'+'\n'
 text+=f"The primary expanded design now has 730 model evaluations and 520 fresh neural fits, including the thirty added Heston evaluations (twenty neural fits). There are 100 Student, 110 heterogeneous meta-$t$, and 120 Heston evaluations; 200 five-basket, 100 100-stock, and 100 full-covariance five-basket evaluations. The original 380 evaluations and 272 neural fits remain archived. The forty-one rejected cold-start pilot evaluations remain excluded.\n\nThe matched-selection study replays 380 training trajectories and records 2,150 rule/model evaluations, reusing identical selected states. These are not independent estimation datasets or 2,150 new optimization runs. The comparison includes all ten datasets per controlled law and seed-zero annual folds in both CRSP panels; covariance-filter refits are outside this sensitivity. All primary replay parameters are checked against the retained states. Added Heston dependence diagnostics complete 330 ancillary model evaluations. The leverage ablation has one million coupled paths per setting.\n\nAll {sum(counts)} recorded checks pass: the previous {sum(counts[:-1])} plus {counts[-1]} checks covering model additions, matched replay, validation minima, final states, augmented forecast accounting, FZ0 unit conversion, shared GARCH states, the exact filter decomposition, correlation bounds, and cached-draw identity. These checks verify their stated contracts, not universal predictive superiority.\n\n"
 text+=r'''The entry point is \path{code/run_all.py}; \path{code/run_round2.py} exposes this revision directly. Output mode reconstructs the tables and findings from retained results. Experiment mode completes the fixed additions and matched replays, reusing completed records. All protocols, source checks, parameters, selected states, validation traces, forecast files, reference samples, and review responses are included. Raw licensed CRSP extracts are not redistributed. The public author-uploaded portfolio-paper copy used for the single application citation was independently retrieved; its printed date is December 19, 2025. The unchanged established-distribution attribution does not rely on SSRN accessibility.'''
 (O/'verification.tex').write_text(text+'\n')

def manuscript():
 (O/'filter_findings.tex').write_text((ROOT/'expansion/analysis/filter_findings.tex').read_text().split('\\begin{figure}')[0])
 p=ROOT/'referee_round2/baseline/main_33pages.tex';s=p.read_text().split('\\input{expanded_methods.tex}')[0].replace('expansion/analysis/abstract.tex','referee_round2/analysis/abstract.tex').replace('with common risk-based checkpoint selection, dynamic benchmarks, coherent inference, and explicit calibration checks.','with matched checkpoint and filter controls, coherent inference, and explicit calibration checks.')
 s=s.replace(r'\emergencystretch=1em',r'''\emergencystretch=1em
\setcounter{topnumber}{5}\setcounter{bottomnumber}{4}\setcounter{totalnumber}{8}
\renewcommand{\topfraction}{0.95}\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}\renewcommand{\floatpagefraction}{0.75}''')
 s+=r'''\input{expanded_methods.tex}
\input{referee_round2/analysis/synthetic_narrative.tex}
\input{expanded_crsp_methods.tex}
\input{referee_round2/analysis/crsp5_primary.tex}
\input{referee_round2/analysis/crsp5_findings.tex}
\input{referee_round2/analysis/crsp100_primary.tex}
\input{referee_round2/analysis/crsp100_findings.tex}
\begin{figure}[htbp]\centering\includegraphics[width=0.98\linewidth]{figures/round2_annual.pdf}\caption{Annual one-percent equal-portfolio score differences from FHS for selected neural models and the matched filter controls. FHS is the zero line with circle markers; distinct markers distinguish the two VAE curves. All fifteen model histories are supplied in the annual CSVs.}\label{fig:annual-expanded}\end{figure}
\input{expansion/analysis/filter_table.tex}
\input{referee_round2/analysis/filter_findings.tex}
\input{referee_round2/analysis/matched_filter_findings.tex}
\FloatBarrier
\section{Checkpoint selection and likelihood assessment}\label{sec:checkpoint-results}
\input{round2_methods.tex}
\input{referee_round2/analysis/selection_findings.tex}
\input{referee_round2/analysis/crsp5_selection.tex}
\input{referee_round2/analysis/crsp100_selection.tex}
\FloatBarrier
\section{Implications and limitations}
The controlled heavy-tailed experiments favor a correctly specified parametric family. Their interpretation changes when that family is fitted to Heston, and they do not establish robustness across unknown laws. The direct paired VAE comparison detects no finite-level decoder-tail advantage. Tail-index results describe what a fixed architecture can represent asymptotically; they do not determine estimation error at one percent.

The matched portfolio-filter controls are consequential. Estimated GARCH conditioning explains most of the observed statistical-benchmark lead, whereas changing its empirical residual tail to EVT provides no detectable incremental gain in these panels. The full-covariance EWMA refits address a separate conditioning choice. Neither experiment proves a universal filter ranking.

Common checkpoint selection leaves substantial uncertainty when validation has few exceedances. Larger selection tail probabilities, pooled portfolios, and final states reveal sensitivity without supplying independent validation observations. The native likelihood-selected comparisons remove the earlier ambiguity between risk-selected NLL and family-level density fit; even these states are not global optima. Architectures, compute budgets, estimation objectives, and capacity remain fixed implementations rather than exhaustive family searches. The alternative rules and augmented benchmarks were chosen retrospectively.

The 100-stock study has one optimization seed, established large-cap universes, and three evaluation portfolios. Missing-slot zero returns can suppress downside variation, particularly under distress-related missingness. Forecast-conditional resampling, unadjusted calibration diagnostics, and the supplied stress checks do not identify all missing-data effects, remove structural change, or account for model re-estimation in every bootstrap draw. Ten synthetic datasets still leave heavy-tailed squared-error uncertainty imprecise. Heston results condition on a finite joint variance bank and a finite-step simulator. Joint weekly generators can encode within-window leverage but are not validated recursive conditional processes.

\section{Conclusion}
No primary neural risk-score improvement over FHS survives the stated multiplicity control. The strongest observed CRSP benchmarks share estimated portfolio-GARCH conditioning, and empirical residuals are competitive with EVT under that common filter. Meta-$t$ advantages on correctly specified laws and residual-flow flexibility are useful controlled results, not general dominance claims. Architecture, specification, selection, density assessment and calibration require separate evidence.

\appendix
\input{heston_appendix.tex}
\section{Inference and data-conditioning audit}
\input{expanded_audit.tex}
The augmented comparison adds GARCH-FHS and portfolio-EWMA-FHS to the previous thirteen specifications. Fourteen-contrast Holm results and fifteen-model MCS results supersede the earlier primary tables; the twelve-contrast arithmetic remains available for audit. The separate matched-filter attribution has six exploratory contrasts, and each CRSP checkpoint sensitivity has forty neural rule/model contrasts. These families are reported explicitly and do not imply one global multiplicity guarantee across the whole retrospective study.
\section{Density, tail precision, and error decomposition}
\input{referee_round2/analysis/density_tables.tex}
\input{results/revision/oracle_table.tex}
\input{referee_round2/analysis/mc_table.tex}
\input{referee_round2/analysis/t_decomposition.tex}
\input{referee_round2/analysis/metat_decomposition.tex}
\input{referee_round2/analysis/heston_decomposition.tex}
\FloatBarrier
\section{Additional forecast and calibration diagnostics}
\input{referee_round2/analysis/crsp5_secondary.tex}
\input{referee_round2/analysis/crsp100_secondary.tex}
\input{referee_round2/analysis/crsp5_calibration.tex}
\input{referee_round2/analysis/crsp100_calibration.tex}
\input{referee_round2/analysis/crsp5_robustness.tex}
\input{referee_round2/analysis/crsp100_robustness.tex}
\input{expansion/analysis/parameter_table.tex}
\input{referee_round2/analysis/crsp5_concentrated.tex}
\input{referee_round2/analysis/crsp100_concentrated.tex}
\input{referee_round2/analysis/missing_table.tex}
\FloatBarrier
\section{Synthetic checkpoint sensitivity}
\input{referee_round2/analysis/t_selection.tex}
\input{referee_round2/analysis/metat_selection.tex}
\input{referee_round2/analysis/heston_selection.tex}
\FloatBarrier
\section{Within-window leverage and radial-tail diagnostics}
\input{referee_round2/analysis/leverage_table.tex}
\input{referee_round2/analysis/heston_dependence_table.tex}
The radial-splice sensitivity retains ten independent estimation datasets per law and 100,000 paired before/after scenarios per fit. It concerns one-percent risk, with the explicit assumption that generated directions are adequate. It is separate from the million-scenario primary tail evaluations. Adverse repairs and paired squared-error statistics remain in the supplied CSV.
\input{expansion/analysis/radial_table.tex}
\FloatBarrier
\input{referee_round2/analysis/verification.tex}
\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
'''
 (ROOT/'main.tex').write_text(s)

def main():
 synthetic();crsp();sensitivity();abstract();verification();manuscript();print('Round-two manuscript generated')
if __name__=='__main__':main()

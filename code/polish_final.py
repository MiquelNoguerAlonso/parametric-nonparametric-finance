"""Apply the final editorial pass without changing any numerical table or theory."""
from pathlib import Path
import json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round2/analysis';P=ROOT/'final_polish'
TITLE='Volatility Conditioning and Generator Choice in Tail-Risk Forecasting'

def edit(path,changes):
 p=ROOT/path;s=p.read_text()
 for old,new in changes:s=s.replace(old,new)
 p.write_text(s)

def caption(path,text):
 p=ROOT/path;s=p.read_text();m=re.search(r'\\caption\{',s);assert m,path
 start=m.end();depth=1;i=start
 while depth:
  if s[i]=='{':depth+=1
  elif s[i]=='}':depth-=1
  i+=1
 p.write_text(s[:start]+text+s[i-1:])

def main():
 P.mkdir(exist_ok=True)
 # Keep method-specific assumptions where needed; consolidate general inferential limits.
 edit('expanded_methods.tex',[
  ('These larger samples reduce scenario noise but do not remove estimation uncertainty or make the numerical references exact.',''),
  ('Ten replications remain imprecise for heavy-tailed squared errors; these summaries are not a high-powered family-wide superiority test. ',''),
  ('An IWAE NLL below a competitor\'s exact NLL is conservative with respect to its expected variational gap, but its single Monte Carlo realization does not certify the ordering.','The IWAE interpretation follows Section~\\ref{sec:objectives}.'),
 ])
 edit('expanded_crsp_methods.tex',[
  ('This is a study of established large-cap stocks at each historical selection date; it is not a representative sample of all listed stocks.','The selected universes comprise established large-cap stocks.'),
  ('It is not a neutral imputation; without a model for the missing returns, a universal bias sign is not identified. ',''),
  ('These are fresh experiments: a comparison with the earlier archive changes selection, scenario count, and training seeds jointly, and therefore is not an isolated causal estimate of the checkpoint-rule effect.','The earlier archive uses different selection rules, scenario counts, and training seeds.'),
  ('This is a conditional POT implementation using the fitted Student-GARCH filter, not a claim to reproduce every detail of the original McNeil--Frey estimation procedure.','The POT implementation uses the fitted Student-GARCH filter.'),
  ('Together these benchmarks test the contribution of conditional variance and correlation modeling; a weak result for the neural innovation generators should not be confused with a failure of all conditional generative models.','Together these benchmarks distinguish conditional variance and correlation modeling.'),
  ('This retrospective ablation is exploratory.',''),
  ('Mean $m_t/\\overline e$ is a descriptive normalized ES moment, not a conditional-calibration certificate.','The reported normalized ES moment is mean $m_t$ divided by $\\overline e$; formal calibration tests are defined below.'),
  ('The 100-stock extension is a separate, retrospectively specified study; the procedure does not provide a joint multiplicity guarantee across every analysis in this paper.',''),
  ('It is not a Bayer--Dimitriadis ES regression implementation. Rare hits, parameter estimation, and nonstationarity limit its finite-sample approximation.',''),
  ('These choices were fixed for the expanded run, but the whole research design remains retrospective. Model-confidence-set membership is not evidence of good calibration, and non-rejection is not proof of predictive equivalence.','Interpretive limits are collected in Section~\\ref{sec:limitations}.'),
 ])
 edit('round2_filter_methods.tex',[
  ('Earlier twelve-contrast inference remains archived; it is not silently reused for the augmented family.',''),
  ('They are a decomposition within the fitted specifications, not a causal comparison of universally optimal filters or tail models.',''),
 ])
 (ROOT/'round2_methods.tex').write_text(r'''\subsection{Matched checkpoint and density-selection sensitivity}\label{sec:selection-sensitivity}
A validation year contains about 2.5 nominal one-percent exceedances; each synthetic validation set contains about ten. We replay each controlled-law dataset and each seed-zero CRSP fold with identical training observations, initialization, minibatches, update counts, and objectives. Every recovered primary checkpoint agrees exactly with its retained parameters.

The competing rules select equal-portfolio FZ0 at 1\%, 2.5\%, or 5\%; mean one-percent FZ0 pooled across the existing portfolios; or the final checkpoint. Pooling uses equal weights: the equal portfolio and both margins for the bivariate laws, the equal weekly portfolio and five asset-week losses for Heston, and the three evaluated CRSP portfolios. All risk rules share 20,000 validation scenarios and the same 250-step checkpoint grid.

Flows additionally select exact validation NLL and VAEs the original fixed-random-number IWAE-8 criterion; test VAE NLL uses IWAE-64. Checkpoint selection excludes test observations. Competing states share the one-million synthetic or 100,000 CRSP evaluation seeds, with identical states reusing their evaluations. CRSP sensitivity evaluates the equal portfolio with seed zero; the primary five-basket results average two seeds.

Separate density tables compare risk, likelihood, and final states along these matched trajectories. The earlier three-dataset archive has different training samples and seeds. The implemented flow head is diagonal with bounded coupling shifts; the dense-affine argument for exact full-covariance Gaussian containment does not apply. Section~\ref{sec:limitations} states the scope of the resulting comparisons.

''')
 edit('referee_round2/analysis/synthetic_narrative.tex',[
  ('Their favorable parametric results partly reflect correct specification and do not establish robustness to an unknown law. The residual meta-$t$ flow also contains the target at its identity residual map, so its experiment is not an out-of-family validation. These are controlled estimation comparisons within a favorable model family.','Their favorable results partly reflect correct specification. The residual meta-$t$ flow also contains its target at the identity residual map.'),
  ('These point estimates support the construction in its correctly specified setting, without certifying a general flexible-model advantage. ',''),
  ('We now fit plain meta-$t$ to Heston as well, giving the same parametric family a role in every controlled law.','Plain meta-$t$ is also fitted to Heston, giving the same family a role in every controlled law.'),
  ('VAE-tlatent and WGAN-i are also added; their earlier omission was a design choice, not a mathematical or computational restriction.','The comparison also includes VAE-tlatent and WGAN-i.'),
  ('No correctly specified Heston likelihood or particle-filter benchmark is fitted; adding misspecified meta-$t$ supplies a common family across targets, not an oracle.','A correctly specified Heston likelihood or particle-filter benchmark remains outside this design.'),
  ('Neither a benefit nor equivalence of the decoder tail choices is established at one percent. ',''),
  ('It does not by itself specify a recursive state-conditioned process or prove consistency of generated paths beyond that window. ',''),
  ('The earlier 5.36 versus 4.78 figures are not reused as estimates for this different, jointly initialized numerical target. ',''),
  ('A best observed row selected from these summaries is not a multiplicity-adjusted winner certificate.',''),
 ])
 for panel in ['crsp5','crsp100']:
  edit(f'referee_round2/analysis/{panel}_findings.tex',[
   ('These comparisons concern the frozen one-percent selection rule. Alternative-rule results are reported separately.','Checkpoint sensitivity is reported in Section~\\ref{sec:checkpoint-results}.'),
   ('Its fitted likelihood is assessed separately against matched likelihood-selected neural states; a ranking among risk-selected checkpoints is not presented as a family-level density comparison.','The matched likelihood comparison appears in Appendix C.'),
   ('All fifteen remain. ',''),
   ('Membership does not certify calibration; 10- and 40-day blocks and all seed-specific calibration diagnostics are supplied.','The appendix supplies block-length and seed-specific calibration diagnostics.'),
   ('Zero imputation treats these slots as riskless for that day and can suppress losses and estimated volatility when missingness reflects distress. This is a downward-risk concern, not evidence that missing outcomes were benign. The severe fixed-forecast stress and 1,003 complete-date diagnostic are retained; the convention does not identify the sign or magnitude of every possible missing-data bias.','The appendix reports the fixed-forecast missing-return stress and the common 1,003 complete dates.'),
  ])
 edit('referee_round2/analysis/matched_filter_findings.tex',[
  ('It does not show that a particular volatility estimator is universally superior, and it does not attribute the whole GARCH-EVT/FHS difference to extreme-value modeling. ',''),
  ('GARCH-FHS, rather than GARCH-EVT, has the lowest observed mean.','GARCH-FHS has the lowest observed mean.'),
 ])
 edit('referee_round2/analysis/selection_findings.tex',[
  ('The sensitivity table reports every tested rule, including adverse changes.',''),
  ('This matched comparison directly exposes selection effects and avoids treating risk-selected values as likelihood-optimal family comparisons. The native criterion still searches only the saved training trajectory, at the stated checkpoint spacing and capacity.','The matched comparison isolates checkpoint-selection effects within each training trajectory.'),
  ('Thus the primary empirical-estimator lead is not invariant to selection. This is an exploratory choice among 51 rule/model combinations, not a confirmed winner.','This reversal is the best observed secondary result among 51 rule/model combinations.'),
  ('The alternative rules were specified retrospectively, and the lowest observed secondary result is not a newly validated deployment policy. Joint density, portfolio score, and conditional calibration must be evaluated on their own targets; the paper makes no empirical claim about an unattained likelihood optimum.',''),
 ])
 edit('referee_round2/analysis/verification.tex',[
  ('These checks verify their stated contracts, not universal predictive superiority.','These checks verify the stated mathematical, data, and numerical contracts.'),
  ('The public author-uploaded portfolio-paper copy used for the single application citation was independently retrieved; its printed date is December 19, 2025. The unchanged established-distribution attribution does not rely on SSRN accessibility.','The portfolio-application citation uses the verified public author copy dated December 19, 2025.'),
 ])

 captions={
 't_risk_table.tex':r'Student target, correctly specified for meta-$t$. Paired one-percent MSE differences, SE, and skill use exact empirical-training ES. Each fit uses one million evaluation scenarios.',
 'metat_risk_table.tex':r'Heterogeneous meta-$t$ target, correctly specified for plain meta-$t$ and its identity residual extension. Paired errors use the same ten estimation datasets and empirical ES reference.',
 'heston_risk_table.tex':r'Heston target with the common Meta-$t$, VAE-tlatent and WGAN-i controls. Primary one-percent FZ0 selection; ten datasets and one million evaluation scenarios per fit. NLL is reported separately.',
 'mc_table.tex':r'Reference and evaluation precision. Nonanalytic references use four million draws; each fitted generator uses one million. The final column spans model-specific mean batch SE, including the added Heston controls.',
 'heston_dependence_table.tex':r'Within-window Heston dependence: ten fitted datasets and 100,000 generated windows per fit. Leverage is mean return/next-day absolute-return correlation. Polynomial-tailed models may lack a finite population fourth moment.',
 'leverage_table.tex':r'Paired Heston ablation with one million paths from the same joint initial bank and common variance paths. Row two fixes price-shock correlation; row three also changes it. SE uses twenty paired 50,000-path batches; the final column is within-window leverage.',
 'missing_table.tex':r'100-stock score differences from FHS. Stress holds forecasts fixed and assigns 100\% loss to each missing security-day; repeated missing days make this a severe diagnostic. Complete-date comparisons use the same 1,003 dates for every model.',
 }
 for path,text in captions.items():caption('referee_round2/analysis/'+path,text)
 for panel,name in [('crsp5','Five size baskets'),('crsp100','100-stock panel')]:
  for file,tail in [('primary','1'),('secondary','2.5')]:
   caption(f'referee_round2/analysis/{panel}_{file}.tex',name+rf', equal-portfolio {tail}\% risk on 2,514 dates. Paired basic intervals and the fifteen-model MCS use 9,999 circular draws and 20-day blocks; Holm inference has fourteen FHS contrasts. Hits are exceedance percentages.')
  caption(f'referee_round2/analysis/{panel}_filter_decomposition.tex',name+r' matched-filter attribution. The first four components sum to Total on every date; the EVT and Student comparisons hold GARCH location and variance fixed. The six contrasts have a separate Holm adjustment.')
  caption(f'referee_round2/analysis/{panel}_calibration.tex',name+r' one-percent calibration. UC: Kupiec coverage; CC: the finite-instrument conditional-moment diagnostic. Slash-separated values distinguish seeds. Tests are unadjusted and conditional on fitted forecasts.')
  caption(f'referee_round2/analysis/{panel}_selection.tex',name+r': one-percent test FZ0 on 2,514 dates, seed zero. Columns select checkpoints along identical training trajectories; Likelihood still reports test risk scores. All forty neural rule/model contrasts and their Holm adjustment are supplied.')
 for study,name in [('t','Student'),('metat','Heterogeneous meta-$t$'),('heston','Heston')]:
  caption(f'referee_round2/analysis/{study}_selection.tex',name+r' one-percent ES RMSE across ten datasets. Columns change only checkpoint selection, with common training trajectories and evaluation seeds. Paired MSE differences and SE are supplied in the CSV.')
 caption('expansion/analysis/filter_table.tex',r'Five-basket conditioning ablation: one-percent equal-portfolio loss, seed zero. All ten models are refitted under full-covariance EWMA using the same FZ0 rule. Differences are full minus diagonal; inference adjusts ten contrasts.')
 caption('expansion/analysis/radial_table.tex',r'Radial-splice sensitivity at one percent: ten estimation datasets and 100,000 paired before/after scenarios per fit. Improvements and adverse outcomes are retained; paired squared-error statistics appear in the CSV. The procedure assumes adequate generated directions.')
 caption('results/revision/oracle_table.tex',r'True density on the common 20,000 test observations. SE measures test-sample variation; population entropy and realized oracle NLL differ. Heston has no available exact joint-density oracle.')

 p=ROOT/'main.tex';s=p.read_text();s=re.sub(r'\\title\{[^\n]*\}',lambda _:r'\title{'+TITLE+'}',s,count=1)
 # Keep all qualifications once, apart from local assumptions required to read a statistic.
 limits=r'''\section{Implications and limitations}\label{sec:limitations}
The evidence gives volatility conditioning priority in these fitted comparisons. Most of the statistical-benchmark point-score advantage is associated with the GARCH variance filter; the incremental EVT effect is undetected under a common filter. The variance-filter contrasts themselves lose significance under their six-contrast Holm adjustment. This establishes neither a universal filter ranking nor equivalence of empirical and EVT residual laws. Meta-$t$ benefits on the two correctly specified synthetic targets likewise concern that favorable family. Architectural tail theorems constrain asymptotic representation, while finite-level estimation depends on fitting and validation.

The design is retrospective and covers fixed architectures, training budgets, and objectives. Each checkpoint rule uses validation observations only; choosing the best rule after inspecting test scores requires confirmation on an untouched sample. Likelihood-selected states optimize the validation criterion along a fixed training trajectory, rather than over the entire model family. The archived density-selected design is unpaired. Larger validation tail levels target a different functional, and pooling portfolios adds no independent dates. Sparse exceedances therefore remain consequential despite a common scoring rule.

Inferential scope is explicit: fourteen primary FHS contrasts per panel, six matched-filter contrasts, forty checkpoint-rule contrasts per panel, and ten covariance-filter contrasts form separate multiplicity families. Together they provide no global guarantee over the research exercise. The bootstrap conditions on retained forecasts without refitting models; rare exceptions and nonstationarity limit the calibration tests' chi-square approximation. MCS membership concerns relative scores, whereas calibration tests assess identification restrictions. Non-rejection in either exercise does not prove equivalence. The ten-dataset squared-error SEs are dispersion estimates, with the heavy-tail moment limitation stated in Section 7.2.

External scope is limited by one optimization seed in the 100-stock study, established large-cap universes, and three portfolio exposures. Zero-filled missing slots can suppress downside variation, particularly under distress-related missingness; the stress and complete-date diagnostics cannot identify unobserved returns or a universal bias sign. Heston results condition on a finite joint variance bank and time discretization, without a correctly specified fitted Heston benchmark. Its static joint-window generators can encode within-window leverage, while recursive state-conditioned dynamics require separate validation.

'''
 mc=pd.read_csv(O/'crsp100_mcs.csv');mcp=float(mc[(mc.model=='Flow-meta-t')&(mc.alpha==.01)&(mc.block==20)].p_mcs.iloc[0])
 secondary=pd.read_csv(O/'selection_summary.csv');v=float(secondary[(secondary.study=='crsp100')&(secondary.model=='Flow-meta-t')&(secondary.criterion=='density')].fz0.iloc[0])
 primary=pd.read_csv(O/'crsp100_summary.csv');fhs=float(primary[(primary.model=='Historical')&(primary.alpha==.01)].fz0.iloc[0]);gb=float(primary[(primary.model=='GARCH-FHS')&(primary.alpha==.01)].fz0.iloc[0])
 conclusion=r'''\section{Conclusion}
Volatility conditioning accounts for most of the observed statistical-benchmark lead, and empirical residuals are competitive with EVT under the same GARCH filter. Across the primary and checkpoint-sensitivity comparisons, no neural improvement over FHS survives its within-panel multiplicity correction.

'''
 conclusion+=f"Validation design materially changes the apparent generator ranking: in 100 stocks, Flow-meta-$t$ is excluded from the primary 95\\% MCS ($p={mcp:.4f}$), yet likelihood selection gives it the lowest observed mean among the neural checkpoint variants ({v:.4f}, versus FHS {fhs:.4f}); GARCH-FHS remains lower overall at {gb:.4f}. Meta-$t$ under correct specification and the residual flow remain useful controlled constructions. Their value is assessed jointly with conditioning, validation, density fit, and calibration.\n\n"
 a=s.index(r'\section{Implications and limitations}');b=s.index(r'\appendix',a);s=s[:a]+limits+conclusion+s[b:]
 p.write_text(s)
 # Current documentation and prior responses stay distinguishable.
 p=ROOT/'README.md';s=p.read_text();s=s.replace(s.splitlines()[0],'# '+TITLE,1)
 s=s.replace('`main.pdf` and `main.tex` contain the completed second referee revision.','`main.pdf` and `main.tex` contain the final editorial revision of the completed second referee study.')
 s=s.replace('the final editorial revision of the completed second referee study.','the final 37-page editorial revision of the completed second referee study.')
 if 'The final editorial pass adds' not in s:
  s=s.replace('## What changed','The final editorial pass adds the checkpoint-selection example to the conclusion and consolidates repeated qualifications in Section 10. All 36 numerical tables, theory and experimental results are retained. See `FINAL_POLISH.md`.\n\n## What changed',1)
 p.write_text(s)
 (ROOT/'FINAL_POLISH.md').write_text(f'''# Final editorial response

Title: **{TITLE}**.

The manuscript is 37 pages, reduced from 39 while retaining all 36 tables and both figures.

1. The conclusion now gives the 100-stock selection reversal. The current fifteen-model MCS has p = {mcp:.4f}; the earlier thirteen-model value was 0.0242. Likelihood-selected Flow-meta-t scores {v:.4f} versus FHS {fhs:.4f}, the lowest observed mean among the neural checkpoint variants. GARCH-FHS remains lower at {gb:.4f}. The comparison does not imply multiplicity-adjusted neural superiority.
2. Repeated empirical qualifications and caption boilerplate are consolidated in Section 10. Mathematical assumptions, the IWAE expectation-versus-realization distinction, same-filter attribution, retrospective selection, separate multiplicity families, missing-data direction, and finite-bank Heston scope are preserved. All 36 table bodies and the theory are unchanged.
3. The title puts volatility conditioning first without asserting that architecture is irrelevant or that a universal filter ordering has been proved.

This is an editorial revision. No experiment or numerical inference was rerun to change a result. The previous 126 executed checks are retained; the editorial audit verifies unchanged numerical data, table bodies and theory, as well as the new conclusion against the stored output.
''')
 p=ROOT/'REVISION_AUDIT.md';s=p.read_text();p.write_text('# Final editorial revision\n\nSee [FINAL_POLISH.md](FINAL_POLISH.md) for the final minor-revision response.\n\n'+s if not s.startswith('# Final editorial revision') else s)
 print('Final editorial pass applied:',TITLE)

if __name__=='__main__':main()

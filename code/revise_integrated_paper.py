"""Single-paper revision: exposition, scope, new conditional-model diagnostics.

Run after round5_diagnostics.py, round5_dcc.py and round5_construction.py.
Primary estimates are retained. New analyses have explicit retrospective scope.
"""
from pathlib import Path
import json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; B=ROOT/'referee_round5/baseline'; O=ROOT/'referee_round5/analysis'; R4=ROOT/'referee_round4/analysis'

def main():
    for p in (B/'round4_analysis').glob('*.tex'):
        (R4/p.name).write_text(p.read_text())
    s=(B/'main.tex').read_text()
    title='Parametric and Nonparametric Models of Joint Returns: Density, Dependence, and Portfolio Tail Risk'
    s=re.sub(r'\\title\{[^\n]*\}',lambda m:r'\title{'+title+'}',s,count=1)
    abstract=r'''We study when flexible models of joint asset returns improve density estimation and portfolio tail-risk forecasts. We compare Gaussian, shrinkage-Gaussian, Student and meta-t distributions; empirical innovations through filtered historical simulation (FHS); and neural generators comprising normalizing flows, variational autoencoders and Wasserstein GANs. Meta-t combines asset-specific Student margins with a t copula; normalizing flows learn invertible transformations of a base law. The neural models generate static innovations or fixed return windows, while separate volatility filters make forecasts conditional on market history. A restricted Heston--Wishart model and GARCH-based controls examine dynamic covariance and volatility. Ten independent estimation datasets for each of three synthetic laws and 2,514 out-of-sample CRSP dates in five-basket and 100-stock panels distinguish specification, conditioning and checkpoint-selection effects. Correctly specified parametric models perform strongly, but Student likelihood fits to Heston improve density fit while producing poor expected-shortfall estimates. On CRSP, the largest observed score gains arise from variance conditioning; their adjusted tests remain inconclusive. No tested neural innovation model improves on FHS after primary multiplicity correction under the common risk-selection rule, and rankings change with checkpoint selection. The fitted Student law improves in the primary 100-stock family but not the broader retrospective comparison. Pairwise downside checks and five-basket DCC controls supplement portfolio scores. These results connect architectural tail restrictions to finite-sample evidence without treating density fit, dependence calibration and portfolio risk as interchangeable objectives.'''
    (R4/'abstract.tex').write_text(abstract+'\n')
    abstract=abstract.replace('Wasserstein GANs','Wasserstein generative adversarial networks').replace('five-basket DCC controls','five-basket dynamic-correlation controls')
    (R4/'abstract.tex').write_text(abstract+'\n')
    start=s.index('Multivariate financial models');end=s.index(r'\paragraph{Relation to existing work.}')
    intro=r'''A multivariate return model must describe each asset's risk and how assets move together, then translate that joint law into portfolio outcomes. We ask when a structured parametric distribution is sufficient and when empirical or neural flexibility improves this task. The comparison separates three objectives: fitting the joint density, reproducing downside dependence, and forecasting portfolio Value-at-Risk (VaR) and expected shortfall (ES). A model can succeed at one and fail at another.

\paragraph{What is compared.}
Gaussian, shrinkage-Gaussian, multivariate Student and meta-$t$ laws provide structured joint distributions. FHS combines empirical innovations with a volatility filter. Normalizing flows learn invertible distributional transformations, VAEs use latent-variable decoders, and Wasserstein GANs learn scenario samplers \citep{papamakarios2021review,kingma2014vae,goodfellow2014gan}. A fixed neural network is itself finitely parameterized; ``neural'' and ``nonparametric'' are therefore distinct classifications. We also examine coordinate- and portfolio-GARCH controls, five-basket dynamic conditional correlation (DCC), and a restricted Heston--Wishart stochastic-covariance specification.

The neural comparison concerns static innovation laws fitted annually, and static five-day joint windows in the Heston experiment. Conditional forecasts arise from separately specified filters; the networks do not learn a recursive state-conditioned generator. The bounded flow architectures, their dense-output controls, and training budgets are stated explicitly. The results therefore describe these implementations rather than all possible neural models or conditional time-series generators such as Quant GAN \citep{wiese2020quantgan,buehler2020simulator}.

\paragraph{Main findings.}
Table~\ref{tab:r5-summary} collects the evidence that organizes the paper. Variance conditioning accounts for the largest observed CRSP score gains, although the adjusted tests do not resolve those effects. Within the common primary filter, the Student gain in 100 stocks is sensitive to the comparison family, and no tested neural innovation model establishes an adjusted improvement over FHS. In controlled Heston data, analytic Student ES exposes a clear density--risk discrepancy. Theoretical tail classes explain representation restrictions; finite-level experiments determine whether the tested restrictions have detectable practical consequences.
\input{referee_round5/analysis/summary_table.tex}

Our contributions are:
\begin{enumerate}
\item An integrated experimental design that separates joint-law specification, predictable filtering, and checkpoint selection through matched data, paired uncertainty and explicit multiplicity families.
\item A one-sided loss-tail result distinguishing elliptical from independent Student latent noise, together with a residual meta-$t$ flow whose marginal indices are controlled.
\item Evidence on where these distinctions matter: favorable parametric specification in synthetic laws, the Heston density--risk discrepancy, selection-sensitive neural rankings, and the separate roles of volatility and dependence.
\end{enumerate}
The same manuscript contains the theoretical statements, empirical tests and supporting proofs. Longer derivations, secondary comparisons and numerical audits are in its appendices.

'''
    s=s[:start]+intro+s[end:]
    s=s.replace('We use that object; we do not propose a different distribution under the same name.','We use that established distribution; the application and residual-flow construction are the contributions here.')
    s=s.replace('motivating a common decision-relevant checkpoint criterion.','motivating a common decision-relevant checkpoint criterion. Our validation criterion does not turn the implemented WGAN into Tail-GAN, whose training objective is different.')
    s=s.replace('Standard Heston mixture and transform calculations are moved to','Standard Heston mixture and transform calculations appear in')
    s=s.replace(r'\log q_\theta(x)=\E_{q_\phi}',r'\log p_\theta(x)=\E_{q_\phi}')
    s=s.replace(r'\input{theory.tex}',r'''\paragraph{Why separate the objectives?}
For integrable loss laws, $|\ES_\alpha(P)-\ES_\alpha(Q)|\le W_1(P,Q)/\alpha$, while a small forward KL divergence alone gives no unrestricted ES-error bound. Theorem~\ref{thm:w1} and Proposition~\ref{prop:kl} state these standard facts precisely. They motivate measuring density and risk separately; they do not predict a finite-sample ranking.
\input{theory_main.tex}''')
    # Retain mathematical statements centrally; place proofs and background in this PDF's appendices.
    theory=(B/'theory.tex').read_text()
    theory=theory.replace('with $P$ positive definite, and $w\\in\\R^d$.','with $P$ positive definite, $Z\\in\\R^m$, and $w\\in\\R^d$.')
    theory=theory.replace('independent of the regularly varying radius $R$, and define',r'independent of the regularly varying radius $R$, where $R^2/m\sim F_{m,\nu}$ after whitening, and define')
    theory=theory.replace('with $ca^\\beta b>1$ and enough slack for $M$.','with $ca^\\beta b>1$ and enough slack for $M$. The bounded location $\\mu$ cannot cancel this growing scale-noise product on the cone.')
    split=theory.index(r'\section{What objective tolerances certify}')
    core,background=theory[:split],theory[split:]
    proofs=[]
    def relocate(m):
        before=core[:m.start()]; labels=re.findall(r'\\label\{([^}]+)\}',before)
        # Equation labels can follow the theorem label: select the enclosing theorem block.
        environments=list(re.finditer(r'\\begin\{(?:theorem|proposition|corollary|lemma)\}',before))
        section=before[environments[-1].start():]; label=re.findall(r'\\label\{([^}]+)\}',section)[0]
        proofs.append(r'\paragraph{Proof of result~\ref{'+label+'}.}\n'+m.group(0))
        return r'\noindent\emph{Proof.} See Appendix~\ref{app:proofs}.'
    core=re.sub(r'\\begin\{proof\}[\s\S]*?\\end\{proof\}',relocate,core)
    core+=r'''
\subsection{From representation to finite-level tests}\label{sec:theory-bridge}
The results determine what a fixed architecture can represent asymptotically, not the amount of data required to distinguish two fitted models at one percent. They motivate three concrete checks. Bounded Gaussian decoders and growing-scale decoders are compared on Heston; elliptical and independent Student noise are evaluated at both one percent and 0.1 percent; and meta-$t$ is tested in both correctly specified and misspecified laws. The paired VAE-H/VAE-G comparison does not resolve a finite-level difference on ten datasets of 4,000 observations. This is limited empirical resolution, not a proof that tail class is statistically unidentifiable. The cross-level failures of Flow-t and WGAN-i show why selection at a single level does not establish accuracy deeper in the tail.
'''
    background=background.replace('What objective tolerances certify','Objective bounds and radial tail calibration')
    background=background.replace(r'\section{A radial EVT repair with explicit angular conditions}',r'\subsection{Radial EVT calibration with angular conditions}')
    background=background.replace('The amplification is $\\alpha^{-1}$, not $\\alpha^{-2}$. ','')
    (ROOT/'theory_main.tex').write_text(core)
    (ROOT/'theory_appendix.tex').write_text(r'\section{Proofs of the architectural results}\label{app:proofs}'+'\n'+'\n\n'.join(proofs)+'\n'+background)
    s=s.replace(r'\input{heston_appendix.tex}',r'\input{theory_appendix.tex}'+'\n'+r'\input{heston_appendix.tex}')
    # Main results keep risk and density tables. Detailed sensitivity tables go to the same PDF appendix.
    details=['selection_combined','dense_crsp','dense_heston','coordinate_combined','joint_combined']
    for name in details:s=s.replace(r'\input{referee_round4/analysis/'+name+'.tex}','')
    s=s.replace(r'\input{referee_round4/analysis/joint_findings.tex}',r'\input{referee_round5/analysis/dependence_findings.tex}')
    s=s.replace(r'\input{referee_round4/analysis/control_findings.tex}',r'\input{referee_round4/analysis/control_findings.tex}'+'\n'+r'\input{referee_round5/analysis/dcc_findings.tex}')
    appendix=r'''\section{Selection and conditioning sensitivity}\label{app:controls}
\input{referee_round4/analysis/selection_combined.tex}
\input{referee_round4/analysis/dense_crsp.tex}
\input{referee_round4/analysis/dense_heston.tex}
\input{referee_round4/analysis/coordinate_combined.tex}
\input{referee_round5/analysis/dcc_table.tex}
\FloatBarrier
\section{Dependence diagnostics}\label{app:dependence}
\input{referee_round5/analysis/tail_table.tex}
\input{referee_round4/analysis/joint_combined.tex}
\input{referee_round4/analysis/joint_findings.tex}
\FloatBarrier
'''
    s=s.replace(r'\section{Synthetic density uncertainty}',appendix+r'\section{Synthetic density uncertainty}')
    # Consolidated limits, without removing essential local qualifications.
    a=s.index(r'\section{Implications and limitations}');b=s.index(r'\section{Conclusion}',a)
    s=s[:a]+r'''\section{Implications and limitations}\label{sec:limitations}
The strongest observed filter effects prioritize volatility conditioning in this design, but their adjusted tests remain inconclusive. The neural conclusion concerns small bounded architectures for static innovations and a fixed training budget. Dense-output controls admit their nested parametric laws; they do not exhaust wider networks, unbounded shifts, alternate objectives or recursive conditional generators. The five-basket DCC extension supplies a conventional dynamic-correlation control; no 100-stock DCC evaluation is claimed. Coordinate-GARCH/meta-$t$ supplies a static-copula GARCH control in both panels.

Each checkpoint rule uses validation observations only. The candidate grid was amended retrospectively, before the associated new evaluations, to include initialization, early states and admissible nested laws. Selecting a different rule after inspecting its test performance still requires independent confirmation. One-percent validation contains about 2.5 nominal exceedances per CRSP year; pooled portfolios add exposures rather than independent dates. The 100-stock panel has one optimization seed, so its between-seed training variation is unmeasured.

Inference conditions on retained forecasts. Date-block resampling preserves cross-model pairing but does not refit models, regenerate scenarios, or establish equivalence. The Holm families in Table~\ref{tab:r4-families} have separate scopes; there is no global error guarantee over the retrospective research exercise. MCS membership depends on studentized pairwise precision and the candidate set, rather than ordering every retained model above every excluded one. Seed-score averaging measures typical fitted forecasts; an ensemble of forecasts would need its own evaluation.

Established large-cap universes, zero-filled missing allocations and a small set of portfolio exposures limit external validity. Missing distress returns can suppress measured downside risk. The partial fixed-forecast stress and complete-date diagnostics cannot recover those outcomes. The original co-exceedance score measures fold-level joint-event calibration, with constant probabilities inside a fold. Pairwise diagnostics additionally separate asymptotic tail coefficients from finite-level joint exceedances; their uncertainty still includes marginal-calibration error and only 2,514 dates.

Synthetic results condition on ten estimation datasets, and Heston on a finite joint variance bank and time discretization. Heavy tails limit normal-theory interpretations of squared-error SEs. The restricted Wishart fit understates synthetic risk; its CRSP filter fails the declared numerical tolerance and remains excluded from primary scores, Holm families and MCS. Neither finding rejects the general Wishart family or identifies the incremental benefit of stochastic correlation.

'''+s[b:]
    s=s.replace(r'\input{referee_round2/analysis/missing_table.tex}',r'\input{referee_round5/analysis/missing20_table.tex}')
    s=s.replace('The latter comes from a staged profile with marginal fits and Kendall-based dependence held fixed; it should not be interpreted as a dimension-invariant structural tail parameter.','The 100-dimensional value profiles $\\nu_c$ over a Kendall-tau-fixed correlation matrix $P$ after marginal fitting. This staged estimator differs from the low-dimensional fit, so the values are not directly comparable as structural tail parameters. The finite-level checks in Appendix~\\ref{app:dependence} assess their downside implications.')
    s=s.replace('The new joint-event diagnostic broadens the dependence evidence but still measures one feature of the joint law.','The dependence diagnostics assess specified features of the joint law.')
    (ROOT/'main.tex').write_text(s)
    # Replace narrative revision history with final protocol; retain one concise audit disclosure.
    methods=(B/'round4_methods.tex').read_text();a=methods.index(r'\subsection{Paired density uncertainty}')
    methods=r'''\subsection{Checkpoint selection and its sensitivity}\label{sec:checkpoint-repair}
The candidate grid contains initialization, updates $10,25,50,100$, and each 250th update thereafter. Flows also admit an identity coupling map; meta-$t$ residual flows admit the exact fitted base, and dense Gaussian and Student flows admit their fitted full-scatter parametric benchmarks. Every candidate is evaluated on the same validation observations and fixed scenario draws. Alternative rules use portfolio FZ0 at 2.5\% or 5\%, pooled-portfolio FZ0 at 1\%, likelihood, or the final state. Flows use exact validation NLL; VAEs use IWAE-8 for validation and IWAE-64 for testing. Table~\ref{tab:r4-selection} reports the matched comparison. The grid was amended during review to include omitted early and nested candidates before the new evaluations; replay and the excluded earlier comparisons are documented in the reproducibility audit. Test-period performance never selects a candidate under an individual rule.

Forty neural rule/model forecasts form one FHS comparison family per panel; the 33 alternatives against their own model's primary rule form another. The sensitivity uses seed zero, while the primary five-basket comparison averages two seed scores. A validation year contains only about 2.5 nominal one-percent exceedances.

'''+methods[a:]
    methods=methods.replace('The repaired diagonal and dense runs use the same training trajectories and validation rules','The diagonal and dense controls use matched training budgets and validation rules')
    methods=methods.replace('The existing 1,050 GARCH and 1,050 EWMA scalar fits and predictable states are retained.','There are 1,050 GARCH and 1,050 EWMA scalar fits. Coordinate-GARCH with meta-$t$ innovations is a copula-GARCH benchmark: coordinate scales are dynamic, while its fitted Student margins and t copula are fixed within the year.')
    methods=methods.replace('These restrictions support the coordinate-index guarantee in Proposition~\\ref{prop:marginal}, while limiting dependence representation.','These restrictions support the coordinate-index guarantee in Proposition~\\ref{prop:marginal}, while limiting dependence representation. A dense map retains radial growth bounds but can mix heterogeneous marginal indices: dependence capacity and coordinate-level tail guarantees are different requirements.')
    methods=methods.replace('One co-exceedance event supplements the portfolio projections; it does not identify every aspect of the joint distribution.','Because both the innovation law and innovation thresholds are fixed within a fold, $p_t$ is constant within that fold. This is a fold-level calibration check, not a test of within-year timing or dynamic discrimination. Appendix~\\ref{app:dependence} retains it alongside the more direct pairwise check below.')
    methods+=r'''
\subsection{Finite-level pairwise downside calibration}\label{sec:pair-tail}
For each fitted t copula and pair $(i,j)$, compute
\[
\lambda_{L,ij}=2t_{\nu_c+1}\!\left(-\sqrt{\frac{(\nu_c+1)(1-P_{ij})}{1+P_{ij}}}\right),\qquad
\lambda_{ij}(u)=\frac{C_{ij}(u,u)}u,\quad u\in\{0.01,0.05\}.
\]
The first is the limiting tail coefficient \citep{demarta2005tcopula}; the second is a finite-level conditional exceedance probability. Numerical one-dimensional quadrature evaluates the latter, with a refinement check. Held-out indicators use training-fitted marginal quantiles. We average equally over pairs within each fold and then over dates, and bootstrap dates jointly across all pairs. This avoids treating 4,950 pairs as independent observations. We also report marginal exceedance rates: deviations in the joint statistic can reflect both copula and marginal error. These descriptive intervals do not estimate a limiting coefficient from one-percent data.
'''
    (ROOT/'round4_methods.tex').write_text(methods)
    # Concrete synthetic findings and level-transfer failure for both highlighted models.
    syn=(B/'round4_analysis/synthetic_narrative.tex').read_text()
    syn=syn.replace(r'\paragraph{Heston misspecification.}',r'\paragraph{Finding: density fit can improve while tail-risk estimation deteriorates.}')
    syn=syn.replace('The lowest observed Heston RMSE after repair is Flow-t at 0.243.','The lowest observed one-percent Heston RMSE is Flow-t at 0.243.')
    syn=syn.replace('WGAN-i changes from RMSE 0.392 at one percent to 3.415 at 0.1 percent, illustrating that selection at one level need not transfer to another. The tail theorems constrain asymptotic representation; they do not predict these finite-level errors.','Flow-t changes from RMSE 0.243 at one percent to 1.635 at 0.1 percent, where empirical ES has 1.009 and VAE-tlatent 0.730. WGAN-i changes from 0.392 to 3.415. Both demonstrate sensitivity across probability levels. The asymptotic results motivate these comparisons; they supply no ordering of finite-level errors. The unresolved VAE comparison is evidence of limited resolution in this design, rather than evidence that the architectures have the same tail class.')
    syn=syn.replace('Student MLE 2.841, and dense Flow-G 0.392.','Student MLE 2.841, Gaussian 1.508, and dense Flow-G 0.392. Meta-$t$ is worse than empirical ES but better than Gaussian at one percent; Student MLE is worse than both.')
    syn=syn.replace('This target-law result is unaffected by checkpoint repair; ','')
    syn=syn.replace('Repaired one-percent ES MSE differences','One-percent ES MSE differences')
    (R4/'synthetic_narrative.tex').write_text(syn)
    # Protocol and covariance-ridge details from the implemented code and new audit.
    crsp=(B/'expanded_crsp_methods.tex').read_text()
    crsp=crsp.replace('with a pessimistic $-100\\%$ sensitivity','with a fixed-forecast $-20\\%$ sensitivity for the 100-stock panel')
    ridge=json.loads((O/'ridge_audit.json').read_text());maxshift=max(r['max_relative_min_eigenvalue_shift'] for r in ridge)
    crsp=crsp.replace('A small diagonal ridge makes the Cholesky operation numerically defined.',r'The ridge is $10^{-10}I$ in five dimensions and $10^{-8}I$ in 100 dimensions, in squared percentage-point units. In the 100-stock test states the minimum unregularized eigenvalue reaches $5.44\times10^{-13}$, so the ridge can dominate the weakest directions; the maximum regularized condition number is $4.64\times10^{10}$. This is a numerical floor rather than covariance shrinkage, and the small absolute ridge does not imply negligible whitening effects. With decay 0.94, the asymptotic weighting effective sample size is $(1+0.94)/(1-0.94)\simeq32.3$, so 100-dimensional covariance estimation can be noisy despite a long historical panel. These diagnostics qualify the worse EWMAcov-FHS point scores; they do not identify their cause or establish a general cost of dynamic correlation.')
    crsp=crsp.replace('The earlier archive uses different selection rules, scenario counts, and training seeds.','')
    crsp=crsp.replace('The repaired primary set','The primary set')
    crsp=crsp.replace(r'\input{round2_filter_methods.tex}',r'''\input{round2_filter_methods.tex}
\paragraph{DCC and constant-correlation controls.}
The five-basket extension fits Gaussian and variance-normalized Student laws under constant correlation (CCC) and DCC, using the same estimated coordinate-GARCH scales \citep{engle2002dcc}. With standardized returns $z_t$, the recursion is
\[
Q_t=(1-a-b)\overline Q+a z_{t-1}z_{t-1}^{\top}+bQ_{t-1},\qquad
R_t=\operatorname{diag}(Q_t)^{-1/2}Q_t\operatorname{diag}(Q_t)^{-1/2}.
\]
Training likelihood estimates $a,b\ge0$, $a+b\le0.999$, and, for Student laws, $2.05\le\nu\le200$. Three DCC starts are used. The training second moment $\overline Q$ has a $10^{-10}I$ ridge; $Q_0=\overline Q$. CCC sets $a=b=0$. Parameters remain fixed during validation and testing; states update only after observations. Analytic portfolio risk eliminates scenario error for these four controls. Their eight contrasts against diagonal-EWMA-FHS and coordinate-GARCH-FHS form a retrospective family; a 21-model, 20-contrast encompassing sensitivity is archived separately. No high-dimensional DCC fit is included.
''')
    (ROOT/'expanded_crsp_methods.tex').write_text(crsp)
    methods=(B/'expanded_methods.tex').read_text()
    methods=methods.replace('The current experiment retains the expanded training datasets and trajectories, repairs the checkpoint candidate set, and adds two training-only joint statistical benchmarks. A machine-readable protocol records seeds, budgets, dates and scenario counts. Earlier forecasts remain archived; the main tables use the repaired selection rule.','A common protocol fixes training datasets, budgets, validation criteria and scenario counts. All model comparisons use matched estimation datasets; the final candidate grid includes initialization, early checkpoints and admissible nested parametric laws.')
    (ROOT/'expanded_methods.tex').write_text(methods)
    sel=(B/'round4_analysis/selection_findings.tex').read_text().replace('Selection fragility after repair.','Selection sensitivity.')
    # Report computed differences rather than subtracting rounded score levels.
    sel=sel.replace('from 0.010 to 0.058','from 0.0103 to 0.0577 (computed before rounding)')
    (R4/'selection_findings.tex').write_text(sel)
    (R4/'control_findings.tex').write_text((B/'round4_analysis/control_findings.tex').read_text().replace('Repaired controls.','Matched conditioning controls.'))
    (R4/'conclusion.tex').write_text(r'''The comparison is organized by conditioning, specification and selection. Switching the variance recursion accounts for score changes of approximately $-0.055$ in five baskets and $-0.115$ in 100 stocks, the largest observed gains in the portfolio attribution. Their six-contrast adjusted tests remain inconclusive. Replacing empirical residual tails by EVT adds no detectable benefit at the evaluated one-percent level. The additional five-basket DCC controls do not establish a risk-score improvement over FHS.

Within the common primary filter, fitted Student innovations improve in the 100-stock comparison with Holm $p=0.0352$, but not in the broader retrospective family ($p=0.0594$). No tested neural innovation model establishes a primary adjusted improvement over FHS under the common one-percent selection rule. Alternative rules materially change the observed rankings. These are results for the stated static architectures, budgets and validation procedures, not a general verdict on conditional neural modeling.

The synthetic studies connect representation to estimation. Correct specification favors structured parametric laws; on Heston, Student likelihood improves NLL while its analytic ES estimate performs poorly. Flow-t and WGAN-i show that favorable one-percent errors need not transfer to 0.1 percent. The architectural theorems explain permissible asymptotic behavior, while the VAE-H/VAE-G comparison remains unresolved at the available sample size. Meta-$t$'s 100-stock near-Gaussian limiting copula tail coefficient also differs substantially from its finite-level joint exceedance probabilities. Density, dependence and portfolio risk therefore require separate assessments within the same model comparison.

The restricted Wishart fit understates synthetic ES, and its unconverged CRSP filter remains appendix-only. A more flexible, converged stochastic-covariance specification would require a new evaluation. The integrated evidence favors checking the filter, dependence representation and selection procedure before attributing a result to model-family flexibility.
''')
    # New results are synthesized only from completed numerical files.
    tail=pd.read_csv(O/'tail_summary.csv');t=tail[(tail.panel=='crsp100')&(tail.u==.05)].iloc[0]
    (O/'dependence_findings.tex').write_text(r'\paragraph{Finite-level dependence and fold-level calibration.}'+'\n'+rf'The original co-exceedance probability is constant within each annual fold, so its Brier score assesses fold-level calibration rather than within-year timing. The new pairwise check gives a different perspective. In 100 stocks the mean fitted limiting coefficient is {t.lambda_limit:.4f}, while the mean model value $C_{{ij}}(0.05,0.05)/0.05$ is {t.model:.3f}; the observed value is {t.observed:.3f}, with marginal exceedance rate {100*t.marginal_rate:.2f}\%. The observed-minus-model interval is $[{t.low:.3f},{t.high:.3f}]$. Thus a small limiting coefficient does not imply negligible finite-level joint downside risk. All four pair-averaged difference intervals include zero; this limited resolution does not establish correct copula specification. Table~\ref{{tab:r5-tail}} reports both levels and panels, and Table~\ref{{tab:r4-joint}} retains the original event diagnostic.'+'\n')
    dcc=pd.read_csv(O/'dcc_summary.csv');g=dcc[dcc.benchmark=='Historical'].set_index('model')
    (O/'dcc_findings.tex').write_text(r'\paragraph{Conventional dynamic-correlation controls.}'+'\n'+rf'The five-basket mean FZ0 scores are {g.loc["CCC-G","fz0"]:.3f} for CCC-G, {g.loc["CCC-t","fz0"]:.3f} for CCC-t, {g.loc["DCC-G","fz0"]:.3f} for DCC-G, and {g.loc["DCC-t","fz0"]:.3f} for DCC-t, against FHS 1.370. No improvement survives the eight-contrast adjustment against the two FHS filters. The primary-plus-DCC 21-model sensitivity likewise detects no improvement over FHS. These controls fill the conventional dynamic-correlation comparison in five dimensions; they do not identify all possible benefits of dynamic correlation. See Table~\ref{{tab:r5-dcc}}.'+'\n')
    from analyze import tex_table
    rows=[['Variance conditioning','$-0.055$ / $-0.115$','Five / 100; Holm/6 $p=0.072/0.096$'],['Incremental EVT','$-0.005$ / $+0.002$','Same GARCH filter; neither resolved'],['Student versus FHS','$-0.023$','100 stocks; Holm $0.0352$ / $0.0594$'],['Heston Student fit','NLL 28.480; RMSE 2.841','Gaussian: 31.500; 1.508'],['Heston level transfer','Flow-t: 0.243 / 1.635','ES RMSE at 1\\% / 0.1\\%']]
    (O/'summary_table.tex').write_text(tex_table('Five comparisons that organize the study. FZ0 differences are negative when the named change improves the point score. Student p-values refer to the primary and broader retrospective families. RMSE and NLL are distinct objectives; no row asserts universal model dominance.','tab:r5-summary',['Comparison','Observed result','Scope / reference'],rows,'lll'))
    selection=pd.read_csv(R4/'selection_summary.csv');rows=[]
    for (panel,name),g in selection.groupby(['panel','model'],sort=False):
        v=g.set_index('rule').fz0
        rows.append([panel.replace('crsp',''),name,f'{v["fz1"]:.4f}',f'{v["density"]:.4f}' if 'density' in v else '--',f'{v["final"]:.4f}',f'{v["final"]-v["fz1"]:.4f}'])
    (R4/'selection_combined.tex').write_text(tex_table('Matched checkpoint rules, seed zero. Values are one-percent equal-portfolio FZ0; the difference column is computed before rounding. All candidates use the common training trajectory for that architecture. The forty FHS contrasts and 33 alternatives against their own primary rule are exported separately.','tab:r4-selection',['Panel','Model','Risk','Likelihood','Final','$\\Delta$ final'],rows,'llrrrr'))
    # Brief audit disclosure and accurate record counts.
    audit=r'''\paragraph{Final inference and data protocol.}
All displayed primary intervals use 9,999 circular block-bootstrap draws and basic equal-tailed intervals with matching centered tests. An earlier interval/test mismatch was corrected before the current comparisons. The 100-stock universe requires 252 finite pre-estimation observations; 41 cold-start pilot evaluations were excluded and every retained model was refitted under the amended rule. The incomplete preliminary 2024 export was rebuilt before any retained 2024 evaluation. The source audit verifies every annual panel and missingness mask across all 2,514 test dates. Earlier calculations remain separately archived.

\paragraph{Conditional-moment calculation.}
For each seed, use $v_t=(\ind\{L_t>q_t\}-\alpha,m_t/e_t)$ and instruments 1 and $\ind\{L_{t-1}>q_{t-1}\}-\alpha$. Stacking their products gives $z_t$. The statistic is $n\bar z^\top\widehat\Omega^+\bar z$, with Bartlett HAC bandwidth 20 and a chi-square reference using $\operatorname{rank}(\widehat\Omega)$. These are necessary finite-instrument conditional restrictions, separate from unconditional coverage; rare events and dependence limit the asymptotic approximation.

\paragraph{Implementation checks.}
Fitted copula correlation matrices are positive definite with unit diagonal. Degrees of freedom are optimized within their bounds. The empirical ES implementation includes fractional tail mass; the total-error decomposition retains its signed cross term. Full parameter records, candidate selection, covariance-filter identities and data checks are supplied with the source. The added CCC/DCC and pairwise-tail checks are described in the reproducibility ledger.
'''
    (ROOT/'expanded_audit.tex').write_text(audit)
    ver=(B/'round4_analysis/verification.tex').read_text().replace('Executed repair and reproducibility.','Reproducibility and amendments.')
    ver=ver.replace('1,110 model/fold records','1,150 model/fold records (1,110 original and forty added CCC/DCC controls)')
    ver=ver.replace('The release checks reconstruct','The earlier release checks reconstruct')
    ver+='\nThe added CCC/DCC audit verifies positive correlation matrices, agreement between vectorized and literal recursions, and that a forecast does not use its contemporaneous or future return. Pairwise-tail quadrature is checked by resolution doubling. The new diagnostics and partial missing-return stress are exported separately; none changes the primary forecast records.\n'
    (R4/'verification.tex').write_text(ver)
    # Remove revision narration in displayed captions, while retaining audit history in source notes.
    for p in [*R4.glob('*.tex'), ROOT/'expanded_methods.tex',ROOT/'expanded_crsp_methods.tex']:
        txt=p.read_text()
        txt=txt.replace('after repaired likelihood selection','under likelihood selection').replace('after repairing both candidate grids','with the common candidate grids').replace('after candidate repair','with the common candidate grid').replace('after candidate repair','under the stated candidate rule').replace('after repair','under the stated rule')
        txt=txt.replace('repaired primary forecasts','primary forecasts').replace('repaired primary set','primary set').replace('repaired selection','the stated selection').replace('Repaired selection','The stated selection').replace('repaired validation FZ0','validation FZ0').replace('repaired risk','the expanded-grid risk').replace('Unchanged portfolio-filter attribution','Portfolio-filter attribution')
        p.write_text(txt)
    # Add families without silently enlarging or replacing the primary test family.
    p=R4/'families.tex';txt=p.read_text();txt=txt.replace(r'\bottomrule',r'DCC / two FHS filters & 8, five baskets & 1\% FZ0 \\'+'\n'+r'Primary plus DCC / FHS & 20, five baskets & 1\% FZ0 \\'+'\n'+r'\bottomrule');p.write_text(txt)
    # The fitted model is self-contained; no dependency on a companion's transform equations.
    p=ROOT/'wishart_methods.tex';txt=p.read_text();txt=txt.replace('We add the matrix covariance SDE in Section~4 of \\citet{noguer2025wishart} as a fitted parametric benchmark, denoted Heston--Wishart, in the established Wishart-process and multivariate-volatility literature \\citep{bru1991,gourieroux2006,dafonseca2008wishart}.','The Heston--Wishart benchmark uses the explicitly specified matrix covariance SDE below, within the established Wishart-process and multivariate-volatility literature \\citep{bru1991,gourieroux2006,dafonseca2008wishart}. The related portfolio formulation is discussed by \\citet{noguer2025wishart}; all equations used for this benchmark are given here, independently of its transform system.');p.write_text(txt)
    bib=(B/'refs.bib').read_text()
    if 'engle2002dcc' not in bib:bib+='\n@article{engle2002dcc, author={Engle, Robert}, title={Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models}, journal={Journal of Business \\& Economic Statistics}, volume={20}, number={3}, pages={339--350}, year={2002}, doi={10.1198/073500102288618487}}\n'
    bib=bib.replace('note = {Preprint, manuscript dated December 31, 2025. Public copy on ResearchGate},','note = {Preprint, manuscript dated December 31, 2025. Public author copy},\n  doi = {10.2139/ssrn.5996134},')
    (ROOT/'refs.bib').write_text(bib)
    print('Integrated manuscript written; abstract words',len(abstract.split()),'proofs moved',len(proofs))

if __name__=='__main__':main()

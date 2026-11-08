"""Final idempotent editorial layer after the archived revision processors."""
from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'referee_round3/analysis'
def write(name,value):(ROOT/name).write_text(value)
def section(text,start,end,replacement):
    a=text.index(start);b=text.index(end,a);return text[:a]+replacement+'\n\n'+text[b:]
def main():
    abstract=r'''On 2,514 CRSP dates, neither neural nor statistical forecasts improve detectably on filtered historical simulation (FHS) after the primary within-panel multiplicity correction. GARCH's point-score lead is associated mainly with variance filtering. This null conditions on a noisy one-percent validation rule whose alternatives change rankings. We compare Gaussian and meta-t laws, empirical estimators, normalizing flows, variational autoencoders, and Wasserstein GANs. Meta-t joins heterogeneous Student margins with a t copula; flows transform Gaussian, Student, Laplace, or meta-t bases. Portfolio GARCH controls use Student, empirical, or extreme-value innovations. Ten independent datasets per synthetic law and chronological five-basket and 100-stock panels separate density fit from risk accuracy. Correct specification favors meta-t on both heavy-tailed targets. Dense mixing improves observed Gaussian-flow Heston error but gives no uniform advantage. Coordinate-GARCH FHS improves five-basket scores under a targeted exploratory correction; this finding disappears when all 24 model forecasts share the broader comparison. No innovation model beats FHS under common coordinate-GARCH filtering. A restricted Heston-Wishart benchmark has a structural dispersion mismatch; its unconverged CRSP filter remains an appendix pilot. The contribution is the paired empirical design and its limits of resolution.
'''
    write('referee_round2/analysis/abstract.tex',abstract)
    p=ROOT/'main.tex';s=p.read_text()
    s=s.replace('and no primary CRSP neural improvement over FHS survives within-panel multiplicity correction.','and no primary CRSP improvement over FHS, neural or statistical, survives within-panel multiplicity correction.')
    for panel in ['crsp5','crsp100']:
        for name in ['primary','findings','secondary']:
            for prefix in ['wishart_extension/analysis/','referee_round2/analysis/']:
                s=s.replace(prefix+panel+'_'+name+'.tex','referee_round3/analysis/'+panel+'_'+name+'.tex')
    for name in ['crsp5_calibration','crsp5_robustness','crsp5_concentrated']:
        s=s.replace('wishart_extension/analysis/'+name+'.tex','referee_round2/analysis/'+name+'.tex')
    s=s.replace('figures/wishart_annual.pdf','figures/round2_annual.pdf')
    s=s.replace('The five-basket panel includes Heston--Wishart; all sixteen five-basket and fifteen 100-stock histories are supplied in the annual CSVs.','All fifteen model histories per panel are supplied in the annual CSVs; the unresolved Wishart CRSP pilot is confined to Appendix~\\ref{sec:wishart-numerics}.')
    s=s.replace('fifteen primary FHS contrasts in five baskets and fourteen in 100 stocks,','fourteen primary FHS contrasts per panel,')
    s=s.replace('The earlier augmentation added GARCH-FHS and portfolio-EWMA-FHS to thirteen specifications. Heston--Wishart now expands the five-basket set to sixteen models and fifteen FHS contrasts; the 100-stock set remains fifteen models and fourteen contrasts. Earlier twelve- and fourteen-contrast results remain archived.','The primary set contains fifteen models and fourteen FHS contrasts in each panel. The unresolved Wishart CRSP pilot is excluded from those families and from the MCS; its prior augmented inference remains archived. Earlier twelve-contrast results also remain available.')
    s=re.sub(r'The restricted Heston--Wishart benchmark has scalar-Heston ES RMSE.*?Wishart optimum\.\n\n','',s)
    old='\\input{referee_round3/analysis/crsp100_findings.tex}'
    if '\\input{referee_round3/analysis/power_table.tex}' not in s:s=s.replace(old,old+'\n'+r'''\paragraph{Resolution of the primary comparison.}
The GARCH/FHS unadjusted interval half-widths are approximately 0.05 in five baskets and 0.12 in 100 stocks. They describe interval precision, not minimum detectable effects at a specified power. A conditional location-shift diagnostic gives unadjusted 80\%-power effects of 0.074--0.077 and 0.177--0.178 respectively, rising to 0.098--0.105 and 0.255--0.256 at a conservative fourteen-contrast threshold. The observed 0.05--0.11 advantages therefore lie near the present resolution rather than establishing equivalence or superiority. A prospective design could specify 2.5\% as co-primary; the existing 2.5\% results remain secondary.
\input{referee_round3/analysis/power_table.tex}''')
    marker=r'\FloatBarrier'+'\n'+r'\section{Implications and limitations}'
    if r'\input{round3_methods.tex}' not in s:s=s.replace(marker,r'\input{round3_methods.tex}'+'\n'+marker)
    s=section(s,r'\paragraph{Multivariate Heston and Wishart scope.}',r'\section{Conclusion}',r'''\paragraph{Multivariate Heston and Wishart scope.}
The fitted Wishart restriction allows stochastic correlations but cannot match the scalar-Heston target's stationary marginal variance dispersion within its admissible shape range. Its mean reversion is also much faster than the known target. The synthetic result therefore cannot isolate a benefit or cost of stochastic correlation. The CRSP particle approximation remains unconverged at the scale of the score differences and is excluded from primary comparison and inference. Neither experiment assesses an unrestricted, adequately fitted Wishart alternative.''')
    s=section(s,r'\section{Conclusion}',r'\appendix',r'''\section{Conclusion}
The main empirical result is symmetric: at one percent, none of the fifteen primary models is distinguishable from FHS after within-panel Holm correction on 2,514 dates. GARCH has the lowest observed scores, and the matched decomposition associates most of its lead with variance filtering. Neither that lead nor the six-contrast variance-filter attribution establishes a statistical improvement. The available resolution leaves economically relevant differences unresolved. The broader retrospective comparison of all 24 risk-selected forecasts likewise rejects none of its 23 FHS contrasts.

This finding is conditional on the validation design. Final checkpoints lower observed 100-stock scores for six of seven neural models; final WGAN-G scores 1.266 versus FHS 1.269. Flow-meta-$t$ is excluded from the primary MCS ($p=0.0273$), yet likelihood selection gives it the lowest mean among the neural checkpoint variants (1.2646). These are secondary observations, not validated choices of a winning rule. MCS exclusion reflects studentized pairwise evidence and does not imply that Flow-meta-$t$ is worse than every retained model, including VAE-G.

Correct specification explains part of meta-$t$'s synthetic advantage. Under misspecification, density fit, central concentration, extreme moments, and risk accuracy can pull in different directions. WGAN-i's change from Heston RMSE 0.392 at one percent to 3.415 at 0.1 percent illustrates why single-level selection need not transfer. Dense Flow-G reduces observed Heston RMSE to 0.335, close to empirical ES at 0.352; its paired difference from empirical ES is unresolved. Coordinate-GARCH-FHS improves the five-basket score under the targeted seven-contrast correction, but loses significance in the broader 23-contrast comparison. These controls separate dependence representation from conditioning and make the multiplicity scope consequential. The empirical design, paired records, and explicit inferential limits are the principal contribution; the tail results supply representation constraints rather than a general ranking of architectures.''')
    if r'\input{round3_appendix.tex}' not in s:s=s.replace(r'\input{referee_round2/analysis/verification.tex}',r'\input{referee_round2/analysis/verification.tex}'+'\n'+r'\input{round3_appendix.tex}')
    p.write_text(s)
    p=ROOT/'expanded_crsp_methods.tex';s=p.read_text()
    if r'\citep{riskmetrics1996}' not in s and r'\citet{riskmetrics1996}' not in s:s=s.replace('initialized from the first 252 observations.','initialized from the first 252 observations. The fixed daily decay follows \\citet{riskmetrics1996}.')
    s=s.replace('We use 9,999 paired circular block-bootstrap draws with block length 20.','We use 9,999 paired circular block-bootstrap draws with block length 20 \\citep{politis1992circular}. Averaging seed scores assesses typical fitted forecasts; it does not evaluate a mixture-of-scenarios or averaged-forecast ensemble a practitioner might use.')
    s=s.replace('The augmented five-basket model set has fifteen non-FHS contrasts; the 100-stock set has fourteen. Holm adjustment is applied \\emph{within each panel}.','There are fourteen non-FHS contrasts in each panel. Holm adjustment is applied \\emph{within each panel} \\citep{holm1979}.')
    s=s.replace('over sixteen five-basket or fifteen 100-stock specifications','over fifteen specifications per panel')
    p.write_text(s)
    p=ROOT/'expanded_methods.tex';s=p.read_text()
    s=s.replace('giving filtered historical simulation (FHS).','giving filtered historical simulation (FHS) \\citep{baroneadesi1999fhs}.')
    if r'\citet{joe2005}' not in s:s=s.replace('This is a staged marginal-and-copula estimator, not full joint maximum likelihood.','Kendall-tau inversion follows \\citet{lindskog2003kendall}; the distinction between staged and joint estimation is discussed by \\citet{joe2005}. This is a staged marginal-and-copula estimator, not full joint maximum likelihood.')
    s=s.replace('Flow-G has Gaussian noise; Flow-t has an elliptical Student base with learned', 'Flow-G has an identity-covariance Gaussian base; Flow-t has an elliptical Student base with fixed identity scatter $P=I$ and learned')
    p.write_text(s)
    p=ROOT/'theory.tex';s=p.read_text();equation=r'''\begin{equation}\label{eq:constant}
k_\nu=\frac{\Gamma((\nu+1)/2)\nu^{\nu/2-1}}{\sqrt\pi\Gamma(\nu/2)},
\end{equation}'''
    if 'With\n'+equation in s:
        s=s.replace('With\n'+equation+'\nthe positive-tail coordinate asymptote is','With the Student constant defined in \\eqref{eq:constant}, the positive-tail coordinate asymptote is')
        marker=r'\begin{proposition}[Independent Student coordinates'
        pos=s.find(marker)
        if pos<0:pos=s.rfind(r'\begin{proposition}',0,s.index('Let $Z_1,Z_2$'))
        s=s[:pos]+'For later use, the standard Student positive-tail constant is\n'+equation+'\nso $\\Prob(t_\\nu>r)\\sim k_\\nu r^{-\\nu}$.\n\n'+s[pos:]
    p.write_text(s)
    p=ROOT/'wishart_methods.tex';s=p.read_text();s=s.replace('as a fitted parametric benchmark, denoted Heston--Wishart.','as a fitted parametric benchmark, denoted Heston--Wishart, in the established Wishart-process and multivariate-volatility literature \\citep{bru1991,gourieroux2006,dafonseca2008wishart}.')
    start=s.find('The new benchmark joins')
    if start<0:start=s.find('The synthetic benchmark remains')
    s=s[:start]+r'''The synthetic benchmark remains in the comparison, but the CRSP particle results are a numerical pilot in Appendix~\ref{sec:wishart-numerics}, excluded from primary scores, Holm families, and MCS membership.

\paragraph{Structural dispersion restriction.}
For a frozen, driftless stationary variance mixture $R=\sqrt V Z$, with independent $Z\sim N(0,1)$, the restricted Wishart marginal has
\[
\frac{\E R^4}{(\E R^2)^2}-3
=3\frac{\operatorname{Var}V}{(\E V)^2}=\frac6b\le1,
\qquad b\ge d+1=6.
\]
The scalar-Heston stationary gamma shape is $a=2\kappa\theta/\xi^2=0.96$, giving frozen-mixture excess $3/a=3.125$. Matching its variance coefficient of variation $1/a=1.0417$ would require $2/b=1/a$, or $b=1.92$, outside the fitted admissible class. The fitted synthetic kurtosis gap (observed 2.9--3.6 versus simulated 0.9--1.1) is consistent with this structural restriction. The upper CRSP observed range, roughly 7--14, would require $b=6/K<1$ under this same frozen approximation. The bound is not a theorem about finite-horizon leveraged returns; it diagnoses a restriction in stationary covariance dispersion. All fitted $b$ equal six, and fitted $\kappa=0.05$--$0.2$ per day contrasts with the known target's $3/252=0.0119$ per day. Consequently, this misspecified comparison cannot identify an effect attributable to stochastic correlation alone.
''' if start>=0 else s
    p.write_text(s)
    p=ROOT/'wishart_appendix.tex';s=p.read_text();s=section(s,r'\paragraph{Transform caution.}' if r'\paragraph{Transform caution.}' in s else r'\paragraph{Transform scope.}',r'\paragraph{Conditional mixture calculation.}',r'''\paragraph{Transform scope.}
The implementation uses the companion paper's stochastic equations directly. Its printed Riccati system, transform-based likelihood, and numerical risk premium are not used here. The present SDE simulation and covariance filter are assessed independently through the checks below.''')
    if r'\paragraph{Unresolved CRSP pilot.}' not in s:s+=r'''
\paragraph{Unresolved CRSP pilot.}
The guided-filter pilot has one-percent five-basket FZ0 1.520, compared with FHS 1.370, and 40 exceedances on 2,514 dates. Doubling the numerical resolution changes 2025 FZ0 by $-0.097$, exceeding the entire observed primary GARCH-EVT/FHS gap of $-0.058$. Up to 16.6\% of dates have effective particle sample size below ten. These approximations are not converged at the scale of the effects being compared. Their descriptive results and refinements remain here; no primary Holm or MCS claim uses them. This relocation concerns numerical adequacy and is not a response to an unfavorable score.
'''
    if r'\input{referee_round3/analysis/wishart_pilot.tex}' not in s:s+='\n'+r'\input{referee_round3/analysis/wishart_pilot.tex}'+'\n'
    p.write_text(s)
    p=ROOT/'referee_round2/analysis/synthetic_narrative.tex';s=p.read_text()
    if r'\paragraph{Central fit and extreme moments.}' in s:s=section(s,r'\paragraph{Central fit and extreme moments.}',r'\paragraph{Leverage within a joint weekly window.}','')
    if r'\paragraph{Central fit and extreme moments.}' not in s:s=s.replace(r'\paragraph{Leverage within a joint weekly window.}',r'''\paragraph{Central fit and extreme moments.}
Appendix~\ref{sec:heston} gives a variance-gamma \emph{frozen stationary snapshot} with shape $a=0.96$, close to the Laplace case $a=1$. Its sharp central peak offers a mechanism for a symmetric Student likelihood to trade central fit against overly heavy tails. This snapshot is not the exact dynamic daily Heston law. In the actual Heston fits, all 250 marginal degrees of freedom lie in $[2.73,3.80]$ (median 3.15), below four. Their population fourth moments are infinite: the generated sample excess kurtosis 153.7 is a sample diagnostic, not an estimated finite theoretical moment. The combination of best exact NLL 28.165 and ES RMSE 1.126 is consistent with that peak/tail compromise; likelihood rankings alone do not identify its cause.

The plain meta-$t$ law has a centrally symmetric $t$ copula and symmetric margins. For centered window returns, $R\overset d=-R$ implies $\E[R_t g(R_{t+1})]=0$ for even $g$, whenever the expectation exists. Thus centered returns have zero covariance with future centered absolute returns. The reported table uses raw absolute returns, for which fitted location shifts can introduce a small effect; its meta-$t$ value is approximately zero against reference $-0.055$. The exact symmetry restriction needs symmetric margins as well as the copula, and it does not extend to arbitrary nonlinear residual flows.

WGAN-i supplies a particularly direct level-transfer warning: its one-percent RMSE is 0.392, among the best fitted-model values under primary selection, but rises to 3.415 at 0.1\%. The training and checkpoint target is the former level only.

\paragraph{Leverage within a joint weekly window.}''')
    s=s.replace('below every parametric or neural model in this experiment.', 'below every model in the original comparison; the dense-output controls in Section~\\ref{sec:lu-controls} subsequently change the observed ordering.')
    p.write_text(s)
    p=ROOT/'referee_round2/analysis/matched_filter_findings.tex';s=p.read_text()
    if 'level-specific' not in s:s+=r'''
The EVT result is level-specific. At one percent, training samples of roughly 2,500 observations still contain about 25 tail observations, so empirical quantiles interpolate within the observed sample range. EVT can still reduce estimation error, but this comparison does not test its distinctive role in extrapolating far beyond observed extremes. The six-contrast Holm values for the total GARCH-EVT/FHS changes are 0.185 and 0.390 in five and 100 dimensions.
'''
    p.write_text(s)
    write('round3_methods.tex',r'''\subsection{Dependence representation and dense-output controls}\label{sec:lu-controls}
The likelihood-selected 100-stock NLLs are 167.309 for Flow-G, 154.195 for Flow-t, and 143.800 for Flow-meta-$t$, versus 156.116 for the full-covariance Gaussian and 137.981 for plain meta-$t$. Thus Flow-t has lower observed NLL than the Gaussian; Flow-meta-$t$ is the best flow, not the only flow to beat it. These are matched likelihood-selected states, not the risk-selected density columns. Final-state NLL deterioration (Flow-G 398.551 in 100 stocks and 58.373 in Heston) shows substantial overfitting along the retained trajectories.

The original Flow-G and Flow-t bases use $P=I$; only the Student degrees of freedom are learned. Flow-meta-$t$ instead begins with the staged fitted correlation matrix. Bounded shifts and a diagonal output map support Proposition~\ref{prop:marginal}'s coordinate-tail statement but exclude a general dense affine Gaussian map from that parametrization. This is a representation/guarantee trade-off, not proof that it causes the observed likelihood ordering.

We add a trainable dense output $A=L_AU_A$ to Flow-G and Flow-t, with unit lower-triangular $L_A$, upper-triangular $U_A$, and positive exponential diagonal. Triangular solves give the inverse and $\log|\det A|=\sum_i\log (U_A)_{ii}$; this standard invertible mixing parametrization is used in \citet{kingma2018glow}. Initial $A$ is the Cholesky factor of the training standardized covariance with $10^{-6}I$ ridge. The coupling initialization, data, seed, 3,000 updates, 250-step checkpoints, and validation draws match the original runs. The control changes both representational capacity and its covariance initialization; it does not isolate optimization from architecture. For each fixed nonsingular $A$, radial two-sided linear-growth results survive. Arbitrary mixing need not preserve heterogeneous coordinate indices.

Sixty new fits cover two bases, ten Heston datasets and ten folds in each CRSP panel. We evaluate one-percent risk, likelihood, and final states using common scenario seeds. Tables~\ref{tab:lu-crsp5}--\ref{tab:lu-heston} compare paired outcomes. The six CRSP head/rule contrasts form a separate exploratory Holm family per panel and tail level. Dataset-paired synthetic MSE differences carry dispersion SEs.
\input{referee_round3/analysis/crsp5_lu.tex}
\input{referee_round3/analysis/crsp100_lu.tex}
\input{referee_round3/analysis/heston_lu.tex}
\FloatBarrier\par
\input{referee_round3/analysis/lu_findings.tex}

\subsection{Coordinate-level conditioning controls}\label{sec:coordinate-controls}
We fit a constant-mean GARCH(1,1)-$t$ independently to every coordinate in each training fold, using the existing three-start estimator and parameter bounds. The new filter states start from the first 252 training observations and continue predictably through validation and test; parameters remain fixed. FHS, Gaussian, meta-$t$, Flow-G, Flow-t, and Flow-meta-$t$ are refitted to the resulting joint standardized innovations. The neural controls retain seed zero, budgets and primary one-percent FZ0 checkpoint selection. A second control fits each coordinate's EWMA decay in $[0.8,0.9995]$ by training Gaussian quasi-likelihood, using the training mean, and then fits empirical innovations. Neither decay nor GARCH parameters use held-out data.

The two panels require 1,050 scalar GARCH fits and 1,050 fitted-decay EWMA fits. Their 140 model/fold comparisons include sixty new neural fits. Changes against the original EWMA versions include location estimation and model refitting, so they are not the exact variance-only attribution above. Seven such contrasts, five innovation contrasts against coordinate-GARCH-FHS, and seven contrasts against portfolio-GARCH-FHS form three separately adjusted exploratory families per panel and level. The original primary family remains unchanged. Full 2.5\% results are archived as secondary diagnostics.
\input{referee_round3/analysis/crsp5_coordinate.tex}
\input{referee_round3/analysis/crsp100_coordinate.tex}
\FloatBarrier\par
\input{referee_round3/analysis/coordinate_findings.tex}
''')
    write('round3_appendix.tex',r'''\section{Resolution, set composition, and additional controls}\label{sec:round3-diagnostics}
The MCS compares studentized pairwise differences. A model with a worse mean can remain when its differences are noisy, while a model with a better mean can be excluded through a more precise adverse pair. Flow-meta-$t$ in 100 stocks has mean 1.334 versus VAE-G 1.377; its unadjusted difference from VAE-G is $-0.043$, interval $[-0.191,0.138]$. Its most adverse standardized comparison is with plain meta-$t$: the difference is 0.085 with bootstrap SE 0.025. Its FHS comparison is less precise (SE 0.041), and does not reject individually at five percent. Hence the MCS result should not be described simply as a precise FHS rejection.

Removing EWMAcov-FHS and portfolio-EWMA-FHS leaves membership unchanged for all thirteen remaining models; the 100-stock Flow-meta-$t$ value changes from 0.027 to 0.024. The near-duplicate variants therefore do not explain its exclusion in this sensitivity. No MCS value is a probability that a model is optimal.
\input{referee_round3/analysis/mcs_table.tex}
\input{referee_round3/analysis/pairwise_table.tex}
\input{referee_round3/analysis/within_garch.tex}
\input{referee_round3/analysis/coordinate_portfolio.tex}
\input{referee_round3/analysis/filter_parameters.tex}
\input{referee_round3/analysis/augmented_24.tex}

\paragraph{Staged copula estimates.}
The original annual copula degrees of freedom average 7.35 (range 6.52--8.84) in five baskets and 29.27 (28.33--30.35) in 100 stocks. These are staged profile estimates conditional on marginal fits and Kendall-tau correlation projection, not joint-MLE estimates or evidence that dimension itself changes a common true parameter. The panels also contain different underlying return coordinates. Complete fold parameters, including the new coordinate filters, remain in the machine-readable records.
''')
    p=ROOT/'referee_round2/analysis/selection_findings.tex';s=p.read_text()
    if 'six of seven' not in s:s+=r'''
In 100 stocks, final states improve the observed score for six of seven neural models, including WGAN-G at 1.266 versus FHS 1.269. Heston WGAN-G reaches secondary RMSE 0.194 and VAE-tlatent 0.149, compared with their primary 0.485 and 0.394. These reversals show fragility of sparse-tail checkpoint selection; they do not establish that a rule chosen after viewing test results would dominate out of sample.
'''
    p.write_text(s)
    p=ROOT/'referee_round2/analysis/verification.tex';s=p.read_text();s=s.replace('All 126 recorded checks pass','All 126 retained checks pass');s=s.replace('The primary expanded design now has 730 model evaluations','The retained pre-Wishart expanded design has 730 model evaluations')
    s=s.replace('The prior forecast files are unchanged; all five-basket inference and related tables are regenerated for the augmented model set.','The prior forecast files are unchanged. Including these twenty fits gives 750 retained model evaluations; ten Wishart CRSP fits are now appendix-only numerical pilots and are excluded from primary inference.')
    if r'\paragraph{Dense-output and coordinate-filter extension.}' not in s:s+=r'''
\paragraph{Dense-output and coordinate-filter extension.}
The current extension adds sixty dense-flow and 140 coordinate-filter model/fold comparisons, for 950 cumulative comparisons including pilots. Of these additions, 120 are new neural fits. The 2,100 scalar coordinate-filter estimations are counted separately. Selected likelihood, risk and final states are evaluations of those same trajectories, not independent fits. Independent dense-map inverse, Jacobian, Gaussian-embedding and forecast-predictability checks are accompanied by complete-result accounting and reconstructed score checks. The current output workflow ends with the third-round analysis and integration processors so archived editorial layers cannot overwrite this revision.
'''
    p.write_text(s)
    p=ROOT/'README.md';s=p.read_text()
    s=re.sub(r'The preceding editorial pass.*?See `FINAL_POLISH.md`\.\n\n','',s,flags=re.S)
    s=re.sub(r'## Current third-round revision.*?(?=## )','',s,flags=re.S)
    a=s.find('## Fitted Heston-Wishart addition');b=s.find('## What changed',a)
    if a<0:a=s.index('## What changed');b=a
    if a>=0:s=s[:a]+"""## Current third-round revision

The primary comparison contains fifteen models and fourteen FHS contrasts in each CRSP panel. No neural or statistical improvement survives that correction. The Wishart CRSP filter is numerically unresolved and appears only as an appendix pilot; its synthetic comparison remains, with an explicit stationary-dispersion restriction.

`referee_round3/` contains the protocol, baseline source, sixty dense-LU flow fits, 140 coordinate-filter model/fold comparisons, 1,050 scalar GARCH and 1,050 fitted-decay EWMA fits, paired inference and checks. New results are retrospective controls with separately declared multiplicity families. All original forecast files remain unchanged. The matched likelihood comparison specifies identity scatter for Flow-t and distinguishes dense mixing from coordinate-index preservation.

Current analyses are `code/round3_analysis.py` and `code/round3_control_analysis.py`; `code/integrate_round3.py` is the final manuscript processor. Rebuild the complete current output using `python code/run_all.py --mode outputs --pdf`, rather than invoking an earlier editorial processor alone. Recompute new fits with `python code/run_round3_experiments.py --jobs 3` in an isolated clean copy.

"""+s[b:]
    s=s.replace('The primary neural comparisons do not improve FHS after within-panel Holm correction.','Neither neural nor statistical primary forecasts improve detectably on FHS after within-panel Holm correction.')
    s=s.replace('730 retained primary evaluations and 520 fresh primary neural fits:', '730 pre-Wishart expanded evaluations and 520 fresh neural fits:')
    s=s.replace('The current primary Holm families contain fifteen five-basket and fourteen 100-stock FHS contrasts;', 'The current primary Holm families contain fourteen FHS contrasts per panel;')
    s=s.replace('Experiment mode completes the expanded design and second-round additions,','Experiment mode completes the expanded design and all subsequent additions,')
    s=s.replace('| `ROUND2_RESPONSE.md` | Point-by-point response to the latest report |','| `referee_round3/response.json` | Current point-by-point response and corrections |\n| `ROUND2_RESPONSE.md` | Earlier round-two response |')
    s=re.sub(r'The twenty Wishart additions bring.*?independent fits\.\n\n','',s,flags=re.S)
    s=s.replace('## Executed design and verification','## Executed design and verification\n\nThe twenty Wishart additions bring the retained count to 750 (including ten numerical CRSP pilots); 200 new dense-output/coordinate-filter comparisons bring the cumulative count to 950. The 2,100 scalar coordinate-filter fits are additional submodel estimations. New neural fits number 120; multiple selected-state evaluations are not counted as independent fits.')
    p.write_text(s)
    write('PUBLICATION.md',r"""# Publication record

Reserved DOI: 10.5281/zenodo.24024955
Overleaf: https://www.overleaf.com/project/6aee6e8f84dcc0b2d4647dab
Zenodo draft: https://zenodo.org/uploads/24024955

Current revision: symmetric primary inference; sparse-checkpoint sensitivity; Wishart dispersion proof and appendix-only CRSP numerical pilot; central-density/tail-moment explanation; dense-LU and coordinate-filter controls; MCS precision diagnostics; updated references and counts. The date remains \today. The companion Wishart transform is not used.

Zenodo is a saved draft with one manuscript PDF. Do not publish without explicit publication authorization. The compact editable source belongs in Overleaf and the separate source bundle; licensed inputs and checkpoints are not uploaded to Zenodo.
""")
    bibliography()
    print('Third-round text integrated; abstract words',len(abstract.split()))
def bibliography():
    p=ROOT/'refs.bib';s=p.read_text();s=s.replace("and Osband's Principle","and {Osband}'s Principle").replace('in the Heston Model','in the {Heston} Model').replace("{Kendall's Tau", "{{Kendall}'s Tau")
    s=s.replace('note = {Working paper, September 22},','note = {Working paper, September 17},\n  doi = {10.2139/ssrn.1474728},').replace('https://www.researchgate.net/publication/228290665_Efficient_Simulation_of_the_Wishart_Model','https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1474728')
    refs={
    'baroneadesi1999fhs':r'''@article{baroneadesi1999fhs, author={Barone-Adesi, Giovanni and Giannopoulos, Kostas and Vosper, Les}, title={{VaR} without Correlations for Portfolios of Derivative Securities}, journal={Journal of Futures Markets}, volume={19}, number={5}, pages={583--602}, year={1999}, url={https://susi.usi.ch/rerodoc/9092/files/wp9904.pdf}}''',
    'riskmetrics1996':r'''@manual{riskmetrics1996, author={{J.P. Morgan} and {Reuters}}, title={{RiskMetrics}: Technical Document}, edition={Fourth}, year={1996}, url={https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}}''',
    'holm1979':r'''@article{holm1979, author={Holm, Sture}, title={A Simple Sequentially Rejective Multiple Test Procedure}, journal={Scandinavian Journal of Statistics}, volume={6}, number={2}, pages={65--70}, year={1979}, url={https://www.jstor.org/stable/4615733}}''',
    'politis1992circular':r'''@incollection{politis1992circular, author={Politis, Dimitris N. and Romano, Joseph P.}, title={A Circular Block-Resampling Procedure for Stationary Data}, booktitle={Exploring the Limits of Bootstrap}, editor={LePage, Raoul and Billard, Lynne}, publisher={Wiley}, pages={263--270}, year={1992}}''',
    'bru1991':r'''@article{bru1991, author={Bru, Marie-France}, title={{Wishart} Processes}, journal={Journal of Theoretical Probability}, volume={4}, number={4}, pages={725--751}, year={1991}, doi={10.1007/BF01259552}}''',
    'gourieroux2006':r'''@article{gourieroux2006, author={Gourieroux, Christian}, title={Continuous Time {Wishart} Process for Stochastic Risk}, journal={Econometric Reviews}, volume={25}, number={2--3}, pages={177--217}, year={2006}, doi={10.1080/07474930600713234}}''',
    'joe2005':r'''@article{joe2005, author={Joe, Harry}, title={Asymptotic Efficiency of the Two-Stage Estimation Method for Copula-Based Models}, journal={Journal of Multivariate Analysis}, volume={94}, number={2}, pages={401--419}, year={2005}, doi={10.1016/j.jmva.2004.06.003}}''',
    'kingma2018glow':r'''@inproceedings{kingma2018glow, author={Kingma, Diederik P. and Dhariwal, Prafulla}, title={{Glow}: Generative Flow with Invertible 1x1 Convolutions}, booktitle={Advances in Neural Information Processing Systems}, volume={31}, year={2018}, url={https://papers.nips.cc/paper_files/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html}}'''}
    for key,val in refs.items():
        if '{'+key+',' not in s:s+='\n'+val+'\n'
    p.write_text(s)
if __name__=='__main__':main()

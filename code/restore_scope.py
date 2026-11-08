"""Restore the paper's multivariate parametric/nonparametric research scope."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
TITLE='Parametric and Nonparametric Multivariate Models in Finance'
DOI='10.5281/zenodo.24024955'

ABSTRACT='''We compare parametric distributions, empirical nonparametric estimators, and neural generators for multivariate financial modeling. Parametric baselines fit a Gaussian or meta-t distribution; meta-t joins Student-t margins with potentially different tail indices through a t copula. Empirical estimators use the observed joint distribution; filtered historical simulation (FHS) combines empirical standardized innovations with conditional volatility. Normalizing flows transform base distributions invertibly to obtain exact densities; we test Gaussian, Student-t, Laplace, and residual meta-t variants. Variational autoencoders generate observations through stochastic latent-variable decoders, while Wasserstein GANs learn samplers through an adversarial critic. Portfolio-level GARCH benchmarks combine estimated volatility with Student-t innovations, empirical residuals, or extreme-value tails. Synthetic experiments use ten independent datasets for each Student-t, heterogeneous meta-t, and multivariate Heston target; the Heston system incorporates correlated stochastic variances and leverage. Empirical experiments use chronological CRSP panels in five and 100 dimensions. Meta-t benefits from correct specification in the first two synthetic laws. Empirical expected shortfall leads under primary Heston selection, although alternative checkpoints reverse that ranking. No tested neural model improves on FHS after within-panel multiplicity correction in the primary CRSP comparisons. Matched controls associate most of the statistical-benchmark advantage with volatility filtering and detect no incremental extreme-value gain. Density fit, tail representation, and calibrated portfolio risk require separate evidence. The author's Heston-Wishart model, with stochastic variances and correlations, is discussed as a parametric extension outside these tests.
'''

INTRO=r'''\section{Introduction}
Multivariate financial models describe marginal risks and their dependence, then translate that joint law into portfolio outcomes. The central question of this study is when a structured parametric distribution is sufficient and when empirical or neural flexibility improves density estimation and risk forecasting. We compare these approaches through joint density fit, tail representation, finite-level expected-shortfall error, and conditional VaR--ES scores. Volatility conditioning is one component of this comparison.

\paragraph{Model classes and scope.}
The fitted joint parametric baselines are Gaussian and meta-$t$ laws. The multivariate Student law supplies a controlled target and neural base distributions; GARCH-t supplies a portfolio-level conditional benchmark. The nonparametric estimators use the empirical distribution and resampling. Neural generators form a separate flexible class: a fixed network architecture also has finitely many parameters, so neural and nonparametric are not synonyms. Filtered historical simulation (FHS) combines a volatility filter with empirical innovations; the filter and innovation law are separate modeling choices. Portfolio-level GARCH benchmarks evaluate conditional risk forecasts without supplying a competing multivariate joint law.

Normalizing flows provide tractable densities, variational autoencoders (VAEs) provide latent-variable representations, and generative adversarial networks (GANs) provide samplers trained without an explicit likelihood. All three have applications to financial scenarios and risk estimation \citep{papamakarios2021review,kingma2014vae,goodfellow2014gan,wiese2020quantgan,buehler2020simulator}. Their flexibility changes the restrictions placed on the joint distribution, but density fit and portfolio risk remain distinct evaluation targets.

Three theoretical distinctions guide the comparison. First, an asymptotic tail statement for a fixed network does not determine its attainable error at a fixed probability level. Second, a joint radial tail index differs from a vector of marginal indices. Third, independent Student coordinates and an elliptical Student vector have different extreme dependence and different consequences under nonlinear maps.

\Needspace{10\baselineskip}
Our contributions are:
\begin{enumerate}
\item A paired comparison of structured parametric, empirical nonparametric, and flexible neural multivariate models, using ten datasets per synthetic law and chronological CRSP panels in five and 100 dimensions, with matched checkpoint and filter controls.
\item A risk-relevant distinction between elliptical and independent Student latent noise, with a one-sided loss-projection theorem and a heterogeneous-index counterexample.
\item A tractable residual meta-$t$ flow with explicit marginal-index guarantees and an exact statement of which marginal and copula properties it does not preserve.
\end{enumerate}
The controlled simulations provide known or Monte Carlo reference laws. CRSP supplies realized outcomes but no observable true conditional ES, so proper scores and identification moments evaluate forecasts. The empirical findings are conditional on these designs: correct specification favors meta-$t$, the empirical estimator is a strong synthetic benchmark, and no primary CRSP neural improvement over FHS survives within-panel multiplicity correction. Matched filter controls help explain the conditional-forecast results within this broader model comparison.

'''

def main():
 p=ROOT/'main.tex';s=p.read_text()
 s=re.sub(r'\\title\{[^\n]*\}',lambda _:r'\title{'+TITLE+'}',s,count=1)
 s=re.sub(r'\\date\{[^\n]*\}',lambda _:r'\date{\today\\[0.4em]\small DOI: \href{https://doi.org/'+DOI+'}{'+DOI+'}}',s,count=1)
 a=s.index(r'\section{Introduction}');b=s.index(r'\paragraph{Relation to existing work.}',a)
 s=s[:a]+INTRO+s[b:]
 s=s.replace('The evidence gives volatility conditioning priority in these fitted comparisons.','The comparison separates distributional restrictions, neural flexibility, and conditional filtering.')
 if r'\paragraph{Multivariate Heston and Wishart scope.}' not in s:
  s=s.replace(r'\section{Conclusion}','\\paragraph{Multivariate Heston and Wishart scope.}\nThe simulated Heston target uses one Cox--Ingersoll--Ross variance process per asset, correlated variance and price shocks, and leverage. With fixed price-shock correlation matrix $C_S$, its instantaneous return covariance is $D_V(t)C_SD_V(t)$, where $D_V(t)=\\operatorname{diag}(\\sqrt{V_1(t)},\\ldots,\\sqrt{V_d(t)})$. Thus stochastic marginal variances change covariances while instantaneous return correlations remain fixed. The matrix extension is the Heston--Wishart model in Section~4 of \\citet{noguer2025wishart}: the full positive-semidefinite covariance matrix follows a Wishart diffusion, and correlated return and covariance shocks encode leverage. This adds stochastic return correlations, a central feature of Wishart covariance models \\citep{dafonseca2007correlations}. The companion paper studies portfolio laws and tail risk; its numerical illustrations use a separate protocol from the forecasting comparisons here. The Heston--Wishart specification requires its own matched fitting and forecasting experiment before any performance ranking can be reported.\n\n'+r'\section{Conclusion}',1)
 a=s.index(r'\section{Conclusion}');b=s.index(r'\appendix',a)
 old=s[a:b]
 example=old[old.index('Validation design materially changes'):].split('\n\n',1)[0]
 if ' Meta-$t$ under correct specification' in example:
  example=example[:example.index(' Meta-$t$ under correct specification')]
 example=example.strip()+'\n\n'
 conclusion=r'''\section{Conclusion}
The parametric--nonparametric comparison in multivariate finance depends on which aspect of the joint law is being evaluated. Structured meta-$t$ laws perform well when correctly specified; empirical ES is a strong benchmark under misspecified synthetic models; neural flexibility can improve some density or secondary risk comparisons without establishing a general forecasting advantage. Tail representation, density fit, and calibrated portfolio risk require separate evidence.

Within the conditional CRSP experiments, no neural improvement over FHS survives the primary or checkpoint-sensitivity within-panel multiplicity correction. Most of the observed statistical-benchmark lead is associated with volatility conditioning, and empirical residuals are competitive with EVT under the same GARCH filter. These results describe the tested filters, datasets, and validation rules.

'''
 conclusion+=example
 conclusion+=r'''The residual meta-$t$ flow connects structured marginal tails with flexible multivariate dependence. Together with the empirical benchmarks and matched controls, it provides a way to assess whether added flexibility improves the financial quantity of interest. The choice between distributional restrictions and flexibility should therefore be evaluated jointly with specification, conditioning, sample size, and validation.

'''
 s=s[:a]+conclusion+s[b:];p.write_text(s)
 (ROOT/'referee_round2/analysis/abstract.tex').write_text(ABSTRACT)

 p=ROOT/'README.md';s=p.read_text();s=s.replace(s.splitlines()[0],'# '+TITLE,1)
 if 'DOI: https://doi.org/'+DOI not in s:
  s=s.replace('\n\n','\n\nDOI: https://doi.org/'+DOI+'\n\n',1)
 s=s.replace('the final 37-page editorial revision of the completed second referee study.','the current revision restoring the study of parametric and nonparametric multivariate modeling in finance.')
 scope='The current framing is described in `SCOPE_RESTORATION.md`: structured parametric laws, empirical nonparametric estimators, and flexible neural models are compared on density fit, dependence, and portfolio risk. Volatility conditioning is a finding within this comparison.\n\n'
 if 'The current framing is described' not in s:s=s.replace('## What changed',scope+'## What changed',1)
 s=s.replace('The final editorial pass adds','The preceding editorial pass added')
 p.write_text(s)
 p=ROOT/'FINAL_POLISH.md';s=p.read_text()
 if not s.startswith('# Previous editorial pass'):
  s=s.replace('# Final editorial response','# Previous editorial pass',1)
  s=s.replace('\n\nTitle:', '\n\nThe broader framing requested subsequently is recorded in `SCOPE_RESTORATION.md`; the title below documents the preceding pass.\n\nTitle:',1)
 p.write_text(s)
 p=ROOT/'REVISION_AUDIT.md';s=p.read_text()
 if not s.startswith('# Current scope correction'):
  s='# Current scope correction\n\nSee `SCOPE_RESTORATION.md` for the restored multivariate model comparison. Numerical experiments and formal mathematics are unchanged.\n\n'+s
 p.write_text(s)
 (ROOT/'SCOPE_RESTORATION.md').write_text(f'''# Restored research scope

Title: **{TITLE}**.

The study compares structured parametric distributions, empirical nonparametric estimators, and flexible neural multivariate models in finance. The title, abstract, introduction, contributions and conclusion now state this objective. Volatility conditioning remains an empirical finding within the comparison.

The terminology distinguishes empirical nonparametric estimators from neural models with fixed finite architectures. It also distinguishes joint multivariate models from portfolio-level conditional GARCH controls, and the volatility filter from the innovation distribution in FHS.

All 36 numerical table bodies, both figures, formal mathematical results and experimental outputs are retained. The negative neural/FHS result, correct-specification qualification, likelihood/risk distinction, checkpoint sensitivity and the current MCS value of 0.0273 remain explicit. No experiments or inferential calculations were rerun for this framing correction.

Reproduction applies `code/restore_scope.py` after the prior editorial postprocessor. The preceding 37-page framing is preserved in `scope_revision/baseline/`.
''')
 print(TITLE)
 print('Abstract words:',len(ABSTRACT.split()))

if __name__=='__main__':main()

"""Idempotent manuscript integration after the previous editorial processors."""
from pathlib import Path
import json,re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];O=ROOT/'wishart_extension/analysis'


def main():
    cs=pd.read_csv(O/'crsp5_summary.csv');ss=pd.read_csv(O/'synthetic_summary.csv')
    w=cs[(cs.model=='Heston-Wishart')&(cs.alpha==.01)].iloc[0]
    h=ss[(ss.model=='Heston-Wishart')&(ss.alpha==.01)].iloc[0]
    verification=json.loads((O/'verification.json').read_text())
    abstract='''We compare parametric distributions, empirical nonparametric estimators, and neural generators for multivariate financial modeling. Parametric models include Gaussian and meta-t laws and a restricted Heston-Wishart diffusion with stochastic covariance and leverage. Meta-t combines heterogeneous Student margins through a t copula. Empirical estimators use observed joint returns; filtered historical simulation (FHS) rescales empirical innovations by conditional volatility. Normalizing flows use Gaussian, Student, Laplace, or residual meta-t bases; variational autoencoders use latent-variable decoders; Wasserstein GANs learn adversarial samplers. Portfolio GARCH controls use Student innovations, empirical residuals, or extreme-value tails. Ten independent datasets per synthetic law and chronological five-basket and 100-stock CRSP panels separate density fit from portfolio-risk accuracy. Heston-Wishart is fitted to the five-asset studies by simulated moments and evaluated conditionally with a covariance-state particle filter. Correct specification favors meta-t on its synthetic targets. Empirical ES leads under primary scalar-Heston selection, including the misspecified Wishart benchmark. No primary neural improvement over FHS survives within-panel multiplicity correction. Matched controls associate most of the statistical-benchmark advantage with volatility filtering and detect no incremental extreme-value gain. Wishart results depend on the restricted calibration and numerical filter; its addition supplies no general family ranking.
'''
    (ROOT/'referee_round2/analysis/abstract.tex').write_text(abstract)
    p=ROOT/'main.tex';s=p.read_text()
    s=s.replace('The fitted joint parametric baselines are Gaussian and meta-$t$ laws.', 'The fitted joint parametric baselines are Gaussian and meta-$t$ laws, supplemented in the five-asset studies by a restricted Heston--Wishart stochastic-covariance model.')
    if r'\input{wishart_methods.tex}' not in s:s=s.replace(r'\input{expanded_methods.tex}',r'\input{expanded_methods.tex}'+'\n'+r'\input{wishart_methods.tex}')
    names=['crsp5_primary','crsp5_findings','crsp5_secondary','crsp5_calibration','crsp5_robustness','crsp5_concentrated','heston_decomposition','heston_dependence_table']
    for name in names:s=s.replace('referee_round2/analysis/'+name+'.tex','wishart_extension/analysis/'+name+'.tex')
    s=s.replace('figures/round2_annual.pdf','figures/wishart_annual.pdf')
    s=s.replace('All fifteen model histories are supplied in the annual CSVs.', 'The five-basket panel includes Heston--Wishart; all sixteen five-basket and fifteen 100-stock histories are supplied in the annual CSVs.')
    s=s.replace('fourteen primary FHS contrasts per panel,', 'fifteen primary FHS contrasts in five baskets and fourteen in 100 stocks,')
    start=s.index(r'\paragraph{Multivariate Heston and Wishart scope.}');end=s.index(r'\section{Conclusion}',start)
    scope=r'''\paragraph{Multivariate Heston and Wishart scope.}
The scalar-Heston target has correlated marginal CIR variances and a fixed instantaneous return-correlation matrix. The fitted Heston--Wishart SDE instead allows the whole covariance matrix, including correlations, to evolve. The two laws are different: adding the Wishart benchmark does not make the controlled Heston experiment correctly specified. Its scalar mean-reversion restriction, integer shape grid, moment-calibration objective and particle filter are part of the evaluated specification. Boundary estimates and simulation-to-data moment gaps limit any conclusion about unrestricted Wishart models. The current addition covers five assets; the 100-stock model set is unchanged.

'''
    s=s[:start]+scope+s[end:]
    old='The augmented comparison adds GARCH-FHS and portfolio-EWMA-FHS to the previous thirteen specifications. Fourteen-contrast Holm results and fifteen-model MCS results supersede the earlier primary tables; the twelve-contrast arithmetic remains available for audit.'
    new='The earlier augmentation added GARCH-FHS and portfolio-EWMA-FHS to thirteen specifications. Heston--Wishart now expands the five-basket set to sixteen models and fifteen FHS contrasts; the 100-stock set remains fifteen models and fourteen contrasts. Earlier twelve- and fourteen-contrast results remain archived.'
    s=s.replace(old,new)
    conclusion=f'The restricted Heston--Wishart benchmark has scalar-Heston ES RMSE {h.rmse:.3f} and five-basket FZ0 {w.fz0:.3f}. Its FHS score difference is {w.delta_fhs:.3f}, with fifteen-contrast Holm $p={w.p_holm:.3f}$. These findings describe the fitted restriction and covariance filter, rather than an unrestricted Wishart optimum.\n\n'
    marker='The residual meta-$t$ flow connects structured marginal tails with flexible multivariate dependence.'
    # Remove the generated sentence if this processor is called repeatedly.
    s=re.sub(r'The restricted Heston--Wishart benchmark has scalar-Heston ES RMSE.*?Wishart optimum\.\n\n','',s)
    s=s.replace(marker,conclusion+marker,1)
    if r'\input{wishart_appendix.tex}' not in s:s=s.replace(r'\input{referee_round2/analysis/verification.tex}',r'\input{wishart_appendix.tex}'+'\n'+r'\FloatBarrier'+'\n'+r'\input{referee_round2/analysis/verification.tex}')
    p.write_text(s)
    p=ROOT/'referee_round2/analysis/synthetic_narrative.tex';s=p.read_text()
    s=s.replace('A correctly specified Heston likelihood or particle-filter benchmark remains outside this design.', 'The additional Heston--Wishart benchmark is fitted and evaluated below; it is also misspecified for this scalar-Heston target.')
    s=s.replace(r'\input{referee_round2/analysis/heston_risk_table.tex}',r'\input{wishart_extension/analysis/heston_risk_table.tex}')
    if r'\input{wishart_extension/analysis/heston_findings.tex}' not in s:s=s.replace(r'\input{wishart_extension/analysis/heston_risk_table.tex}',r'\input{wishart_extension/analysis/heston_risk_table.tex}'+'\n'+r'\input{wishart_extension/analysis/heston_findings.tex}')
    s=s.replace('figures/round2_paired_mse.pdf','figures/wishart_paired_mse.pdf').replace('The Heston panel includes the three added controls.', 'The Heston panel includes the fitted Heston--Wishart benchmark.')
    p.write_text(s)
    p=ROOT/'expanded_crsp_methods.tex';s=p.read_text()
    s=s.replace('Fourteen non-FHS contrasts in the augmented model set receive Holm adjustment \\emph{within each panel}.', 'The augmented five-basket model set has fifteen non-FHS contrasts; the 100-stock set has fourteen. Holm adjustment is applied \\emph{within each panel}.')
    s=s.replace('A 95\\% model confidence set over the fifteen augmented specifications uses', 'A 95\\% model confidence set over sixteen five-basket or fifteen 100-stock specifications uses')
    s=s.replace('For the five-basket panel, daily scores are averaged over the two seeds before time-series inference.', 'For the five-basket panel, daily scores are averaged over the two seeds where available before time-series inference; deterministic or singly fitted benchmarks have one series.')
    p.write_text(s)
    p=ROOT/'expanded_methods.tex';s=p.read_text();s=s.replace('There are 100 Student, 110 heterogeneous meta-$t$, and 120 Heston model evaluations.', 'The retained design has 100 Student, 110 heterogeneous meta-$t$, and 120 Heston model evaluations; ten additional Heston--Wishart fits expand the Heston comparison to 130.')
    p.write_text(s)
    p=ROOT/'referee_round2/analysis/verification.tex';s=p.read_text()
    extra='\\paragraph{Executed Wishart extension.}\nTwenty additional parametric fits use the same ten Heston datasets and ten annual five-basket folds. They add ten million primary synthetic scenarios and 15,084 portfolio/level/date forecasts over 2,514 CRSP test dates. Six numerical-refinement evaluations reuse fitted parameters. Analytic covariance, scalar-limit, Gaussian-mixture, characteristic-function and forecast-timing checks pass. The prior forecast files are unchanged; all five-basket inference and related tables are regenerated for the augmented model set. The extension is reproduced by \\path{code/run_wishart.py}, \\path{code/check_wishart.py}, and \\path{code/analyze_wishart.py}, and is included in the default output reconstruction.\n'
    if r'\paragraph{Executed Wishart extension.}' not in s:s+='\n'+extra
    p.write_text(s)
    p=ROOT/'refs.bib';s=p.read_text()
    if '@unpublished{gauthier2009wishart' not in s:s+='''
@unpublished{gauthier2009wishart,
  author = {Gauthier, Pierre and Possama{\\"i}, Dylan},
  title = {Efficient Simulation of the {Wishart} Model},
  year = {2009},
  note = {Working paper, September 22},
  url = {https://www.researchgate.net/publication/228290665_Efficient_Simulation_of_the_Wishart_Model}
}
'''
    p.write_text(s)
    p=ROOT/'refs.bib';s=p.read_text()
    if '@article{pitt1999auxiliary' not in s:s+='''
@article{pitt1999auxiliary,
  author = {Pitt, Michael K. and Shephard, Neil},
  title = {Filtering via Simulation: Auxiliary Particle Filters},
  journal = {Journal of the American Statistical Association},
  year = {1999}, volume = {94}, number = {446}, pages = {590--599},
  doi = {10.1080/01621459.1999.10474153}
}
'''
    p.write_text(s)
    p=ROOT/'README.md';s=p.read_text()
    addition=f'''## Fitted Heston-Wishart addition

The current manuscript adds twenty historical-return-calibrated Heston-Wishart fits: ten scalar-Heston datasets and ten five-basket CRSP folds. This is a restricted full-matrix covariance SDE with diagonal leverage, fitted by a prespecified simulated-moment grid. It is not unrestricted Wishart MLE, a pricing calibration, or a fitted model for the 100-stock panel. The source preprint's printed Riccati formula is not used.

Heston ES RMSE is {h.rmse:.6f}; the five-basket one-percent FZ0 is {w.fz0:.6f}, with FHS difference {w.delta_fhs:.6f} and fifteen-contrast Holm p {w.p_holm:.6f}. Five-basket inference now contains sixteen models. The earlier study and all its forecasts remain archived. Current augmented tables/results and the frozen extension protocol are in `wishart_extension/`.

Run `python code/run_wishart.py --study heston --units 0 1 2 3 4 5 6 7 8 9` and the analogous `--study crsp5 --units 2016 ... 2025` to reproduce primary additions. Add `--refined` for Heston datasets 0, 1, 2 and CRSP folds 2016, 2020, 2025. Then run `code/check_wishart.py`, `code/analyze_wishart.py`, and `code/integrate_wishart.py`. The default `code/run_all.py` workflow includes this extension.

'''
    if '## Fitted Heston-Wishart addition' in s:
        a=s.index('## Fitted Heston-Wishart addition');b=s.index('## ',a+4);s=s[:a]+s[b:]
    pos=s.index('## What changed');s=s[:pos]+addition+s[pos:]
    s=s.replace('No correctly specified Heston likelihood or particle filter is fitted.', 'No correctly specified scalar-Heston likelihood is fitted; the additional Wishart covariance filter is a different model.')
    s=s.replace('The primary Holm families contain fourteen FHS contrasts per panel;', 'The current primary Holm families contain fifteen five-basket and fourteen 100-stock FHS contrasts;')
    p.write_text(s)
    p=ROOT/'PUBLICATION.md';s=p.read_text();s=s.replace('A matched Wishart forecasting experiment has not been performed in this study. Numerical results are unchanged.', 'The fitted Wishart extension now evaluates ten Heston datasets and ten CRSP five-basket folds. The five-basket inference, synthetic comparisons, figures and numerical diagnostics are updated; no 100-stock Wishart fit is claimed.')
    p.write_text(s)
    print('Wishart integration complete; abstract words',len(abstract.split()),'; verification',verification)

if __name__=='__main__':main()

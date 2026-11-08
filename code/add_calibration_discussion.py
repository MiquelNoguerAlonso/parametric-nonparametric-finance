"""Apply the third integrated-paper review using existing calibration outputs.

Run after revise_integrated_paper.py and polish_integrated_review.py.
No models, forecasts, tests or numerical tables are recomputed.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def replace(name, old, new):
    p = ROOT / name
    text = p.read_text()
    if old in new and new in text:
        return
    if old in text:
        p.write_text(text.replace(old, new))
    else:
        assert new in text, (name, old)


cal = pd.read_csv(ROOT / 'referee_round4/analysis/calibration.csv')
for panel, expected in [('crsp5', .271), ('crsp100', .394)]:
    fhs = cal[(cal.panel == panel) & (cal.model == 'GARCH-FHS')].iloc[0]
    evt = cal[(cal.panel == panel) & (cal.model == 'GARCH-EVT')].iloc[0]
    assert fhs.cc_p < .001 and round(evt.cc_p, 3) == expected
assert cal[(cal.panel == 'crsp100') & (cal.model == 'EWMAcov-FHS')].iloc[0].hits == 14

discussion = r'''\paragraph{Coverage and conditional calibration.}
Table~\ref{tab:r4-calibration} separates two diagnostics from the FZ0 comparisons. With the same portfolio-GARCH filter, the joint conditional-moment (CM) diagnostic rejects GARCH-FHS in both panels ($p<0.001$), whereas GARCH-EVT is not rejected ($p=0.271$ and $0.394$). This contrast accompanies no detectable incremental FZ0 gain from EVT. It suggests a role for residual-tail modeling in the tested calibration moments even when mean scores are unresolved. These are unadjusted, finite-instrument tests with sparse tail observations; different rejection outcomes neither establish full conditional calibration nor constitute a paired test that EVT improves it.

At the unadjusted 5\% threshold, Kupiec coverage rejects Gaussian, Gaussian-LW, GARCH-t and WGAN-G in both panels; GARCH-t has 1.71\% and 1.43\% exceedances. In five baskets, meta-$t$ is also rejected for one seed. In 100 stocks, Flow-G, Flow-t, VAE-G, VAE-H and EWMAcov-FHS are additionally rejected; EWMAcov-FHS has only 14 exceedances (0.56\%), against 25.14 expected. CM additionally rejects WGAN-t in both panels; VAE-G and VAE-H for seed zero and Gaussian for seed one in five baskets (the latter $p=0.0496$ before rounding); and EWMAcov-FHS, Flow-meta-$t$ and VAE-G in 100 stocks. FHS and GARCH-EVT are not rejected by either diagnostic in either panel. These calibration checks complement density and mean-score comparisons without changing their multiplicity-adjusted conclusions.
'''
(ROOT / 'referee_round5/analysis/calibration_findings.tex').write_text(discussion)
replace('main.tex',
        r'\input{referee_round5/analysis/dependence_findings.tex}',
        r'\input{referee_round5/analysis/calibration_findings.tex}'+'\n\n'+r'\input{referee_round5/analysis/dependence_findings.tex}')
replace('referee_round5/analysis/summary_table.tex',
        'Same GARCH filter; neither resolved',
        'Score unresolved; CM: see caption')
replace('referee_round5/analysis/summary_table.tex',
        'Student p-values refer to the primary and broader retrospective families.',
        r'The unadjusted conditional-moment (CM) diagnostic rejects GARCH-FHS in both panels ($p<0.001$) but not GARCH-EVT ($p=0.271/0.394$; Table~\ref{tab:r4-calibration}). Student p-values refer to the primary and broader retrospective families.')
replace('referee_round4/analysis/conclusion.tex',
        'Replacing empirical residual tails by EVT adds no detectable benefit at the evaluated one-percent level.',
        r'Replacing empirical residual tails by EVT adds no detectable FZ0 benefit at one percent. Under the same GARCH filter, however, CM rejects GARCH-FHS in both panels and does not reject GARCH-EVT. This unadjusted diagnostic contrast suggests a calibration distinction; it does not establish full conditional calibration or a significant paired improvement.')
replace('expanded_crsp_methods.tex',
        'The empirical scenario law is enumerated, not resampled.',
        r'The 100-stock forecast has 14 exceedances (0.56\%), a Kupiec $p$-value of 0.015 and normalized ES moment $+0.156$ (Tables~\ref{tab:r4-crsp100-primary} and~\ref{tab:r4-calibration}); these outcomes do not isolate whitening noise as their cause. The empirical scenario law is enumerated, not resampled.')
for opening in ['The word ``Student\'\'', 'If exact marginal calibration']:
    replace('theory_main.tex',
            r'\noindent\emph{Proof.} See Appendix~\ref{app:proofs}.'+'\n'+opening,
            r'\noindent\emph{Proof.} See Appendix~\ref{app:proofs}.\par'+'\n\n'+opening)
replace('refs.bib',
        r'DOI: \href{https://doi.org/10.2139/ssrn.5996134}{10.2139/ssrn.5996134}. Public author copy',
        r'Original SSRN DOI: \href{https://doi.org/10.2139/ssrn.5996134}{10.2139/ssrn.5996134}. Public author copy')
print('Added calibration interpretation and separated proof-pointer remarks; numerical outputs unchanged.')

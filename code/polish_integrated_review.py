"""Apply the second integrated-paper review without changing numerical results.

Run after revise_integrated_paper.py and before compiling the manuscript.
Reapplying this script is safe.
"""
from pathlib import Path

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


for name in ['t_risk', 'metat_risk', 'heston_risk']:
    replace(f'referee_round4/analysis/{name}.tex',
            'Neural estimates use the repaired one-percent checkpoint rule;',
            r'Neural estimates use one-percent checkpoint selection on the amended candidate grid (Section~\ref{sec:checkpoint-repair});')
for panel in ['crsp5', 'crsp100']:
    for level in ['primary', 'secondary']:
        replace(f'referee_round4/analysis/{panel}_{level}.tex',
                r'\caption{Repaired ', r'\caption{')
    replace(f'referee_round4/analysis/{panel}_selection.tex',
            'Matched repaired checkpoint rules, seed zero.',
            r'Matched checkpoint rules on the amended candidate grid (Section~\ref{sec:checkpoint-repair}), seed zero.')
    replace(f'referee_round4/analysis/{panel}_selection.tex',
            'compares repaired with original one-percent selection.',
            'compares the amended with the original one-percent candidate grid.')

replace('round2_filter_methods.tex', 'the repaired primary comparison', 'the primary comparison')
replace('referee_round4/analysis/density_findings.tex', 'its repaired NLL', 'its likelihood-selected NLL')
replace('referee_round4/analysis/synthetic_narrative.tex', 'The complete repaired comparison', 'The complete comparison')
replace('referee_round4/analysis/verification.tex', 'The repair replays prefixes', 'The amended candidate-grid evaluation replays prefixes')
replace('referee_round4/analysis/verification.tex',
        'Every eligible the stated selection is at least as good on its own recorded validation criterion as the old admitted candidate set.',
        'For every eligible trajectory, the selected candidate performs at least as well on the recorded validation criterion as the best candidate in the earlier grid.')
replace('referee_round4/analysis/abstract.tex',
        'These results connect architectural tail restrictions to finite-sample evidence',
        'We test architectural tail restrictions against finite-sample evidence')
replace('main.tex', r'\input{round4_methods.tex}'+'\n\n'+r'\input{referee_round4/analysis/selection_findings.tex}',
        r'\input{round4_methods.tex}'+'\n\n'+r'\subsection{Results}\label{sec:control-results}'+'\n'+r'\input{referee_round4/analysis/selection_findings.tex}')

replace('referee_round5/analysis/summary_table.tex',
        'Student p-values refer to the primary and broader retrospective families.',
        r'Variance-conditioning magnitudes depend on the missing-return convention (Table~\ref{tab:missing100}). Student p-values refer to the primary and broader retrospective families.')
replace('main.tex',
        'The partial fixed-forecast stress and complete-date diagnostics cannot recover those outcomes.',
        r'The partial fixed-forecast stress and complete-date diagnostics cannot recover those outcomes. In 100 stocks, GARCH-FHS minus FHS is $-0.115$ at baseline, $-0.023$ under the 20\% missing-return stress and $-0.163$ on complete dates; its sign persists in these diagnostics, but its magnitude depends on the convention. Portfolio-EWMA-FHS moves from $+0.002$ to $+0.289$ under the same stress (Table~\ref{tab:missing100}).')

replace('referee_round5/analysis/dcc_findings.tex',
        'No improvement survives the eight-contrast adjustment against the two FHS filters.',
        r'The observed means favor constant-correlation counterparts in all three five-basket comparisons: EWMAcov-FHS versus FHS (1.379 versus 1.370), DCC-G versus CCC-G (1.435 versus 1.423), and DCC-t versus CCC-t (1.349 versus 1.330). This point-score pattern does not establish a significant or general disadvantage of dynamic correlation. No improvement survives the eight-contrast adjustment against the two FHS filters.')
replace('referee_round5/analysis/dcc_findings.tex',
        'The five-basket mean FZ0 scores are 1.423 for CCC-G, 1.330 for CCC-t, 1.435 for DCC-G, and 1.349 for DCC-t, against FHS 1.370. The observed means',
        'The observed means')
replace('round4_methods.tex',
        'These descriptive intervals do not estimate a limiting coefficient from one-percent data.',
        'The fitted model probabilities are also constant within each fold. Their additional value is pairwise calibration at a stated tail level, not within-year timing resolution; fitted marginal quantiles prevent a pure isolation of copula error. These descriptive intervals do not estimate a limiting coefficient from one-percent data.')
replace('referee_round5/analysis/dependence_findings.tex',
        'The new pairwise check gives a different perspective.',
        'The pairwise model probabilities are likewise fold-constant; their different perspective is a pair-specific event at an explicit tail level.')
replace('referee_round5/analysis/dependence_findings.tex',
        'All four pair-averaged difference intervals include zero;',
        r'At 1\% in five baskets, the observed pairwise statistic is 0.859 versus 0.531 under the model, with a difference interval $[-0.134,0.704]$. There are 183 coordinate exceedances on 79 dates, including 41 dates with at least two; the average is 36.6 per coordinate, against 25.14 expected under nominal marginal calibration. The 1.46\% marginal exceedance rate can contribute to the joint discrepancy, so it cannot be attributed solely to the copula. All four pair-averaged difference intervals include zero;')
replace('main.tex',
        'Pairwise diagnostics additionally separate asymptotic tail coefficients from finite-level joint exceedances;',
        'Pairwise model probabilities are also fold-constant; these diagnostics separate asymptotic tail coefficients from finite-level joint exceedances, without adding timing resolution;')
replace('expanded_crsp_methods.tex',
        'These diagnostics qualify the worse EWMAcov-FHS point scores;',
        'This effective sample size is a weighting diagnostic, not an algebraic rank bound. Shrunk or factor covariance filters, including factor-DCC, remain relevant 100-stock controls beyond the evaluated full EWMA covariance. These diagnostics qualify the worse EWMAcov-FHS point scores;')
replace('refs.bib',
        'note = {Preprint, manuscript dated December 31, 2025. Public author copy}',
        r'note = {Preprint, manuscript dated December 31, 2025. DOI: \href{https://doi.org/10.2139/ssrn.5996134}{10.2139/ssrn.5996134}. Public author copy}')
print('Applied the integrated-paper editorial review; numerical files unchanged.')

"""Apply the final publication edit without changing numerical results.

Run after write_completion_notes.py when reconstructing historical analyses.
The included LaTeX is ready to compile directly; this script is idempotent.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def replace(path, old, new):
    p = ROOT / path
    s = p.read_text()
    if old in s:
        p.write_text(s.replace(old, new))
    elif new not in s:
        raise ValueError(f"Publication-edit anchor missing: {path}: {old[:60]}")


def main():
    baseline = json.loads((ROOT / 'final_submission/editorial_numerical_baseline.json').read_text())
    for name, expected in baseline.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    status = json.loads((ROOT / 'completion_audit/analysis/numerical_decision.json').read_text())
    assert not status['conditional_ranking_eligible']

    (ROOT / 'referee_round4/analysis/abstract.tex').write_text(r'''We compare parametric, empirical and neural models of joint asset returns to determine whether greater flexibility improves density, dependence and portfolio risk. Gaussian, shrinkage-Gaussian, Student and meta-$t$ laws provide parametric benchmarks; meta-$t$ joins asset-specific Student margins with a Student copula. Filtered historical simulation (FHS) supplies empirical innovations. Normalizing flows learn invertible density transformations, variational autoencoders learn latent-variable distributions, and Wasserstein GANs learn scenario generators. Separate volatility and correlation controls distinguish conditioning from innovation modeling. Ten estimation datasets per Student, meta-$t$ and Heston synthetic law, and 2,514 CRSP test dates in five-basket and 100-stock panels, support paired comparisons with multiplicity control. Meta-$t$ achieves the best observed exact density fit in both CRSP panels. Under common one-percent validation selection, no tested neural model establishes a risk-score improvement over FHS; alternative checkpoint rules change rankings. Student's 100-stock advantage depends on the comparison family. Variance filtering produces the largest point-score gains, with inconclusive adjusted tests. Synthetic results distinguish correct specification from architectural flexibility and separate likelihood quality from expected-shortfall accuracy. A Wishart--CIR extension relaxes a restricted covariance model's dispersion ceiling but fails the conditional numerical-stability gate. The evidence makes conditioning, specification and validation design central to assessing distributional flexibility.
''')

    replace('main.tex',
        'The comparison separates three objectives: fitting the joint density, reproducing downside dependence, and forecasting portfolio Value-at-Risk (VaR) and expected shortfall (ES). A model can succeed at one and fail at another.',
        'The comparison separates three objectives: fitting the joint density, reproducing downside dependence, and forecasting portfolio Value-at-Risk (VaR) and expected shortfall (ES). Density measures the whole distribution, dependence measures joint downside events, and risk forecasts measure losses on fixed portfolio exposures. These objectives can favor different fitted models.')
    replace('main.tex',
        'The same manuscript contains the theoretical statements, empirical tests and supporting proofs. Longer derivations, secondary comparisons and numerical audits are in its appendices.',
        'The controlled laws isolate specification and representation; the chronological CRSP panels evaluate conditional forecasting. Matched filter controls attribute changes in point scores, checkpoint alternatives assess selection sensitivity, and paired inference quantifies the resolution of each comparison. Longer derivations, secondary comparisons and numerical audits are in the appendices.')
    replace('expanded_methods.tex', 'Both new laws have analytic portfolio VaR and ES.',
        'Student and shrinkage-Gaussian laws have analytic portfolio VaR and ES.')
    replace('expanded_methods.tex',
        'Earlier fits used post-estimation clipping, but the archive audit found no active clipping; the present fits explicitly impose the bounds during optimization.',
        'The bounds are imposed during optimization; no post-estimation clipping defines the reported estimator.')
    replace('expanded_methods.tex',
        'Estimation and validation seeds are disjoint from the original three-dataset study. The current synthetic comparison has',
        'The synthetic comparison has')
    replace('expanded_methods.tex', 'including the added Student and shrinkage-Gaussian laws',
        'including Student and shrinkage-Gaussian laws')
    replace('expanded_methods.tex', 'The completed Heston matrix includes', 'The Heston comparison includes')
    replace('expanded_methods.tex', 'The refined Wishart evaluation instead integrates',
        'The Wishart evaluation instead integrates')
    replace('expanded_methods.tex', 'The added Student and shrinkage-Gaussian benchmarks use',
        'The Student and shrinkage-Gaussian benchmarks use')

    replace('expanded_audit.tex', r'\paragraph{Final inference and data protocol.}',
        r'\paragraph{Inference and data protocol.}')
    replace('expanded_audit.tex',
        'An earlier interval/test mismatch was corrected before the current comparisons. The 100-stock universe requires 252 finite pre-estimation observations; 41 cold-start pilot evaluations were excluded and every retained model was refitted under the amended rule. The incomplete preliminary 2024 export was rebuilt before any retained 2024 evaluation. The source audit verifies every annual panel and missingness mask across all 2,514 test dates. Earlier calculations remain separately archived.',
        'The 100-stock universe requires 252 finite pre-estimation observations. Every retained fit uses this eligibility rule and the complete annual inputs. The source audit verifies the annual panels and missingness masks across all 2,514 test dates. Earlier pilot evaluations, data amendments and inference corrections are documented in the research archive; they do not enter the reported comparisons.')

    replace('wishart_cir_main.tex',
        r'\section{Extending the stochastic-covariance benchmark}',
        r'\section{Stochastic covariance: specification and numerical evidence}')
    replace('wishart_cir_main.tex',
        'This separately reported extension does not change the original model-selection rules or multiplicity families.',
        'This retrospective specification experiment uses the same Heston estimation datasets. Its synthetic results and numerical diagnostics are reported separately from the primary conditional comparisons and their multiplicity families.')
    replace('wishart_cir_extension/analysis/filter_findings.tex',
        'Wishart--CIR is therefore numerically ineligible for the conditional CRSP ranking in this release. Its completed synthetic experiment diagnoses specification; it supplies no full-panel score or portfolio-optimization recommendation and does not enter the primary Holm or MCS families.',
        'The numerical gate excludes Wishart--CIR from the conditional CRSP comparison and from its Holm and MCS families. Its evidence consists of the stated mathematical results, synthetic specification experiment and numerical failure diagnostics. A full-panel forecast score or an allocation claim would exceed that evidence.')

    (ROOT / 'referee_round4/analysis/conclusion.tex').write_text(r'''The comparison yields different answers for density estimation and portfolio forecasting. Meta-$t$ has the best observed exact density fit in both CRSP panels. In 100 stocks, likelihood validation selects the parametric identity in all ten residual-flow folds, so that result supplies no evidence of improvement from the learned residual. Under the common one-percent checkpoint rule, none of the tested neural innovation models establishes an adjusted FZ0 improvement over FHS. Student improves in the primary 100-stock family (Holm $p=0.0352$), while its broader-family result ($p=0.0594$) shows the sensitivity to the inferential scope. Checkpoint alternatives materially change rankings.

Changing the portfolio variance recursion accounts for the largest observed score gains: approximately $-0.055$ in five baskets and $-0.115$ in 100 stocks. Their adjusted tests remain inconclusive. EVT supplies no detectable incremental FZ0 gain at one percent; the differing conditional-moment outcomes for GARCH-FHS and GARCH-EVT concern the tested calibration restrictions and do not establish a significant paired improvement. Thus conditioning is the leading point-score explanation in this design, with finite-sample uncertainty applying equally to statistical and neural models.

The synthetic experiments show why specification and the evaluation level matter. Parametric models benefit when their family contains the target. Under Heston misspecification, a better fitted density can accompany substantially worse ES, and favorable one-percent errors can deteriorate at 0.1 percent. The Wishart--CIR construction removes the restricted model's marginal-dispersion ceiling, yet matching kurtosis leaves joint-risk errors and leverage sensitivity. Its frozen-law equivalence also prevents static density fit from identifying a benefit of stochastic correlation. Both evaluated Wishart filters fail their numerical gates and are excluded from conditional ranking.

These findings apply to the stated architectures, samples, portfolios and retrospective comparison design. They support a practical order of assessment: establish numerically stable forecasts, compare predictable conditioning, then test whether a more flexible joint law improves the specified density, dependence or risk objective under a fixed validation procedure. A model's likelihood ranking alone cannot answer the portfolio-risk question, and fixed-exposure forecasts do not validate optimized portfolio weights.
''')

    replace('README.md',
        'python code/write_completion_notes.py\nlatexmk',
        'python code/write_completion_notes.py\npython code/finish_manuscript.py\nlatexmk')
    replace('COMPLETION_REVIEW.md',
        'Closing the audit records those limits; it does not resolve them by wording or warrant an unconditional “A+” scientific claim.',
        'The numerical exclusion is a final reported outcome for the evaluated implementation. The paper retains the underlying evidence and makes no claim of convergence, universal model superiority or validated portfolio optimization.')
    marker = '## Publication edit\n'
    p = ROOT / 'COMPLETION_REVIEW.md'
    s = p.read_text()
    if marker not in s:
        s += '\n' + marker + '\nThe final edit explains every model family in the abstract, brings the meta-t density result into the abstract and conclusion, distinguishes the experiment roles in the introduction, removes historical-workflow phrasing from methods, and gives the stochastic-covariance study a definitive evidence-based disposition. It changes no numerical outputs, model selections, tests or admissibility criteria. Reproduction applies `code/finish_manuscript.py` after the existing analysis and editorial sequence.\n'
        p.write_text(s)
    (ROOT / 'FINALIZATION.md').write_text('''# Manuscript status

The integrated manuscript is complete for the evaluated research design. It contains the theory, empirical comparisons, synthetic stochastic-covariance study and numerical audit in one document. The failed Wishart conditional-numerical checks are reported results; those implementations are excluded from conditional rankings. This does not imply that their numerical limitation has been repaired.

The abstract and conclusion distinguish observed density rankings, adjusted risk-score comparisons, conditioning effects and selection sensitivity. Optimized portfolio weights are outside the evaluated scope.

See `COMPLETION_REVIEW.md` for evidence and limitations, `README.md` for the current build sequence, `PUBLICATION.md` for the publication destinations, and `final_submission/final_checks.json` for the document checks. The DOI remains reserved and the Zenodo record remains an unpublished draft.
''')
    for name, expected in baseline.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print(f'Publication edit complete; {len(baseline)} numerical hashes unchanged; '
          f'abstract: {len((ROOT / "referee_round4/analysis/abstract.tex").read_text().split())} words.')


if __name__ == '__main__':
    main()

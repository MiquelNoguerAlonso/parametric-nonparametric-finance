"""Write a current, unambiguous release record from verified outputs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    s=json.loads((ROOT/'completion_audit/analysis/numerical_decision.json').read_text());f=s['independent_banks'][-1]
    assert not s['conditional_ranking_eligible']
    review=f'''# Completion review — 7 November 2026

One integrated manuscript is retained. This release completes the editorial and numerical-eligibility audit; it does not claim an independently validated portfolio-optimization strategy, an untouched confirmation sample, or a converged Wishart CRSP competitor.

## Changes in this pass

- Rewrote the abstract around the research question, model classes, common filters, data and conditional inference. The neural conclusion remains conditional on one-percent validation selection. Student's primary-versus-broader-family distinction and the unresolved GARCH effects remain explicit.
- Added and proved frozen-law observational equivalence: the stationary driftless Wishart–CIR snapshot equals a scalar normal variance mixture with a sum of independent gamma scales. Static density or marginal-tail fit cannot establish the value of dynamic correlation. This is a consequence of the specified model, not a new distribution-ownership claim.
- Added exact-ratio independent Metropolis refreshes of ancestor and complete daily path. Four complete 2025 runs use two independent banks at each of 8,192 and 16,384 particles, 16 substeps, 32,768 predictive paths and 504 warmup days. No fitted parameter or acceptance tolerance changed.
- The final 16,384-particle banks have relative-ES RMS {100*f['es_relative_rms']:.4f}% and annual FZ0 difference {f['score_difference']:+.6f}. Numerical eligibility remains false. All thirty completed Wishart–CIR filter runs are retained. The synthetic specification study is complete; no full-panel conditional row is invented.
- Added an explicit allocation-scope paragraph: the fixed-portfolio evaluation does not validate optimized weights. The identity selection in all ten 100-stock residual-flow folds is stated when discussing model candidates.
- Corrected the reproduction instructions to name all six archives and compile the current integrated source directly. Historical processors are clearly separated from the current build.

## Independent checks repeated or added

The repeated primary audit reconstructs 945,236 forecast rows across 1,054 files, checks 640 retained neural trajectories, 160 joint parametric fits, common dates, Holm arithmetic, analytic risk and the exact error decomposition. Maximum FZ0 reconstruction error is 1.14e-12. The 47-file pre-extension numerical baseline is unchanged.

New finite-state detailed-balance checks verify the independent-Metropolis acceptance ratio. Independent stationary-state simulation checks the frozen characteristic function on an equal-weight portfolio, a spread and a single asset. Existing Wishart–CIR transition, conditional-moment, covariance and Gaussian-tail checks pass again. These checks test specified contracts; they are not a proof of all code or of filter convergence.

## Disposition of the substantive referee concerns

| Concern | Final treatment |
|---|---|
| Symmetric inference and power | Neural and statistical contrasts use the declared families; GARCH effect sizes and adjusted uncertainty are reported symmetrically. |
| Sparse checkpoint validation | Initialization and nested candidates, 2.5%/5%, pooled, final and density rules; retrospective status and sparse-tail qualification remain explicit. |
| Restricted Wishart dispersion | Frozen ceiling, inadmissible matching shape and fitted versus target persistence are stated; additive CIR repair is evaluated. |
| Wishart numerical uncertainty | Both specifications are excluded from primary conditional ranking; failed diagnostics and exact-ratio refresh controls are retained. |
| Heston density–risk discrepancy | Central-peak explanation is qualified; fitted Student/meta-t fourth moments and analytic ES errors are distinguished. |
| Flow dependence restriction | Identity Flow-t scatter is specified, dense LU controls and nested laws are evaluated, likelihood-selected NLL is in the main discussion. |
| Filter confounding | Portfolio decomposition, coordinate-GARCH, fitted-EWMA and five-basket CCC/DCC controls remain. |
| MCS interpretation | Current all-pairs results and candidate-set/precision dependence replace earlier ranking interpretations. |
| Meta-t attribution | Established construction is credited; the author's application is cited without a distribution-invention claim. |
| Tail-level transfer | Flow-t and independent-noise WGAN discrepancies at 0.1% are explicit. |
| Bookkeeping and artifacts | Fit records, replayed trajectories and numerical runs are separate; current build and archive requirements are explicit. |

## Remaining scientific limits

The study is retrospective, evaluates fixed exposures and bounded static neural architectures, has sparse one-percent validation tails and one 100-stock optimization seed, and uses an explicit missing-return convention. No correctly specified dynamic scalar-Heston benchmark is fitted. The Wishart–CIR conditional filter fails its tolerance. Closing the audit records those limits; it does not resolve them by wording or warrant an unconditional “A+” scientific claim.
'''
    (ROOT/'COMPLETION_REVIEW.md').write_text(review)
    (ROOT/'README.md').write_text('''# Parametric and Nonparametric Models of Joint Returns: Density, Dependence, and Portfolio Tail Risk

Miquel Noguer i Alonso — Artificial Intelligence Finance Institute

Reserved DOI: 10.5281/zenodo.24024955. Current integrated revision: 7 November 2026. See `COMPLETION_REVIEW.md` and `final_submission/final_checks.json`. One manuscript contains the theory, experiments and appendices. Zenodo contains one manuscript PDF in an unpublished draft.

## Compile this manuscript

The compact Overleaf ZIP contains five files: `main.tex`, `refs.bib`, `README.md` and two figure PDFs. Upload all five to one project and compile `main.tex`. It requires no datasets or Python. The date is `\\today`.

In the complete source tree, compile the included current source directly:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Do not run historical editorial processors alone: they can restore an earlier manuscript. The current PDF and included source are authoritative.

## Research archive structure

For full numerical reconstruction, extract these six archives into one directory, preserving relative paths:

1. `Generative_Models_Finance_LaTeX.zip`: current manuscript, code, derived panels, records and analyses.
2. `Generative_Models_Finance_Reference_Samples.zip`: large reference-loss samples.
3. `Generative_Models_Finance_Round3_Checkpoints.zip`: dense-flow and coordinate-filter states.
4. `Generative_Models_Finance_Round4_Results_Checkpoints.zip`: amended checkpoints, forecasts and pointwise densities.
5. `Generative_Models_Finance_Wishart_Numerical_Audit.zip`: restricted-Wishart paths, filter audits and restart states.
6. `Wishart_CIR_Scenario_Archive.zip`: additive Wishart–CIR conditional-mixture arrays.

The separate archives are parts of one research package, not separate papers. Raw licensed CRSP extracts are not redistributed. Derived panels and masks support the recorded output audit; source-construction scripts require the authorized extracts.

## Verify current outputs without resetting the paper

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/verify_current_release.py --pdf
```

This checks the frozen numerical baseline, primary forecasts and inference, new frozen-law identity and completed resample-move audit. It compiles the current manuscript. It does not rerun model training or prove filter convergence.

The main comparison has 1,150 model/fold records, including forty CCC/DCC additions and ten excluded restricted-Wishart CRSP pilots. The 640 neural prefix replays reuse retained trajectories; 2,100 scalar coordinate-filter fits are separate submodels. The additive extension contributes twenty parameter fits and thirty numerical filter runs, not thirty new fitted models.

## Reconstruct analyses and the current editorial state

With all archived outputs present, the complete editorial sequence is:

```bash
python code/revise_integrated_paper.py
python code/polish_integrated_review.py
python code/add_calibration_discussion.py
python code/analyze_wishart_cir.py
python code/write_wishart_cir_report.py
python code/integrate_wishart_cir.py
python code/analyze_completion.py
python code/finalize_completion.py
python code/write_completion_notes.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The older `code/run_all.py --mode outputs` rebuilds the preceding core experimental release; apply the full sequence above afterward. `--mode experiments --jobs 3` reruns missing core records, reusing completed records. For an independent clean rerun, use an isolated copy and move its output/checkpoint directories aside while preserving protocols and derived inputs. `--raw-crsp /absolute/path/to/authorized/extracts` rebuilds from licensed sources.

## Wishart–CIR experiment and numerical eligibility

`code/wishart_cir.py` defines the model and exact state transitions. `run_wishart_cir.py` fits parameters and produces synthetic and initial conditional audits; `wishart_cir_move.py` and `run_wishart_cir_move.py` supply the invariant refresh control. Protocols were recorded before the associated evaluations, but the extension is retrospective relative to the main study. The model fits and acceptance tolerances remain fixed during numerical refinement.

`wishart_cir_extension/analysis/final_status.json` describes the initial 26-run audit. The authoritative subsequent numerical decision is `completion_audit/analysis/numerical_decision.json`, which includes four additional complete runs. Wishart–CIR remains ineligible for conditional ranking. No full 2,514-date forecast, optimized portfolio or production-ready state filter is claimed. Its completed synthetic experiment and mathematical specification results remain in the integrated paper.

## Interpretation and traceability

No primary neural FHS contrast establishes improvement after Holm control under the stated one-percent validation rule. Student improves in the 100-stock primary family but not in the broader retrospective family. GARCH variance conditioning has the largest observed gains, with inconclusive adjusted tests. Density, calibration and fixed-portfolio risk are separate objectives. The study does not validate optimized portfolio weights.

Machine identifiers `Historical` and `VAE-t` are retained for traceability; paper labels use FHS in CRSP, empirical ES in synthetic studies, and VAE-tlatent for the bounded-decoder model. Historical response files describe earlier releases and are not the current numerical decision.
''')
    (ROOT/'PUBLICATION.md').write_text('''# Current publication record

Title: Parametric and Nonparametric Models of Joint Returns: Density, Dependence, and Portfolio Tail Risk

Author: Miquel Noguer i Alonso. Affiliation: Artificial Intelligence Finance Institute.

Reserved DOI: 10.5281/zenodo.24024955
Overleaf project: https://www.overleaf.com/project/6aee6e8f84dcc0b2d4647dab
Zenodo draft: https://zenodo.org/uploads/24024955

One integrated manuscript, including all appendices. One manuscript PDF in the unpublished Zenodo draft. The DOI is reserved; registration requires publication. Publication has not been performed. The paper uses `\\today`.

Current scope and scientific limitations are recorded in `COMPLETION_REVIEW.md`. Current build checks are in `final_submission/final_checks.json`. Primary numerical outputs remain unchanged; the new resample-move results are separately archived. The full research package has six parts, listed in `README.md`; the compact Overleaf package is independently sufficient for manuscript compilation.
''')
    (ROOT/'FINALIZATION.md').write_text('''# Finalization status

See `COMPLETION_REVIEW.md` for this revision's changes, evidence and limits; `README.md` for the current build instructions; `PUBLICATION.md` for the publication destinations; and `final_submission/final_checks.json` for numerical and document checks.

The manuscript is a completed report of the evaluated design. The Wishart–CIR numerical gate remains unmet, and that model is excluded from conditional ranking. This editorial status is not a declaration of numerical convergence or of universal model superiority. There is no independently validated portfolio-optimization strategy in this study.
''')

if __name__=='__main__':main()

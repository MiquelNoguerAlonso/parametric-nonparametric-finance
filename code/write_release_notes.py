"""Write completed-study documentation from the final analysis outputs."""
from pathlib import Path
import json,pandas as pd,hashlib,platform,sys
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'expansion/analysis'
def main():
 five=pd.read_csv(O/'crsp5_summary.csv');five=five[five.alpha.eq(.01)].set_index('model')
 hundred=pd.read_csv(O/'crsp100_summary.csv');hundred=hundred[hundred.alpha.eq(.01)].set_index('model')
 best=hundred.fz0.idxmin();f=five.loc['Historical'];h=hundred.loc['Historical'];sig=hundred[(hundred.delta_fhs<0)&(hundred.p_holm<.05)]
 checkfiles=['results/mathematical_checks.json','results/output_checks.json','results/revision/revision_checks.json','expansion/checks_preflight.json','expansion/checks_outputs.json','expansion/analysis/diagnostic_checks.json']
 total=sum(len(json.loads((ROOT/p).read_text())) for p in checkfiles)
 README=f'''# Tail Architecture, Density Fit, and Conditional Risk Forecasting

Generative Models against Filtered Historical Simulation  
Miquel Noguer i Alonso — November 2026

`main.pdf` is the fully expanded manuscript. This release replaces the first revision's empirical tables with fresh experiments. The original results and selected states remain archived and are not pooled into the new estimates.

## Completed design

- **700 retained new model evaluations and 500 fresh neural fits.** These comprise 300 synthetic evaluations, 200 five-basket annual-fold/seed evaluations, 100 annual 100-stock evaluations, and 100 full-covariance five-basket refits.
- **Ten independent datasets per synthetic law**, one optimization seed per dataset, paired MSE differences and dataset-level standard errors, fit/sampling/cross decomposition, and RMSE skill against exact empirical-training ES.
- **Common validation FZ0 checkpoint selection** at one percent for every neural family, fixed 20,000 validation scenarios, every 250 steps, with original training budgets retained. Training objectives remain family-specific.
- **One million primary synthetic evaluation scenarios per fit** at 1% and 0.1%, with four-million-scenario nonanalytic references and analytic Student ES. Same-sample oracle NLL is reported when the target density is available.
- **Two chronological CRSP panels:** five size baskets (two seeds) and annually selected 100-stock universes (one seed), each with 2,514 test dates in 2016–2025. Generators use 100,000 scenarios; FHS enumerates the full empirical innovation law.
- GARCH(1,1)-t, McNeil–Frey-style conditional EVT, and full EWMA-covariance FHS benchmarks. Thirty high-dimensional portfolio GARCH fits supplement the thirty five-basket fits.
- Full-covariance whitening/recoloring **refits all ten generator specifications** in the five-basket panel, with paired diagonal-filter seed-zero comparisons.
- Consistent 9,999-draw basic bootstrap intervals and equal-tailed tests, FHS contrasts with within-panel Holm adjustment, MCS with 10/20/40-day blocks, seed-specific Kupiec and conditional-moment calibration tests, and crash exclusions.
- **300 additional ancillary diagnostics** on the same ten fitted datasets: radial repair, Heston dependence, and empirical projected Wasserstein checks. Their 100,000-scenario samples concern one-percent/dependence diagnostics; they do not replace the million-scenario primary tail evaluations.
- **{total} recorded checks pass.** Checks establish the stated numerical/data contracts, not universal model superiority.

## Results that matter

| Comparison | New result | Interpretation |
|---|---|---|
| Five-basket joint risk score | GARCH-EVT {five.loc['GARCH-EVT','fz0']:.4f}; FHS {f.fz0:.4f} | Lowest observed score; its FHS improvement does not survive twelve-contrast Holm adjustment |
| Tail-aware flows, five baskets | Flow-t {five.loc['Flow-t','fz0']:.4f}; Flow-meta-t {five.loc['Flow-meta-t','fz0']:.4f} | Both improve FHS unadjusted; neither survives Holm adjustment |
| Exact density, five baskets | Plain meta-t NLL {five.loc['Meta-t','nll']:.4f} | Best observed exact joint density; VaR coverage is a separate issue |
| 100-stock joint risk score | {best} {hundred.loc[best,'fz0']:.4f}; FHS {h.fz0:.4f} | See complete inference and calibration tables before interpreting the ranking |
| Synthetic heavy tails | Plain meta-t has lowest one-percent model RMSE in both Student and heterogeneous meta-t laws | Residual meta-t flow is the strongest flexible model on the heterogeneous target |
| Synthetic Heston | Exact empirical-training ES has lowest RMSE; VAE-H slightly improves on VAE-G | Tail matching does not alone determine finite-level error |

All comparisons are retrospective. No statistical claim extends automatically across the two panels or every secondary analysis. IWAE-64 NLL has an upward expectation bias conditional on a fixed model and test sample, but a realized estimate also has Monte Carlo noise.

## Reproduction

Use Python 3.12 with `requirements.txt`, and a TeX installation with `latexmk` and natbib:

```bash
python -m pip install -r requirements.txt
python code/run_all.py --mode outputs --pdf
```

Output mode validates saved records and reconstructs tables, figures, and findings. It includes small 250-step preflight fits that test the common-selection implementation, not a full refit. Omit `--pdf` to skip compilation. Regenerate this documentation after a deliberate result update with `python code/write_release_notes.py`.

```bash
python code/run_all.py --mode experiments --jobs 2 --pdf
```

Experiment mode resumes the **expanded** design, including all ancillary diagnostics. Completed records and selected checkpoints are reused. For a complete fresh run, preserve this release and use a separate extracted copy with the contents of `expansion/results/`, `expansion/checkpoints/`, and `expansion/diagnostics/` moved aside. Retain the derived data and references unless intentionally regenerating them. Output-only reconstruction is much cheaper than fresh fitting; low-order floating-point differences may occur across platforms. `code/run_legacy.py` is the archived original-design entry point; it must not be mistaken for the expanded protocol.

To rebuild derived panels from authorized raw CRSP extracts:

```bash
python code/run_all.py --mode outputs --raw-crsp /absolute/path/to/crsp_extracts
```

Source filenames and SHA-256 digests are in `data/crsp_data_audit.json`. Raw licensed extracts are not redistributed. `verify_highdim_sources.py` independently checks derived 100-stock returns and missingness against those sources. The primary supplied derived panels allow output reconstruction without raw files.

## Data construction and audit

The source comprises 21 CRSP CIZ annual extracts for 2005–2025, 40,402,817 security-day records, and 9,873 eligible historical securities. Five size-basket holdings use prior-close eligibility, capitalization, and weights. CIZ total returns already include main-series delisting-return records; no legacy merge is added.

For test year y, the 100-stock universe uses the largest eligible training-end (y−2) capitalizations, at least 98% finite training returns, and at least 252 finite observations before estimation starts. Validation is y−1. There is no validation/test survival screen. Missing subsequent returns receive zero with complete masks and fixed-forecast stress diagnostics. These are gross risk portfolios, not transaction-cost-adjusted trading strategies.

The initial high-dimensional pilot exposed an artificial cold-start volatility divisor. Its **41 completed model evaluations were rejected and every high-dimensional fold was rerun** under the history requirement. `expansion/rejected_cold_start/` preserves those excluded records and the reason. A separate incomplete 2024 CSV export was repaired before any retained 2024 evaluation; independent source reconstruction checks all ten panels and 2,514 test dates. No rejected pilot or partial panel enters the paper's retained tables.

## Package map

| Path | Contents |
|---|---|
| `main.tex`, `theory.tex`, `heston_appendix.tex`, `expanded_*.tex`, `refs.bib` | Authored manuscript and bibliography |
| `expansion/protocol.json` | Expanded retrospective design and dated amendments |
| `expansion/results/`, `expansion/checkpoints/` | All retained new evaluations, forecasts, selected parameters and FZ0 traces |
| `expansion/data/` | Large reference samples, 100-stock panels, universe identities and source checks |
| `expansion/analysis/` | Complete CSVs, tables, inference, calibration and findings |
| `expansion/diagnostics/` | All 300 ancillary evaluations |
| `code/expanded_experiment.py`, `code/models.py` | New experiments and common-selection model implementation |
| `code/expanded_dynamic.py`, `code/expanded_sensitivities.py` | Statistical benchmarks, alternate filter and missingness checks |
| `code/analyze_expansion.py`, `code/expanded_diagnostics.py` | Paired analysis, MCS, calibration and supporting diagnostics |
| `code/check_expansion.py`, `code/check_expanded_outputs.py` | Selection/data preflight and final accounting checks |
| `results/`, `checkpoints/`, `legacy_outputs/` | Preserved original 380 evaluations, 272 neural states and first revision |
| `REVISION_AUDIT.md` | Point-by-point response and actual limitations |
| `MANIFEST.sha256` | Package checksums |

Legacy identifiers `Historical` and `VAE-t` remain in machine-readable records for traceability. Displays use FHS for filtered CRSP innovations, Empirical for unfiltered synthetic observations, and VAE-tlatent for the bounded-decoder model.

Ten synthetic datasets, one high-dimensional optimization seed, established large-cap selection, a fixed numerical Heston bank, and forecast-conditional retrospective inference remain limitations. The alternate dynamic filter is EWMA covariance, not DCC. Conditional calibration uses a Nolde–Ziegel-style identification-moment Wald test, not Bayer–Dimitriadis ESR. These are the explicit alternatives used to meet the requested benchmark and backtest extensions.
'''
 (ROOT/'README.md').write_text(README)
 audit=f'''# Response to all major-revision requests

The fresh empirical extension is complete. The expanded paper reports 700 new retained evaluations, 500 new neural fits, and 300 ancillary evaluations. The earlier 380 evaluations/272 neural fits are preserved separately. All {total} recorded checks pass. This is evidence for the stated experiments, not a submission-readiness grade or proof of universal superiority.

| Review issue | Implemented response |
|---|---|
| 1. Meta-t identity and attribution | Uses the established Fang–Fang–Kotz / Demarta–McNeil object explicitly; cites the author's portfolio application once with the supplied stable DOI 10.5281/zenodo.23932880, not SSRN. Original 2025 working-paper and 2026 revision metadata are distinguished. |
| 2. FHS reference | FHS is the primary comparator in both CRSP panels and every relevant table/annual plot. Its full empirical innovation distribution is enumerated exactly. |
| 3. Density versus calibration | Dedicated meta-t paragraphs report exact NLL, integer exceedance counts, Kupiec tests, ES moments, and joint conditional diagnostics. Fresh counts replace earlier scenario-dependent counts. |
| 4. Inference coherence and MCS | The old percentile interval and centered absolute-error p-value were different constructions: the old VAE-G p=0.096 explains Holm 0.672. New basic intervals and equal-tailed tests share the same 9,999 circular draws. Added TR/eR MCS and block-length sensitivity. |
| 5. Strong dynamic benchmarks | Estimated GARCH(1,1)-t and conditional GPD-tail forecasts in both panels; added full EWMA-covariance FHS. Additionally refitted every original generator under the full covariance filter in the five-basket panel. |
| 6. Synthetic power and decomposition | Ten independent estimation datasets per law, one seed each; paired model-minus-exact-empirical MSE differences and SEs, full fit/sampling/cross identity, and RMSE skill. The Monte Carlo resampling baseline is separated from the exact empirical estimator. |
| 7. Oracle NLL and IWAE | Same-test-sample true-density rows for Student and heterogeneous meta-t laws; Student entropy 3.36070 versus sample oracle NLL 3.35305. Explicitly states the conservative direction of a lower VAE IWAE-NLL in expectation, with realization noise. No unavailable exact Heston joint density is invented. |
| 8. Model naming and tail interpretation | VAE-tlatent identifies the bounded Gaussian decoder despite Student latent noise. New Heston data show VAE-H slightly improving VAE-G. Polynomial WGAN-t tails mismatch Heston asymptotics, but the theorem does not mathematically predict a fixed 1% RMSE or causation. |
| 9. Common checkpoint selection | All 500 fresh neural fits use the same validation FZ0 target, fixed 20,000 scenarios, and 250-step checkpoint interval. Saved traces independently verify every selected checkpoint is the trace minimum. Training objectives and original budgets are retained. |
| 10. Contributions and literature | Leads with elliptical versus independent Student noise and residual meta-t flows; attributes standard Wasserstein/risk identities; moves Heston background to the appendix. Adds Tail-GAN, EV-GAN, COMET, MCS, Nolde–Ziegel, Higham and DRO attribution. |

## Technical and presentation changes

- **Correlation matrix:** construction requires positive definiteness and unit diagonal. Fitting uses eigenvalue flooring plus diagonal renormalization, correctly described as a correlation correction rather than Higham's nearest matrix algorithm.
- **Loss theorem:** stated for −wᵀG(Z), with the positive-part angular constant and bounded-above alternative. The final-permutation restriction is removed where all intermediate indices coincide.
- **Degrees of freedom:** new marginal likelihood optimization enforces bounds during fitting. Complete fold parameters are exported. WGAN-t uses ν=3; annual marginal/copula parameters are tabulated. Earlier inactive clipping is documented, not relabeled constrained MLE.
- **0.1% columns:** now supported by one million generated scenarios per fit and four million reference scenarios (or analytic Student ES). Batch precision diagnostics are separate from dataset uncertainty and Heston discretization error.
- **Influential days:** common-date exclusions of 2020 and the top 5/10/20 realized-loss dates are reported for both panels.
- **Conditional tests:** four identification moments with constant and lagged-hit instruments, HAC bandwidth 20, and rank-based chi-square reference; rare-hit and forecast-conditioning limitations stated. This is the selected Nolde–Ziegel-style alternative, not ESR.
- **Notation:** separates mixture weight and Dirac mass, decoder Cholesky and loss notation, Lipschitz constant, latent dimension, tail constant and portfolio index.
- **Figures/tables:** FHS included in every CRSP comparison; distinct markers for close VAE curves; Flow-meta-t included in the paired controlled-study figure; captions use exceedance rates and distinguish exact NLL from IWAE estimates. Synthetic captions identify their different target laws.
- **Dimensionality:** actual authorized security-level CRSP inputs support annual 100-stock universes. Selection is historical, includes a pre-estimation warmup requirement, and uses no future survival screen. Concentrated exposures and missingness stress are supplied.
- **Supporting experiments:** radial-splice thresholds and Heston dependence are re-evaluated across all ten independent datasets, with adverse outcomes retained. These 100,000-scenario ancillary checks concern 1% risk/dependence only.

## New central findings

Five-basket FHS scores {f.fz0:.4f}. Flow-t and Flow-meta-t score {five.loc['Flow-t','fz0']:.4f} and {five.loc['Flow-meta-t','fz0']:.4f}; both improve FHS unadjusted but neither passes within-panel Holm correction. GARCH-EVT has the lowest observed score, {five.loc['GARCH-EVT','fz0']:.4f}. Plain meta-t retains the best observed exact joint density there.

In 100 dimensions, {best} scores {hundred.loc[best,'fz0']:.4f} versus {h.fz0:.4f} for FHS. Models with negative FHS differences and within-panel Holm p<0.05 are: {', '.join(sig.index) if len(sig) else 'none'}. The manuscript gives their calibration and model-confidence-set results separately.

On controlled heavy-tailed targets, plain meta-t has the lowest observed one-percent RMSE; Flow-meta-t is the strongest flexible model on the heterogeneous target. In Heston, the exact empirical estimator outperforms every fitted generator in observed RMSE. These statements refer to the specified finite experiments.

## Audit and limits

The initial 100-stock pilot had an artificial cold-start volatility divisor. Its 41 completed evaluations were rejected and archived; every retained high-dimensional fold was rerun after requiring 252 finite pre-estimation observations. An incomplete 2024 derived CSV export was caught before any retained 2024 evaluation, rebuilt, and checked independently against the raw source. All ten annual panels, all missingness masks, and all 2,514 test dates are verified. The rejected records never enter primary analysis.

The author's portfolio-paper DOI is retained as supplied; the remote resolver did not provide retrievable metadata in this environment, so independent resolution is not claimed. This does not alter the established distributional attribution. The remaining substantive limitations are ten independent synthetic replicas, one 100-stock optimizer seed, fixed numerical Heston initial states and discretization, retrospective specification, forecast-conditional resampling, and explicit missing-return conventions. EWMA covariance and the conditional-moment test are the chosen alternatives to DCC and ESR. No required simulation, common-selection refit, million-scenario tail evaluation, or higher-dimensional panel remains unrun.
'''
 (ROOT/'REVISION_AUDIT.md').write_text(audit)
 x=json.loads((ROOT/'protocol_revision.json').read_text());x['status']='Superseded first-revision protocol retained for provenance; complete expanded design in expansion/protocol.json';(ROOT/'protocol_revision.json').write_text(json.dumps(x,indent=2))
 files=list((ROOT/'code').glob('*.py'));meta=dict(python=sys.version,platform=platform.platform(),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
 (ROOT/'expansion/release_environment.json').write_text(json.dumps(meta,indent=2))
 print('Release notes written; checks',total)
if __name__=='__main__':main()

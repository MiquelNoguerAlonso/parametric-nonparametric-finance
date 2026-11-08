"""Generate release documentation and a point-by-point response from saved results."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round2/analysis'
def main():
 checks=[ROOT/'results/mathematical_checks.json',ROOT/'results/output_checks.json',ROOT/'results/revision/revision_checks.json',ROOT/'expansion/checks_preflight.json',ROOT/'expansion/checks_outputs.json',ROOT/'expansion/analysis/diagnostic_checks.json',O/'round2_checks.json']
 nc=sum(len(json.loads(p.read_text())) for p in checks)
 s=pd.read_csv(O/'synthetic_summary.csv');h=s[(s.study=='heston')&(s.alpha==.01)].set_index('model');pair=json.loads((O/'vae_h_vs_g.json').read_text());lev=json.loads((O/'leverage.json').read_text());a=lev['rows'][0]['metrics']['0.01'];b=lev['rows'][1]['metrics']['0.01']
 density=pd.read_csv(O/'matched_density.csv');d=density[density.study.eq('heston')].set_index('model');ch=pd.read_csv(O/'checkpoint_changes.csv');inf=pd.read_csv(O/'selection_inference.csv')
 sensitivity=pd.read_csv(O/'selection_summary.csv');hs=sensitivity[sensitivity.study.eq('heston')].dropna(subset=['rmse']);hb=hs.loc[hs.rmse.idxmin()]
 cp={p:pd.read_csv(O/(p+'_summary.csv')).query('alpha == .01').set_index('model') for p in ['crsp5','crsp100']}
 intro='''# Tail Architecture, Density Fit, and Conditional Risk Forecasting

Miquel Noguer i Alonso — November 2026

`main.pdf` and `main.tex` contain the completed second referee revision. The prior 33-page manuscript and its response are preserved in `referee_round2/baseline/`. Original and first-expanded results remain traceable; superseded tables are not silently combined with current inference.

## What changed

- Added plain Meta-t, VAE-tlatent and independent-noise WGAN-i to all ten Heston datasets.
- Replayed 380 identical neural training trajectories and evaluated six checkpoint rules where likelihood is defined, five for GANs. All 2,150 rule/model evaluations are retained, including unfavorable results. These are repeated evaluations, not independent estimation datasets.
- Removed NLL from primary risk tables. Separate tables compare risk-selected, native likelihood-selected, and final checkpoints on common test observations; the earlier unpaired density-selected archive is clearly labeled.
- Added GARCH-FHS and portfolio-EWMA-FHS to both CRSP panels. A matched-location control separates aggregation, location, variance filtering, and residual-tail effects. Primary inference now uses fourteen FHS contrasts and a fifteen-model confidence set.
- Added a one-million-path paired leverage experiment, holding cross-asset price-shock covariance fixed in the principal ablation.
- Corrected claims about correct specification, paired VAE errors, static weekly leverage, return-unit dependence and missing-return bias. Verified a public author-uploaded source for the portfolio application of meta-t.

## Principal findings

'''
 intro+=f"The primary neural comparisons do not improve FHS after within-panel Holm correction. GARCH-EVT has the lowest five-basket mean ({cp['crsp5'].loc['GARCH-EVT','fz0']:.4f}); GARCH-FHS has the lowest 100-stock mean ({cp['crsp100'].loc['GARCH-FHS','fz0']:.4f}). Under a common GARCH filter, the incremental EVT contribution is not detectable in either panel. Most of the point-score improvement is associated with the fitted variance filter. These are retrospective fitted-specification comparisons, not universal winners.\n\n"
 intro+=f"Both heavy-tailed synthetic laws belong to the meta-t family, so its favorable estimation errors partly reflect correct specification. Under primary Heston checkpoint selection, Meta-t RMSE is {h.loc['Meta-t','rmse']:.3f}, VAE-tlatent {h.loc['VAE-t','rmse']:.3f}, and WGAN-i {h.loc['WGAN-i','rmse']:.3f}, compared with exact empirical ES {h.empirical_rmse.iloc[0]:.3f}. VAE-H minus VAE-G paired MSE is {pair['delta_mse']:.4f}, SE {pair['paired_se']:.4f}; there is no detectable difference.\n\n"
 intro+=f"The Heston ranking is selection-sensitive: the lowest secondary RMSE is {hb.rmse:.3f} for {hb.model.replace('VAE-t','VAE-tlatent')} under {hb.criterion} selection, with paired MSE difference from empirical ES {hb.paired_mse_from_empirical:.3f} (SE {hb.paired_empirical_se:.3f}). This is an exploratory choice among {len(hs)} rule/model combinations, not a confirmed winner. The primary empirical lead must not be generalized to all selection rules.\n\n"
 intro+=f"Removing leverage at fixed price-shock correlation 0.549 changes one-percent weekly ES from {a['es']:.3f} to {b['es']:.3f}, a {100*(a['es']-b['es'])/a['es']:.2f}% reduction. Static joint-window generators can encode within-window leverage; they are not recursive state-conditioned models. The older 5.36/4.78 ablation is not attributed to the current joint-bank target.\n\n"
 intro+=f'''## Executed design and verification

- 730 retained primary evaluations and 520 fresh primary neural fits: Student 100, heterogeneous meta-t 110, Heston 120; five baskets 200, 100 stocks 100, and full-covariance five-basket refits 100.
- Ten independent datasets per synthetic law and one optimization seed per dataset; one million primary evaluation scenarios per fit; four million nonanalytic reference draws.
- 2,514 CRSP test dates in 2016–2025; 100,000 neural scenarios; exact empirical FHS. Two seeds for the primary five-basket study, one for 100 stocks and checkpoint sensitivity.
- 380 matched training replays; 2,150 checkpoint-rule evaluations; 330 ancillary model diagnostics. GARCH controls reuse fitted parameters and states, rather than fitting new GARCH models.
- {nc} recorded checks pass. They cover the stated mathematical, data, replay, accounting and numerical contracts, not statistical superiority or exhaustive code correctness.
- The original 380 evaluations/272 neural fits remain archived. Forty-one rejected cold-start pilot evaluations remain excluded.

## Reproduction

Extract both `Generative_Models_Finance_LaTeX.zip` and `Generative_Models_Finance_Reference_Samples.zip` into the same directory before running the commands below. The companion archive contains the two large four-million-draw reference-loss files; all experiments, forecasts, checkpoints and other inputs are in the main archive. `REFERENCE_MANIFEST.sha256` verifies the companion files. No results have been removed to reduce the main download size.

```bash
python -m pip install -r requirements.txt
python code/run_all.py --mode outputs --pdf
```

Output mode reconstructs current tables and findings from retained records and runs numerical checks, including small preflight fits. It does not repeat the full training study. `code/run_round2.py` is the direct current entry point. `code/run_expanded_archive.py` reconstructs the prior expanded design and must not be mistaken for the current release.

```bash
python code/run_all.py --mode experiments --jobs 2 --pdf
```

Experiment mode completes the expanded design and second-round additions, reusing completed records. A clean rerun should use a separate extracted copy with result, checkpoint, selection and diagnostic directories moved aside; preserve protocols, derived inputs and references. Fixed seeds and replay assertions expose implementation drift; low-order floating-point changes may occur across platforms.

To rebuild panels from authorized raw CRSP extracts, add `--raw-crsp /absolute/path/to/crsp_extracts`. The 21 annual source digests and construction audit are retained. Raw licensed CRSP extracts are not redistributed. Saved derived panels permit reconstruction without them.

## Data and limitations

The 100-stock annual universe uses training-end size, at least 98% finite training observations, and at least 252 finite observations before estimation. Validation and test survival are not screened. Missing subsequent returns are filled with zero and explicitly masked. This can suppress downside risk under distress-related missingness; the stress check does not identify unobserved returns. The initial cold-start pilot was discarded in full, and an incomplete 2024 derived export was repaired and independently reconstructed before retained fitting.

The alternative validation criteria and augmented benchmarks are retrospective. The primary Holm families contain fourteen FHS contrasts per panel; matched-filter attribution uses six contrasts; neural checkpoint sensitivity uses forty rule/model contrasts per panel. These are separate families, not a global guarantee over every analysis. Rare validation exceedances remain a limitation even after pooling portfolios. Bootstrap inference conditions on saved forecasts rather than rerunning all estimation. No correctly specified Heston likelihood or particle filter is fitted. Ten datasets, one high-dimensional seed, a finite Heston variance bank and numerical time discretization remain explicit limitations.

## Package map

| Path | Contents |
|---|---|
| `main.tex`, `theory.tex`, `heston_appendix.tex`, `expanded_*.tex`, `round2_*.tex`, `refs.bib` | Current manuscript and methods |
| `referee_round2/protocol.json` | Fixed retrospective additions and competing validation rules |
| `referee_round2/results/`, `checkpoints/` | Added Heston fits and matched-filter forecasts |
| `referee_round2/selection/` | Every validation trace, selected state, replay check and evaluation |
| `referee_round2/data/`, `diagnostics/` | Coupled leverage samples and added within-window diagnostics |
| `referee_round2/analysis/` | Current tables, all CSVs, inference, density comparisons and checks |
| `referee_round2/baseline/` | Prior 33-page source and response |
| `expansion/` | First-expanded raw results, primary states, references and all original diagnostics |
| `results/`, `checkpoints/`, `legacy_outputs/` | Original experiment archive |
| `ROUND2_RESPONSE.md` | Point-by-point response to the latest report |
| `REVISION_AUDIT.md` | Current audit and links to earlier resolved issues |
| `MANIFEST.sha256` | Release checksums |

Legacy machine-readable names `Historical` and `VAE-t` remain for traceability. Manuscript labels distinguish filtered CRSP FHS from synthetic empirical estimation and use VAE-tlatent for the bounded-decoder model.
'''
 (ROOT/'README.md').write_text(intro)
 response='''# Response to the second referee report

The manuscript and experiments have been revised throughout. This response distinguishes new computations from interpretive corrections. Prior results remain archived.

## 1. Correct specification and the Heston comparison

The abstract and controlled-results section explicitly state that both Student and heterogeneous meta-t targets belong to the fitted meta-t family; the residual flow contains its target at the identity map. These are favorable specification conditions, not robustness evidence. We chose the requested alternative of fitting plain Meta-t to Heston, rather than claiming an unavailable Heston MLE or particle-filter oracle. VAE-tlatent and WGAN-i were added at the same time, on all ten existing estimation datasets with unchanged training budgets and one-million-scenario evaluation.

'''
 response+=f"Heston RMSEs: Meta-t {h.loc['Meta-t','rmse']:.4f}, VAE-tlatent {h.loc['VAE-t','rmse']:.4f}, WGAN-i {h.loc['WGAN-i','rmse']:.4f}; exact empirical-training ES {h.empirical_rmse.iloc[0]:.4f}. Paired errors and standard errors appear in the revised tables.\n\n"
 response+='''## 2. Separate density assessment from risk selection

NLL columns have been removed from all primary risk tables. New tables retain native likelihood-validation and final checkpoints along the identical training trajectories, evaluated on the same test observations. Flow selection uses exact validation NLL, VAE selection the original fixed IWAE-8 criterion; test VAE estimates use IWAE-64. The earlier density-selected archive is also shown, but explicitly marked unpaired because its samples and seeds differ. No state is described as a global likelihood optimum. The actual output affine map is diagonal, so containment of a general full-covariance Gaussian cannot be inferred from a dense-affine-head argument.

'''
 response+=f"On Heston, Flow-G NLL is {d.loc['Flow-G','fz1']:.3f} under risk selection and {d.loc['Flow-G','density']:.3f} under native likelihood selection. Flow-L changes from {d.loc['Flow-L','fz1']:.3f} to {d.loc['Flow-L','density']:.3f}. Conclusions no longer use risk-selected NLL as a family density ranking.\n\n"
 response+='''## 3. Rare-event checkpoint sensitivity

All 380 matched trajectories compare 1%, 2.5%, 5%, pooled-portfolio 1% FZ0 and the final checkpoint; non-GAN models additionally compare likelihood selection. The original 1% checkpoint parameters are independently compared with the retained primary states. Every criterion uses the same training path, candidate step grid and validation random numbers. Synthetic pooling uses the existing margins or asset-week losses; CRSP pooling uses the three existing evaluation portfolios. The nominal count of roughly 2.5 validation exceedances per CRSP year is stated. Pooling does not create independent observations or eliminate selection noise. The full tables and paired CSVs report adverse as well as favorable changes. CRSP sensitivity is seed zero, clearly distinguished from the two-seed primary five-basket comparison.

'''
 for study in ['crsp5','crsp100']:
  z=ch[(ch.study==study)&ch.criterion.isin(['fz2p5','fz5','pooled1'])];sig=inf[(inf.study==study)&(inf.delta<0)&(inf.p_holm<.05)]
  response+=f"{study}: {100*z.changed.mean():.1f}% of alternative-risk-rule choices change the checkpoint; {len(sig)} improvements survive the separate exploratory forty-contrast Holm correction. Full rule/model results are retained in `selection_inference.csv`.\n\n"
 response+=f"Heston provides a consequential reversal: the lowest secondary RMSE is {hb.rmse:.3f} under {hb.model.replace('VAE-t','VAE-tlatent')} / {hb.criterion}, below empirical ES {h.empirical_rmse.iloc[0]:.3f}. Its paired MSE difference from empirical ES is {hb.paired_mse_from_empirical:.3f}, SE {hb.paired_empirical_se:.3f}. The abstract and main text now restrict the empirical-estimator lead to the primary checkpoint rule. This secondary choice is exploratory among {len(hs)} rule/model combinations; it is not an independently confirmed model-selection policy.\n\n"
 response+='''## 4. Filter versus tail attribution

GARCH-FHS uses precisely the retained GARCH-t location and predictable variance states, replacing only the innovation law by the empirical training residual distribution. Portfolio-EWMA-FHS is added under the same chronological design. A further diagnostic holds the GARCH location fixed under EWMA to separate the mean and variance changes. The daily decomposition is exact; six attribution contrasts receive a separate Holm adjustment. The fifteen-model primary MCS and fourteen FHS contrasts were recomputed, rather than reusing the old twelve-contrast values.

Most of the observed improvement is associated with variance filtering. The incremental EVT-minus-GARCH-FHS differences are -0.0054 in five baskets and +0.0025 in 100 stocks, with both intervals containing zero. GARCH-FHS is the best observed 100-stock score. The abstract therefore says GARCH-filtered statistical benchmarks lead, rather than continuing to call GARCH-EVT the winner in both panels.

## 5. VAE-H versus VAE-G

'''
 response+=f"The contradictory sentences have been replaced by 'no detectable difference.' The direct dataset-paired MSE difference is {pair['delta_mse']:.6f}, with SE {pair['paired_se']:.6f}. This is the correct SE for H-minus-G; the separate model-versus-empirical SE of about 0.081 is not its paired SE. Neither superiority nor equivalence is established.\n\n"
 response+='''## 6. Meta-t definition and public attribution

The distribution implemented here is the established heterogeneous Student-margin/t-copula construction. We retain Fang–Fang–Kotz and Demarta–McNeil attribution and make no unsupported claim of inventing a different law. The single portfolio-application citation now points to the publicly retrieved author-uploaded copy:

https://www.researchgate.net/publication/398906455_Meta-t_Portfolio_Optimization

The PDF itself is dated December 19, 2025; its upload is dated May 22, 2026. The page metadata's January 2025 label does not replace the printed manuscript date. Definition 2.1 agrees with the construction used here. We could not independently retrieve the supplied Zenodo record through the available retrieval route, which is not proof that it is broken; the bibliography uses the verified public copy and does not rely on SSRN.

## 7. Leverage and missing Heston rows

'''
 response+=f"The previously omitted three model rows are now computed. A new controlled ablation uses one million coupled paths from the current joint-bank simulator. Holding price-shock correlation fixed at 0.549, removing leverage lowers 1% weekly ES from {a['es']:.6f} to {b['es']:.6f}, a {100*(a['es']-b['es'])/a['es']:.2f}% reduction (paired SE of the ES difference {b['paired_batch_se']:.6f}). A separate row holds orthogonal-shock correlation fixed and explicitly identifies the resulting confounding change in total price correlation. The older 5.36 versus 4.78 figures are not reused for a different numerical target.\n\n"
 response+='''A static joint distribution on an ordered five-day window can represent within-window return/next-day-volatility dependence. The revision therefore does not make the mathematically false claim that all static-window generators cannot represent leverage. Instead, it distinguishes their within-window statistic from a recursively simulated, state-conditioned dynamic model. Every Heston model now has the same dependence diagnostics.

## Minor issues

- Abstract shortened to about 170 words, beginning with the primary FHS/multiplicity result and reporting the revised statistical-benchmark conclusion.
- The two heavy-tailed studies and their tables finish before the Heston discussion and table, enforced by a float barrier.
- The exact FZ0 identity is stated: common positive return rescaling by c adds log(c), leaving score differences unchanged. Raw score levels require common units and a common scoring convention.
- Main text explicitly explains that zero-filled missing slots remove variation and may suppress downside risk, especially under distress-related missingness. The severe fixed-forecast stress and complete-date comparison now include the added benchmarks. There is no claim that the sign of every unobserved-return bias is identifiable.

## Scope and remaining limits

All requested empirical additions or stated alternatives are executed. The added sensitivity is retrospective, not an untouched confirmation sample. No correctly specified Heston density, globally optimized neural likelihood, additional 100-stock optimization seed, or full bootstrap re-estimation is claimed. The previous mathematical corrections, source-panel reconstruction, calibration diagnostics, literature additions and excluded pilot audit remain in the supplied archive. Explicit theoretical results are unchanged by this revision.
'''
 (ROOT/'ROUND2_RESPONSE.md').write_text(response)
 audit=f'''# Current revision audit

The current paper responds to the second referee report. See [ROUND2_RESPONSE.md](ROUND2_RESPONSE.md) for every major and minor point, including new experiments and qualifications.

The prior first-round response is preserved at `referee_round2/baseline/REVISION_AUDIT.md`; its numerical tables are superseded where the augmented comparison changes the model family or selection design. Its theory, data corrections, public-literature attribution and historical audit remain traceable.

Current counts: 730 primary evaluations, 520 primary neural fits, 380 matched neural training replays, 2,150 rule/model evaluations and 330 ancillary evaluations. The GARCH residual-law controls reuse estimated states. All {nc} recorded checks pass. These categories must not be added together and labeled independent datasets.

The scientific conclusions now separate correct specification, volatility filtering, residual-tail modeling, architectural asymptotics, density selection and finite-level risk selection. No universal best model or family-level dominance is claimed. Reproduce with `python code/run_all.py --mode outputs --pdf`.
'''
 (ROOT/'REVISION_AUDIT.md').write_text(audit)
 print('Current README and referee response written')
if __name__=='__main__':main()

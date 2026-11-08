"""Current release documentation generated from the verified repaired outputs."""
import json
from pathlib import Path
import pandas as pd
from round4_analysis import ROOT,OUT,O

def main():
    v=json.loads((O/'verification.json').read_text());p=pd.read_csv(O/'primary_summary.csv');p=p[p.alpha==.01]
    d=pd.read_csv(O/'density_crsp.csv');s=pd.read_csv(O/'synthetic_summary.csv');c=pd.read_csv(O/'checkpoint_changes.csv')
    findings=[]
    for panel in ['crsp5','crsp100']:
        z=p[p.panel==panel].sort_values('fz0');best=z.iloc[0];sig=z[(z.delta<0)&(z.p_holm<.05)]
        exact=d[(d.panel==panel)&(~d.model.str.startswith('VAE'))].sort_values('nll').iloc[0]
        findings.append(f'- {panel}: lowest observed primary FZ0 is {best.model} ({best.fz0:.4f}); {len(sig)} of 16 FHS contrasts establishes improvement after Holm correction. Lowest observed exact NLL in the original-head comparison is {exact.model} ({exact.nll:.3f}).')
    text='''# Parametric and Nonparametric Multivariate Models in Finance

Miquel Noguer i Alonso — Artificial Intelligence Finance Institute

Reserved DOI: https://doi.org/10.5281/zenodo.24024955

`main.tex` and `main.pdf` are the finalized manuscript for the evaluated design. The primary comparison contains 17 forecasts and 16 FHS contrasts per CRSP panel. Historical results are preserved in explicitly named archives; they are not current tables. The Zenodo draft contains only the manuscript PDF.

## Empirical design and verification

- Replayed the first 250 updates of all 640 retained neural trajectories. Selection now admits initialization, updates 10/25/50/100 and the original 250-step grid, plus exact identity or nested parametric flow candidates. The final-state comparator is unchanged. Validation alone chooses among the expanded candidates.
- Added 160 training-only joint fits: multivariate Student MLE with full location/scatter and bounded degrees of freedom, and Ledoit–Wolf shrinkage Gaussian. Linear-projection VaR and ES are analytic.
- Retained pointwise NLL, paired block uncertainty for CRSP and separately reported fitting-dataset/shared-test dispersion for synthetic laws. Likelihood-selected tables are separate from risk selection.
- Recomputed primary inference, checkpoint sensitivity, dense-output controls, coordinate-GARCH controls, full-covariance filtering, crash-day sensitivity, calibration and model-confidence-set composition sensitivity.
- Added a common-filter joint downside-event Brier comparison beyond the tested portfolios. This retrospective diagnostic is a separate multiplicity family.
- Consolidated the manuscript while retaining parameter summaries, exact error decomposition and all machine-readable sensitivity records.

## Results and limits

'''+ '\n'.join(findings)+'''

These results condition on the selected forecasts, fixed architectures, budgets, annual stock universes and declared exposures. Non-rejection is not equivalence. Selection repair was fixed before evaluating its added states, but was motivated by an audit of the existing study and remains retrospective. Sparse one-percent validation tails remain a limitation. No untouched future sample is claimed.

The two heavy-tailed synthetic laws are correctly specified for meta-t; the homogeneous Student target is also correctly specified for the added Student benchmark. No correctly specified scalar-Heston dynamic benchmark is fitted. The restricted Wishart fit's synthetic tail mismatch and its CRSP filtering failure are separate findings. Neither establishes a ranking of the Heston-Wishart model family or isolates the incremental value of stochastic correlation. The numerically unresolved CRSP filter is excluded from primary inference.

## Reproduction

Extract all four archives into the same directory, preserving their relative paths:

1. `Generative_Models_Finance_LaTeX.zip` — source, derived panels, original records and current analyses.
2. `Generative_Models_Finance_Reference_Samples.zip` — large reference-loss samples.
3. `Generative_Models_Finance_Round3_Checkpoints.zip` — dense and coordinate-filter states.
4. `Generative_Models_Finance_Round4_Results_Checkpoints.zip` — repaired states, forecasts and pointwise densities.

Each part has a SHA-256 manifest. No raw licensed CRSP extracts are redistributed. Derived panels and masks support output reconstruction. Rebuild with:

```bash
python -m pip install -r requirements.txt
python code/run_all.py --mode outputs --pdf
```

This is the current entry point. It validates the retained repair, regenerates inference/tables/figures/manuscript and compiles the PDF. It does not repeat the full training study. Earlier editorial processors rebuild earlier versions and must not be used alone for this release.

To recompute the full study in an isolated copy:

```bash
python code/run_all.py --mode experiments --jobs 3 --pdf
```

Completed records are reused. A clean rerun must move the result, checkpoint, selection and diagnostic directories aside while preserving protocols, derived inputs and reference samples. Add `--raw-crsp /absolute/path/to/authorized/extracts` to rebuild from licensed sources. Fixed seeds and update-250 replay assertions expose implementation drift; low-order floating-point differences can occur across hardware/software.

## Record ledger and verification

The current design retains 1,110 model/fold records, including ten unresolved Wishart CRSP pilots. The 640 prefix replays reuse existing fitted trajectories and are not additional independent fits. The 2,100 scalar coordinate-filter estimates are separate submodels. Multiple checkpoint rules also reuse one trajectory.

'''+f'The release reconstructs {v["forecast_rows_checked"]:,} forecast rows in {v["forecast_files_checked"]} unique files. Maximum score reconstruction error is {v["maximum_score_reconstruction_error"]:.3g}; maximum saved update-250 parameter replay difference is {v["max_replay_parameter_error"]:.3g}. All current release contracts pass. These checks establish implementation consistency, not statistical superiority or exhaustive code correctness.\n'+'''
## File map

| Path | Role |
|---|---|
| `code/run_all.py`, `code/run_round4.py` | Current reproduction entry point |
| `code/round4_*.py` | Estimators, repaired evaluations and analyses |
| `code/check_round4_*.py` | Independent kernel and release checks |
| `referee_round4/protocol.json` | Fixed repair and comparison definitions |
| `referee_round4/results/`, `checkpoints/`, `pointwise/` | Current fitted/evaluated records in companion archive |
| `referee_round4/analysis/` | Current tables, CSVs and verification |
| `referee_round4/baseline/` | Exact manuscript/source state before this repair |
| `referee_round3/`, `referee_round2/`, `expansion/` | Earlier protocols, controls and retained records |
| `results/`, `checkpoints/`, `legacy_outputs/` | Original study archive |

Machine identifiers `Historical` and `VAE-t` remain for traceability. Paper labels distinguish CRSP FHS from synthetic empirical ES and identify the bounded-decoder VAE as VAE-tlatent.
'''
    (ROOT/'README.md').write_text(text)
    response='''# Repair response — 7 November 2026

| Issue | Implemented correction | Remaining scope |
|---|---|---|
| Initialization omitted from checkpoint selection | All 640 retained neural paths receive the same early grid; exact admissible flow base candidates are included | Retrospective repair; no test-selected candidate |
| Missing fitted joint Student and regularized Gaussian | 160 training-only fits across synthetic, primary, covariance and coordinate-filter units; analytic VaR/ES | Numerical MLE with three starts, not a proof of global optimization |
| Unpaired or risk-selected NLL interpretation | Pointwise matched likelihood-selected NLL; paired CRSP intervals and Holm correction | Conditional on fitted models; IWAE bias/noise explicit |
| Synthetic shared-sample uncertainty | Dataset and common-test dispersion shown separately; same-sample oracle rows retained | Heavy tails can invalidate normal-theory squared-error/NLL intervals |
| Multivariate evaluation limited to a few portfolios | Added co-exceedance-event Brier scores for the same joint event under a common filter | One event does not identify the entire joint law |
| Filter attribution and dense dependence | Recomputed matched portfolio, coordinate-GARCH, fitted-EWMA and dense-output controls | Dense comparison changes capacity, initialization and nested candidates together |
| Symmetric null, multiplicity and MCS | 17-model primary set; 16 contrasts; 28-model broad set; full/reduced MCS and 10/20/40-day block sensitivity | Separate analysis families provide no global research-wide error guarantee |
| Wishart mismatch and numerical convergence | One-line stationary dispersion calculation, fitted kappa comparison, CRSP pilot excluded from inference | Frozen-mixture bound is not a dynamic-leverage kurtosis theorem |
| Heston density/risk explanation and leverage | Near-Laplace snapshot, infinite Student fourth moments, symmetry restriction and corrected leverage ablation retained | Central-peak explanation is suggestive, not identified causality |
| Attribution, references and counts | Established meta-t attribution, full parameter tables, original bibliography plus Student MLE and shrinkage sources, 1,110-record ledger | Companion Riccati transform is explicitly not used |
| Excess length and repeated caveats | Consolidated main tables and limitations; detailed sensitivities remain machine-readable | Complete reproducibility requires the core and numerical archive parts |

Current numerical findings are generated in the manuscript and `referee_round4/analysis/`; superseded results remain in the baseline archive. The repair does not change the observed CRSP returns, held-out dates, original training trajectories or original final states.
'''
    (ROOT/'ROUND4_RESPONSE.md').write_text(response)
    (ROOT/'REVISION_AUDIT.md').write_text('# Current revision audit\n\nThe current release is the round-four repair. See `ROUND4_RESPONSE.md`, `referee_round4/protocol.json`, `referee_round4/analysis/verification.json` and `referee_round4/analysis/kernel_checks.json`. Earlier audit documents and their corresponding manuscript states are retained under the named baseline archives.\n')
    print('WROTE current release documentation')

if __name__=='__main__':main()

# Tests and saved results for the comparative paper

Parametric and Nonparametric Models of Joint Returns: Density, Dependence, and Portfolio Tail Risk
Miquel Noguer i Alonso — manuscript dated November 7, 2026
DOI: https://doi.org/10.5281/zenodo.24024955

This folder belongs to the comparative parametric/nonparametric paper, not the separate HWPO manuscript.
Overleaf stores these files and compiles the paper. Python experiments must run locally after download.

## Run the supplied verification

1. Download the project source from Overleaf, unzip it, and enter `paper_tests`.
2. Run `python RUN_TESTS.py` to verify archive hashes and extract the research tree into `research_tests`.
3. Install the recorded dependencies: `python -m pip install -r research_tests/requirements.txt`.
4. Run `python RUN_TESTS.py --run` to check the payload hashes and execute the current-release verifier.

The archives are independent tar.gz volumes; all are required. The runner extracts them into one common directory.
Keep the supplied directory layout: the research code resolves dependencies relative to its project root.
The runner does not edit the manuscript, retrain all models, or change any model ranking.

## Verified in this package

The current-release verifier completed successfully before upload:
- 47 reference numerical files matched their recorded SHA-256 hashes.
- 640 saved neural trajectories and 160 added joint-model fits passed release consistency checks.
- FZ0 scores were reconstructed for 945,236 rows in 1,054 forecast files; maximum error was 1.14e-12.
- The checker verified common evaluation dates, Holm arithmetic, paired error decomposition, and Student-fit diagnostics.
- Independent Wishart–CIR detailed-balance and frozen-characteristic-function checks were rerun.
- The four-run completion audit reconstructed another 6,000 forecast rows.

The Wishart–CIR conditional numerical-stability gate FAILED, as reported in the manuscript.
The package preserves that finding: `conditional_ranking_eligible` remains false.
A successful verification run means the saved evidence is internally consistent; it does not mean every model passes calibration or numerical eligibility.
See `verification_run.txt` and the three JSON summaries beside this guide.

## Contents and scope

- `code/`: original experiment, estimation, simulation, inference and verification scripts.
- `referee_round4/`: repaired experiment records, referenced selected/final checkpoints, pointwise likelihoods, forecasts and analysis tables.
- `expansion/`, `referee_round2/`, `referee_round3/`: supporting analysis and checkpoint dependencies referenced by the repaired records.
- `referee_round5/analysis/`: final dependence, DCC/CCC, missing-return and quadrature outputs.
- `wishart_extension/`, `wishart_completion/`, `wishart_cir_extension/`, `completion_audit/`: specification studies, fit records, numerical diagnostics and the failed conditional-stability audit.
- `final_submission/editorial_numerical_baseline.json`: the 47-file numerical baseline.
- `data/`: the derived five-basket panel used by the completion verifier and its source-audit metadata.
- `requirements.txt`: versions recorded by the research release.

`test_catalog.csv` lists the check/verify scripts. The supported self-contained entry point is `RUN_TESTS.py --run`.
Other scripts are archived research stages; some need the complete research/reference archives and should not be run indiscriminately.
This is a saved-results verification package, not a claim to rerun every historical fit from raw data.
Large reference-simulation banks, all historical scenario arrays and raw licensed CRSP extracts are not included.
Full from-scratch replication requires the original research archives and authorized CRSP source data.
Historical `Historical` and `VAE-t` labels in machine files correspond to FHS and VAE-tlatent in the manuscript.

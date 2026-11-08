# Parametric and Nonparametric Models of Joint Returns

**Density, Dependence, and Portfolio Tail Risk**  
Miquel Noguer i Alonso — Artificial Intelligence Finance Institute  
Manuscript: **November 7, 2026**  
DOI: [10.5281/zenodo.24024955](https://doi.org/10.5281/zenodo.24024955)

This repository accompanies the integrated comparative paper. It compares Gaussian, shrinkage-Gaussian, Student and meta-t laws, filtered historical simulation, normalizing flows, variational autoencoders and Wasserstein GANs. Matched volatility/correlation filters, checkpoint rules, synthetic laws and chronological CRSP panels separate density fit, downside dependence and fixed-portfolio risk forecasting.

Repository: https://github.com/MiquelNoguerAlonso/parametric-nonparametric-finance

## Start here

- [Paper PDF](paper/Paper.pdf) and [LaTeX source](paper/main.tex).
- [Experiment and analysis code](code/).
- [Saved-results test runner](tests/verify_release.py) and [verification output](tests/verification_run.txt).
- [Release asset manifest](release_assets.json): exact filenames, sizes, hashes and extraction order for all six research archives, the Overleaf/test package and the PDF.
- [Reproduction instructions](REPRODUCTION.md).

The accompanying release distributes the large reference samples, selected model states, forecasts, pointwise likelihoods, numerical audits, and the complete Overleaf package. All six research ZIPs are parts of one paper and must be extracted into one common research directory. This source checkout keeps the code and smaller analysis tables browsable; the binary archives are release assets.

## Verify the saved evidence

Download the six research archives listed in `release_assets.json` into a local directory, then run:

```bash
python tests/verify_release.py --archives /path/to/downloaded/assets --extract-to research
python -m pip install -r research/requirements.txt
python tests/verify_release.py --archives /path/to/downloaded/assets --extract-to research --run
```

The verifier checks archive hashes before extraction and executes the original current-release checks. It does not train all models. The separately supplied Overleaf ZIP has its own `paper_tests/RUN_TESTS.py` entry point for the smaller saved-results verification package.

## What the checks establish

The archived release check reconstructs 945,236 forecast rows in 1,054 files, checks 47 frozen numerical files, and verifies common dates, Holm arithmetic, Student-fit diagnostics and paired error decompositions. The subsequent Wishart–CIR audit reconstructs a further 6,000 forecast rows.

**The Wishart–CIR conditional numerical-stability gate fails.** That result is preserved in [the recorded numerical decision](tests/numerical_decision.json); successful evidence verification does not establish filter convergence or model superiority. The study evaluates fixed portfolio exposures and does not validate optimized portfolio weights.

## Data and interpretation

Raw licensed CRSP extracts are not redistributed. Derived panels, masks and recorded forecasts are supplied in the research archives; rebuilding the source construction requires authorized original data. Synthetic reference banks are supplied separately among the six archives. Historical response files and editorial scripts record earlier research stages; the current manuscript, frozen numerical baseline and completion audit determine the current interpretation.

Machine labels `Historical` and `VAE-t` correspond to FHS and VAE-tlatent in the paper. Please cite the manuscript DOI. No new license grant is added by this packaging step.

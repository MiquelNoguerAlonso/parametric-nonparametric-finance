# Reproduction guide

## Compile the paper

The `paper` directory is self-contained. Run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` inside it, or upload its LaTeX, bibliography and two figure PDFs to Overleaf. Python and data are unnecessary for compilation.

## Archive integrity and saved-results verification

Download the six `research` assets named in `release_assets.json`. Run `tests/verify_release.py` from this checkout, pointing `--archives` to those files. It verifies every SHA-256 before extracting, preserves relative paths and checks overlapping archive entries for conflicting bytes. Install the extracted `requirements.txt`, then repeat with `--run`.

The command runs `research/code/verify_current_release.py`, preserving the current manuscript. Its checks include the 47-file frozen baseline, forecast/likelihood consistency, Holm calculations, error accounting, independent Wishart–CIR stationary-law tests and the final numerical eligibility decision. The Wishart–CIR gate remains failed; this is an expected scientific result, not an instruction to delete observations or adjust thresholds.

## Full experiment reruns

The code and retained protocols are supplied. Saved-result verification is distinct from retraining. Work in an isolated copy of the extracted research tree. Consult its `code/run_all.py` options and retained protocol JSONs before rerunning fits. Historical editorial processors can restore older prose; do not run them alone over the current manuscript. Do not infer a newly run experiment from an archived result.

The complete source archive's `README.md` documents the historical editorial reconstruction sequence. Its old publication-status statements describe archive preparation and are not evidence of current Zenodo status. Use the DOI and current manuscript as the publication references.

The raw-CRSP construction option requires an authorized extract supplied by the user. Neither this repository nor its release grants a CRSP data license. Reference samples, fit artifacts and numerical audits are organized in six research archives; the additional Overleaf ZIP is a convenient manuscript and saved-results verification distribution.

## Current scope

Ten estimation datasets per synthetic law; 2,514 chronological CRSP test dates in five-basket and 100-stock panels. Checkpoint alternatives reuse fitted trajectories. Prefix replay is not an independent fit. Numerical eligibility and conditional ranking are separate from mathematical correctness and frozen-law fit.

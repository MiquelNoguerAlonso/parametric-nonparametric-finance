"""Apply extension after the existing round-5/6/7 editorial processors."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def apply():
    p=ROOT/'main.tex';s=p.read_text()
    if '\\input{wishart_cir_main.tex}' not in s:
        s=s.replace('\\section{Implications and limitations}', '\\input{wishart_cir_main.tex}\n\\section{Implications and limitations}')
    if '\\input{wishart_cir_theory.tex}' not in s:
        s=s.replace('\\section{Target-law leverage and reproducibility}','\\input{wishart_cir_theory.tex}\n\\FloatBarrier\n\\section{Target-law leverage and reproducibility}')
    s=s.replace('and a restricted Heston--Wishart stochastic-covariance specification.','a restricted Heston--Wishart stochastic-covariance specification, and an additive Wishart--CIR extension.')
    qualification=' The additive extension relaxes the dispersion restriction, but its synthetic results remain sensitive to the common leverage specification (Section~\\ref{sec:wc-results}).'
    while qualification+qualification in s:s=s.replace(qualification+qualification,qualification)
    if qualification not in s:
        anchor='Neither finding rejects the general Wishart family or identifies the incremental benefit of stochastic correlation.'
        s=s.replace(anchor,anchor+qualification)
    p.write_text(s)
    p=ROOT/'referee_round4/analysis/verification.tex';s=p.read_text()
    s=s.replace('The current archive retains 1,150 model/fold records','The original comparison retains 1,150 model/fold records')
    note=' The Wishart--CIR extension adds twenty training fits and 26 completed numerical filter runs. These reuse the stated estimation samples and are separate from the primary model/fold count.'
    if note not in s:
        s=s.replace('Checkpoint alternatives reuse the same fitted trajectory.','Checkpoint alternatives reuse the same fitted trajectory.'+note)
    p.write_text(s)
    p=ROOT/'referee_round4/analysis/abstract.tex'
    p.write_text('We compare models of joint asset returns to determine when distributional flexibility improves density estimation and portfolio tail-risk forecasts. The study tests Gaussian, shrinkage-Gaussian, Student and meta-t distributions; empirical innovations through filtered historical simulation (FHS); and normalizing flows, variational autoencoders and Wasserstein GANs. Meta-t joins asset-specific Student margins with a t copula, while flows learn invertible transformations of a base distribution. Separate volatility filters turn static innovations into conditional forecasts. Heston--Wishart and an additive Wishart--CIR extension examine stochastic covariance and leverage. Ten independent estimation datasets for each of three synthetic laws and 2,514 out-of-sample CRSP dates in five-basket and 100-stock panels separate specification, conditioning and checkpoint selection. Correctly specified parametric families perform strongly. Heston experiments expose density--risk discrepancies; adding a common variance factor repairs marginal tail dispersion but leaves portfolio ES sensitive to leverage specification. The extension\'s CRSP filter remains unresolved on a stress fold. On CRSP, the largest observed gains arise from variance conditioning, with inconclusive adjusted tests. No tested neural innovation model improves on FHS after primary multiplicity correction under the common risk-selection rule; alternative checkpoint rules change rankings. Student improves in the primary 100-stock family but not the broader retrospective comparison. Density fit, downside dependence and portfolio risk require separate evaluation.\n')
    p=ROOT/'referee_round4/analysis/conclusion.tex';s=p.read_text()
    old='The restricted Wishart fit understates synthetic ES, and its unconverged CRSP filter remains appendix-only. A more flexible, converged stochastic-covariance specification would require a new evaluation.'
    new='The restricted Wishart fit understates synthetic ES, and its unconverged CRSP filter remains appendix-only. The new additive Wishart--CIR experiment relaxes the dispersion restriction, yet the full fitted model still overstates Heston portfolio ES. Its favorable no-leverage intervention is a specification diagnostic rather than evidence that leverage should be discarded. Its finer 2020 numerical checks pass, but the 2025 filter still fails the ES tolerance at 32,768 particles. This separates mathematical flexibility from empirical adequacy; the extension does not enter the original Holm or MCS families.'
    if old in s:s=s.replace(old,new)
    p.write_text(s)
    p=ROOT/'refs.bib';s=p.read_text()
    if '@article{cuchiero2011affine,' not in s:
        s+='\n@article{cuchiero2011affine,\n author={Cuchiero, Christa and Filipovic, Damir and Mayerhofer, Eberhard and Teichmann, Josef},\n title={Affine processes on positive semidefinite matrices},\n journal={Annals of Applied Probability}, year={2011}, volume={21}, number={2}, pages={397--463}, doi={10.1214/10-AAP710}\n}\n'
    if '@article{broadie2006exact,' not in s:
        s+='\n@article{broadie2006exact,\n author={Broadie, Mark and Kaya, Ozgur},\n title={Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes},\n journal={Operations Research}, year={2006}, volume={54}, number={2}, pages={217--231},\n doi={10.1287/opre.1050.0247}\n}\n'
    p.write_text(s)
if __name__=='__main__':apply()

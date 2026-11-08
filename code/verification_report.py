"""Generate the verification appendix directly from saved diagnostic outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'

def main():
    checks=json.loads((R/'mathematical_checks.json').read_text())+json.loads((R/'output_checks.json').read_text())
    assert all(c['passed'] for c in checks)
    h=json.loads((R/'heston_refinement.json').read_text());truth=json.loads((R/'heston_truth.json').read_text())
    s=pd.read_csv(R/'crsp_scenario_sensitivity.csv')
    ranges=s.groupby(['year','model']).mean_fz0.agg(lambda x:x.max()-x.min())
    support=pd.read_csv(R/'crsp_support_diagnostics.csv')
    clip=pd.read_csv(R/'crsp_support_clipping.csv')
    stress=pd.read_csv(R/'crsp_missing_return_stress.csv');stress=stress[(stress.portfolio=='equal')&(stress.alpha==.01)]
    primary=pd.read_csv(R/'crsp_summary.csv');primary=primary[(primary.portfolio=='equal')&(primary.alpha==.01)]
    text=[rf'''\paragraph{{Executed checks.}}
The archived experiment contains 380 completed model evaluations: 60 Student, 66 meta-$t$, 54 Heston, and 200 CRSP annual-fold/seed/model records. There are 272 neural fits. All {len(checks)} mathematical and output checks pass, including unique forecast keys, equal test-date coverage, chronological separation, exact discrepancy accounting, independent score reconstruction, finite readable numerical archives, and four comparisons of simulated Heston exponential moments with the Riccati transform. These checks establish the stated numerical contracts, not a proof of statistical superiority.''']
    t=truth['equal_0.01'];u=truth['equal_0.001'];a,b=h['risk']
    text.append(rf'''\paragraph{{Simulator uncertainty.}}
The 400,000-path Heston reference gives equal-weight weekly ES {t['es']:.5f} at 1\% (20-block Monte Carlo standard-error diagnostic {t['mc_block_se']:.5f}) and {u['es']:.5f} at 0.1\% (diagnostic {u['mc_block_se']:.5f}). These diagnostics do not include uncertainty from the finite initial-state bank. With 100,000 coupled paths, increasing the time resolution from 20 to 40 steps per day changes 1\% ES by {a['difference']:.5f}, with paired block standard error {a['paired_block_se']:.5f}; the 0.1\% change is {b['difference']:.5f}, with standard error {b['paired_block_se']:.5f}. A separately simulated bank at eight steps per day yields mean variance {h['refined_bank']['mean_variance']:.5f}, variance of variance {h['refined_bank']['variance_of_variance']:.6f}, and weekly ES {h['refined_bank']['weekly_es_1']:.3f}. The exact marginal stationary moments are 0.04 and $\xi^2\theta/(2\kappa)=0.0016667$. Small paired discretization differences do not eliminate bank or extreme-tail Monte Carlo uncertainty.''')
    text.append(rf'''\paragraph{{Scenario Monte Carlo sensitivity.}}
For the fixed seed-0 checkpoints in 2020 and 2025, three fresh scenario seeds (11, 22, 33) regenerate 100,000 draws each; the first 20,000 enter each forecast, as in the main protocol. In 2020 the VAE-G annual score ranges over {ranges.loc[(2020,'VAE-G')]:.4f}; the largest range across all evaluated model-years is {ranges.max():.4f}. Close score rankings can therefore be affected by scenario simulation. This sensitivity reuses fitted models and does not estimate training-sample uncertainty.''')
    counts=support.groupby('model').outside_simple_return_support.sum()
    max_score=(clip.clipped_fz0-clip.base_fz0).abs().max();max_es=(clip.clipped_es-clip.base_es).abs().max()
    text.append(rf'''\paragraph{{Arithmetic-return support.}}
Unbounded model innovations can imply arithmetic returns below $-100\%$. At a conservative envelope formed from the maximum forecast scale in each coordinate and year, out-of-support draws total {int(counts['Flow-meta-t'])} for Flow-meta-$t$, {int(counts['Flow-t'])} for Flow-t, and {int(counts['WGAN-t'])} for WGAN-t out of 600,000 draws per model across the two years and three seeds; other models have zero observed envelope violations. This envelope is an upper-bound diagnostic, not a violation frequency on any single date, and zero observations do not imply bounded population support. Recomputing the 20,000-scenario forecasts with each asset return floored at $-100\%$ changes the annual joint score by at most {max_score:.5f} and average forecast ES by at most {max_es:.4f} percentage points in these selected cases. A production model can instead model log returns and transform to arithmetic returns, but must then rederive the portfolio loss and validate exponential moments. That would be a different experiment.''')
    g=stress[stress.model=='Gaussian'].iloc[0];v=stress[stress.model=='VAE-G'].iloc[0]
    assert primary.sort_values('fz0').model.tolist()==stress.sort_values('stress_score').model.tolist()
    text.append(rf'''\paragraph{{Missing returns and multiplicity.}}
Holding forecasts fixed while replacing every missing held return by $-100\%$ changes the primary VAE-G score from {v.base_score:.4f} to {v.stress_score:.4f} and Gaussian from {g.base_score:.4f} to {g.stress_score:.4f}; the primary model ordering is unchanged. This is a sensitivity to the observed losses, not a retrained missing-data analysis. This paragraph concerns the original ten-model panel. The revised primary comparison uses FHS, includes the new dynamic benchmarks, and reports its inference and multiplicity adjustment separately. Density fit, joint risk score, and calibration remain separate criteria.''')
    extra=json.loads((R/'revision/revision_checks.json').read_text())
    assert all(c['passed'] for c in extra)
    text.append(r'\paragraph{Revision checks.} All '+str(len(extra))+r' additional checks pass, including independent GARCH likelihood and ES calculations, predictability, parameter constraints, correlation matrices, the decomposition cross term, interval--test coherence, and model-confidence-set invariance. The dynamic benchmarks add 30 GARCH parameter fits; the existing 272 neural fits are retained.')
    (R/'verification.tex').write_text('\n\n'.join(text)+'\n')
    print('Generated verification appendix')

if __name__=='__main__':main()

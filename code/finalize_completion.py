"""Final editorial pass, driven by the completed numerical eligibility audit."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    status=json.loads((ROOT/'completion_audit/analysis/numerical_decision.json').read_text())
    if status['stress_gate_passed']:
        raise RuntimeError('Stress checks passed: complete the full-panel numerical protocol before final disposition.')
    fine=status['independent_banks'][-1]
    abstract="""We compare models of joint asset returns to determine when distributional flexibility improves density fit, dependence and portfolio tail risk. Parametric benchmarks are Gaussian, shrinkage Gaussian, Student and meta-t; the latter joins asset-specific Student margins with a t copula. Filtered historical simulation (FHS) supplies empirical innovations. Neural alternatives include normalizing flows, which learn invertible transformations, variational autoencoders and Wasserstein GANs. Separate volatility and correlation filters distinguish conditioning from innovation modeling. Ten independent estimation datasets per synthetic law and 2,514 CRSP test dates in five-basket and 100-stock panels support paired comparisons. Under common one-percent validation selection, no tested neural model improves on FHS after primary multiplicity correction; alternative checkpoint rules change rankings. Student's 100-stock improvement is sensitive to the comparison family. Portfolio variance filtering produces the largest observed score gains, with inconclusive adjusted tests. Correct specification favors parametric synthetic fits, while Heston separates likelihood quality from expected-shortfall accuracy. An additive Wishart--CIR construction relaxes the restricted Wishart dispersion ceiling, but its leverage sensitivity and failed filter tolerance prevent a conditional forecasting claim. Density, downside dependence and portfolio risk therefore require separate evaluation.
"""
    (ROOT/'referee_round4/analysis/abstract.tex').write_text(abstract)
    p=ROOT/'main.tex';s=p.read_text()
    contribution=r'\item A stochastic-covariance specification study: an admissibility-induced dispersion ceiling, an additive scalar-variance repair, and a frozen-law equivalence showing why static fit alone cannot identify the value of stochastic correlation.'
    if contribution not in s:s=s.replace(r'\end{enumerate}',contribution+'\n'+r'\end{enumerate}',1)
    start='% BEGIN allocation-scope';end='% END allocation-scope'
    paragraph=r'''% BEGIN allocation-scope
\paragraph{Implications for portfolio modeling.}
These experiments evaluate predeclared portfolios, not weights optimized against each fitted law. Parametric meta-$t$ and its residual flow are tractable candidates for joint scenario modeling, but the likelihood-selected residual is the exact parametric identity in every 100-stock fold. The present evidence therefore establishes no advantage from that learned residual. For allocation, a separate chronological experiment must estimate weights using past information, impose common constraints and costs, and compare realized portfolio outcomes. Density or fixed-portfolio ES rankings alone do not validate an optimization policy.
% END allocation-scope'''
    if start in s:s=re.sub(re.escape(start)+r'.*?'+re.escape(end),lambda _:paragraph,s,flags=re.S)
    else:s=s.replace('\\section{Conclusion}',paragraph+'\n\n\\section{Conclusion}')
    p.write_text(s)
    p=ROOT/'theory_main.tex';s=p.read_text()
    s=s.replace('Let $X=T(Z)$. If', 'Let $X=T(Z)$, with constants $C>0$ and $B\\ge0$. If')
    s=s.replace('for $c>0$, then', 'for $c>0$ and $b\\ge0$, then')
    s=s.replace('For $\\beta>0$, assume', 'For $\\beta>0$ and constants $C,c>0$, assume')
    p.write_text(s)
    p=ROOT/'wishart_cir_main.tex';s=p.read_text()
    anchor='Matching marginal dispersion therefore does not establish better portfolio-tail estimation.'
    extra=' Proposition~\\ref{prop:wc-equivalence} strengthens this point: the frozen stationary return law is exactly a scalar Gaussian variance mixture, so static density or marginal-tail fit cannot identify the benefit of stochastic correlation.'
    if extra not in s:s=s.replace(anchor,anchor+extra)
    p.write_text(s)
    p=ROOT/'wishart_cir_extension/analysis/filter_findings.tex'
    p.write_text(r'\paragraph{Numerical eligibility for conditional comparison.}'+'\n'+
        f"The completed audit retains thirty filter runs. Finer 2020 checks pass, but the 2025 fold fails: at 32,768 particles the guided filter has 4.40\\% independent-bank ES discrepancy, and the final 16,384-particle resample-move control has {100*fine['es_relative_rms']:.2f}\\% against a 2\\% tolerance. Appendix~\\ref{{app:wishart-cir}} reports the proposal correction, invariant refresh and failed refinements. Wishart--CIR is therefore numerically ineligible for the conditional CRSP ranking in this release. Its completed synthetic experiment diagnoses specification; it supplies no full-panel score or portfolio-optimization recommendation and does not enter the primary Holm or MCS families.\n")
    p=ROOT/'referee_round4/analysis/conclusion.tex';s=p.read_text();paragraphs=s.strip().split('\n\n')
    paragraphs[-1]=(r'The restricted Wishart model has an admissibility-induced dispersion ceiling. An additive CIR factor removes that ceiling, but the frozen return law remains observationally equivalent to a scalar variance mixture. In the completed Heston experiment, matching marginal kurtosis does not produce accurate portfolio ES under the fitted common leverage channel. The favorable no-leverage intervention diagnoses sensitivity; it is not an independently selected winner. Both Wishart specifications remain excluded from conditional CRSP ranking because their numerical gates fail. This is a documented limitation of the evaluated specifications and algorithms, not a rejection of stochastic-covariance modeling. The integrated evidence favors checking conditioning, dependence representation and validation design before attributing performance to model-family flexibility.')
    p.write_text('\n\n'.join(paragraphs)+'\n')
    p=ROOT/'referee_round4/analysis/verification.tex';s=p.read_text()
    s=s.replace('twenty training fits and 26 completed numerical filter runs','twenty training fits and thirty completed numerical filter runs')
    s=s.replace('The earlier release checks reconstruct','The final repeated release checks reconstruct')
    s=s.replace('across 1054 unique files','across 1,054 unique files')
    s=s.replace('Rebuilding from derived panels requires the main source archive, reference-sample archive, round-three checkpoint archive, round-four results/checkpoint archive, and Wishart numerical-audit archive.',
        'Rebuilding all numerical analyses from derived panels requires six archives: the main source, reference samples, round-three checkpoints, round-four results/checkpoints, restricted-Wishart numerical audit, and Wishart--CIR scenario arrays. Compiling the paper alone requires only the compact five-file source package.')
    note=' The final numerical audit includes exact finite-state detailed-balance checks for the independent refresh and independent characteristic-function checks of the frozen-law equivalence. The repeat release audit preserves all 47 baseline numerical-output hashes.'
    if note not in s:s=s.rstrip()+note+'\n'
    p.write_text(s)
    # Add final controls to the existing numerical table, avoiding a further table.
    p=ROOT/'wishart_cir_extension/analysis/filter_table.tex';s=p.read_text()
    s=re.sub(r'% BEGIN final-moves.*?% END final-moves\n?', '', s, flags=re.S)
    rows=['% BEGIN final-moves',r'\midrule\multicolumn{4}{l}{2025 resample-move; three independent refreshes per update} \\']
    for r in status['independent_banks']:
        rows.append(f"$N={r['particles']:,}$ & {100*r['es_relative_rms']:.2f} & {r['score_difference']:+.4f} & "+('Pass' if r['passes'] else 'Fail')+r' \\')
    rows+=['% END final-moves']
    s=s.replace(r'\bottomrule\end{tabular}','\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}')
    old='The 2025 runs use 16,384 predictive paths, except the 32,768-particle run, which uses 32,768.'
    new='The prior/guided 2025 runs use 16,384 predictive paths except at 32,768 particles; both resample-move controls use 32,768 predictive paths. All resample-move controls use 16 substeps and 504 warmup days.'
    s=s.replace(old,new);p.write_text(s)
    print('Final editorial disposition applied; abstract words:',len(abstract.split()))

if __name__=='__main__':main()

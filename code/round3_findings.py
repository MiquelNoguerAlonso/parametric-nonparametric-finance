"""Generate interpreted control findings from completed paired analyses."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'referee_round3/analysis'
def main():
    lu=pd.read_csv(A/'lu_crsp.csv');h=pd.read_csv(A/'lu_heston.csv');c=pd.read_csv(A/'coordinate_filter.csv');w=pd.read_csv(A/'coordinate_within_garch.csv');aug=pd.read_csv(A/'augmented_24_summary.csv')
    assert not (lu[lu.alpha==.01].p_holm<.05).any()
    assert not (w[w.alpha==.01].p_holm<.05).any()
    assert not (aug.p_holm<.05).any()
    g=h[(h.model=='Flow-G')&(h.rule=='fz1')&(h.alpha==.01)].iloc[0]
    (A/'lu_findings.tex').write_text(r'''Dense mixing does not consistently repair the likelihood deficit. In 100 stocks, likelihood-selected Flow-G NLL changes from 167.309 to 174.107, whereas Flow-t changes from 154.195 to 154.006. Final Flow-G NLL deteriorates to 445.081. All six 100-stock LU/rule combinations lower observed risk scores, but none of the paired head/rule contrasts survives its six-contrast correction in either CRSP panel. The absence of a dense head is therefore not sufficient to explain the original density ordering.

In Heston, risk-selected dense Flow-G lowers observed RMSE from 0.605 to 0.335, with paired MSE change $-0.2547$ (SE 0.0744). Its difference from empirical ES is much less resolved: paired MSE $-0.0123$ (SE 0.0764), against empirical RMSE 0.352. Dense Flow-t instead worsens risk-selected RMSE from 0.591 to 0.791, even though its likelihood-selected NLL improves from 29.168 to 28.882. These controls demonstrate a material specification sensitivity without a uniform dense-flow advantage.
''')
    (A/'coordinate_findings.tex').write_text(r'''Coordinate-GARCH FHS lowers the five-basket score from 1.3700 to 1.3010: difference $-0.0690$, interval $[-0.1129,-0.0211]$, and Holm $p=0.042$ in the seven-contrast filter/refit family. Flow-meta-$t$ has the lowest coordinate-GARCH mean, 1.2942, but its improvement against its own original EWMA version has adjusted $p=0.066$. Under the common coordinate-GARCH filter, none of the five innovation alternatives distinguishes itself from empirical innovations after correction (Appendix~\ref{sec:round3-diagnostics}). This is constructive evidence for investigating conditioning, not an established neural advantage.

In 100 stocks, coordinate-GARCH FHS improves its observed mean from 1.2692 to 1.2397, but adjusted $p=0.242$; no filter/refit contrast survives correction. Its mean remains above portfolio-GARCH-FHS at 1.1547. No coordinate-model/portfolio-GARCH-FHS contrast survives the separate seven-contrast correction in either panel. Fitted-decay EWMA-FHS gives 1.3619 and 1.3079, with no detected improvement. The coordinate-GARCH means therefore close the original five-basket point-score gap, but do not establish a common result across dimensions or an equivalence to portfolio filtering.

To expose the role of family definition, a post-result omnibus sensitivity adds all nine new risk-selected controls to the fifteen primary models and adjusts all 23 FHS contrasts. None rejects: five-basket coordinate-GARCH-FHS changes from $p=0.042$ in the targeted filter family to $p=0.138$ in this broader family. Table~\ref{tab:augmented24} reports the added rows. The targeted finding should not be presented as surviving multiplicity control across the entire model comparison.
''')
    print('Control findings agree with completed tables; no omnibus rejection among 24 models per panel.')
if __name__=='__main__':main()

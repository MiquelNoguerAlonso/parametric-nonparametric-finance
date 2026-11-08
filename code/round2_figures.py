"""Annual score differences including the matched filter controls."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_analysis import label
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'referee_round2/analysis'
def main():
 fig,axs=plt.subplots(2,1,figsize=(8.5,6.6),sharex=True)
 models=['Historical','Flow-t','Flow-meta-t','VAE-G','VAE-H','GARCH-FHS','GARCH-EVT','Portfolio-EWMA-FHS']
 for ax,panel in zip(axs,['crsp5','crsp100']):
  a=pd.read_csv(O/(panel+'_forecasts.csv.gz'),parse_dates=['date']);a=a[(a.portfolio=='equal')&(a.alpha==.01)]
  g=a.groupby(['date','model'],as_index=False).fz0.mean();annual=g.assign(year=g.date.dt.year).groupby(['year','model']).fz0.mean().unstack();annual.to_csv(O/(panel+'_annual.csv'))
  for i,m in enumerate(models):
   ax.plot(annual.index,annual[m]-annual.Historical,marker=['o','s','^','D','v','P','X','h'][i],ms=6 if m=='VAE-G' else 4,markerfacecolor='none' if m=='VAE-G' else None,lw=1.2,ls=['-','--',':','-.'][i%4],label=label(m).replace('Portfolio-EWMA-FHS','Portfolio EWMA-FHS'))
  ax.set_title('Five size baskets' if panel=='crsp5' else '100-stock annual universes');ax.set_ylabel('FZ0 minus FHS');ax.grid(alpha=.2)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,fontsize=7,loc='upper center',bbox_to_anchor=(.52,.995));axs[-1].set_xlabel('Out-of-sample year');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(ROOT/'figures/round2_annual.pdf');plt.close(fig)
if __name__=='__main__':main()

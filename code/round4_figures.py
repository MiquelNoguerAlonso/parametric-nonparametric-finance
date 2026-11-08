from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from round4_analysis import ROOT,O,lab,matrix

def main():
    x=pd.read_csv(O/'synthetic_summary.csv');fig,axes=plt.subplots(3,1,figsize=(8.4,10.0))
    for ax,law,title in zip(axes,['t','metat','heston'],['Student target','Heterogeneous meta-t target','Correlated scalar-Heston target']):
        g=x[(x.study==law)&(x.model!='Historical')].sort_values('delta_mse');labels=[lab(s).replace('$','') for s in g.model]
        y=np.arange(len(g));colors=['#285f9b' if s.startswith(('Flow','LU-Flow')) else '#b16a24' if s in ['Student-MLE','Meta-t','Gaussian','Gaussian-LW','Heston-Wishart'] else '#777777' for s in g.model]
        for j,(_,r) in enumerate(g.iterrows()):ax.errorbar(r.delta_mse,j,xerr=r.se,fmt='o',ms=3,color=colors[j],capsize=2,lw=.9)
        ax.set_yticks(y,labels,fontsize=7);ax.axvline(0,color='black',lw=.8);ax.set_title(title,fontsize=10);ax.set_xlabel('Paired MSE difference from empirical ES; one dispersion SE',fontsize=8);ax.grid(axis='x',alpha=.18);ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout(pad=1.5);fig.savefig(ROOT/'figures/wishart_paired_mse.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8.4,6.4),sharex=True)
    models=['Student-MLE','Meta-t','Flow-G','Flow-t','Flow-meta-t','VAE-G','GARCH-FHS']
    markers=['o','s','^','v','D','P','X']
    for ax,panel in zip(axes,['crsp5','crsp100']):
        s,_=matrix(panel);d=s.subtract(s.Historical,axis=0).groupby(s.index.year).mean()
        ax.plot(d.index,np.zeros(len(d)),color='black',lw=1.3,marker='o',ms=3,label='FHS')
        for name,marker in zip(models,markers):ax.plot(d.index,d[name],marker=marker,ms=3,lw=1,label=lab(name).replace('$',''))
        ax.set_title('Five baskets' if panel=='crsp5' else '100 stocks',fontsize=10);ax.set_ylabel('Annual FZ0 difference',fontsize=9);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7,ncol=4);axes[1].set_xlabel('Test year');fig.tight_layout();fig.savefig(ROOT/'figures/round2_annual.pdf');plt.close(fig)

if __name__=='__main__':main()

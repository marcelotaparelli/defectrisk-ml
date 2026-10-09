"""Recreate documentation plots from saved training-only bins (Matplotlib)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).parent
curves = pd.read_csv(folder / 'reliability-bins.csv', index_col=0)

fig,(ax,support_ax)=plt.subplots(2,1,figsize=(8,8),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
for method,color in [('raw','#a94c24'),('sigmoid','#1565c0'),('isotonic','#287a37')]:
    bins=curves.query('method == @method')
    valid=bins[bins['count']>0]
    ax.plot(valid.mean_probability,valid.observed_defect_rate,'o-',color=color,label=method)
    support_ax.plot((bins.lower+bins.upper)/2,bins['count'],'o-',color=color,label=method)
ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted defect probability',ylabel='Observed defective fraction',title='JM1 training-only group-aware OOF reliability')
ax.legend()
ax.grid(alpha=.2)
support_ax.set(xlim=(0,1),xlabel='Probability bin midpoint',ylabel='Rows in bin')
support_ax.grid(alpha=.2)
fig.savefig(folder/'reliability-curve.svg')
fig.savefig(folder/'reliability-curve.png',dpi=150)
plt.close(fig)


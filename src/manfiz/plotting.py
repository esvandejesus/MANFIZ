"""Optional Matplotlib figures generated from actual arrays, never synthetic art."""
from pathlib import Path
import numpy as np


def plot_training(report,path):
    import matplotlib.pyplot as plt
    values=np.asarray(report['objective_history'])[:,0]
    fig,ax=plt.subplots(figsize=(7.2,3.6),layout='constrained')
    ax.plot(np.arange(len(values)),values,color='#7d9ab8',lw=.6,alpha=.65,label='Evaluated objective')
    ax.plot(np.minimum.accumulate(values),color='#123c61',lw=1.5,label='Best so far')
    ax.set(xlabel='Objective evaluation (includes initial/final audits)',ylabel='Joint normalized objective',
           title='Shared-premise training from scratch',yscale='log')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    path=Path(path); fig.savefig(path,dpi=180); fig.savefig(path.with_suffix('.svg')); plt.close(fig)


def plot_intervals(indices,y,prediction,path,*,system=None):
    import matplotlib.pyplot as plt
    y=np.asarray(y); indices=np.asarray(indices)
    n=y.shape[1]
    fig,axes=plt.subplots(n,2,figsize=(11,2.6*n),squeeze=False,layout='constrained')
    start=max(0,len(y)//2-60); end=min(len(y),start+120)
    for o in range(n):
        for j,sl in enumerate((slice(None),slice(start,end))):
            ax=axes[o,j]; k=indices[sl]
            ax.fill_between(k,prediction.lower[sl,o],prediction.upper[sl,o],color='#cfe0ee',label='Noise-inclusive interval')
            ax.plot(k,prediction.center[sl,o],color='#174c72',lw=.8,label='MANFIZ center')
            ax.plot(k,y[sl,o],color='#ad491f',lw=.65,label='Measured output')
            ax.set(xlabel='Sample k (zero-based)',ylabel=f'y{o+1} (physical units)',
                   title=f'Output {o+1}' + (' — detail' if j else ' — full run'))
            ax.grid(alpha=.17)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(f'System {system}: independent held-out run' if system else 'Held-out intervals')
    path=Path(path); fig.savefig(path,dpi=180); fig.savefig(path.with_suffix('.svg')); plt.close(fig)


def plot_budget_sensitivity(rows,path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,6.2),layout='constrained')
    colors=('#174c72','#b4582d','#4c8062')
    for system,color in zip((1,2,3),colors):
        subset=[r for r in rows if int(r['System'])==system]
        if not subset:
            continue
        for o in (1,2):
            ss=sorted((r for r in subset if int(r['Output'])==o),key=lambda r:int(r['QMax']))
            q=[int(r['QMax']) for r in ss]
            axes[0,o-1].plot(q,[float(r['NMPIW']) for r in ss],'-o',color=color,label=f'System {system}')
            if o==1:
                axes[1,0].plot(q,[int(r['MaxPosteriorColumns']) for r in ss],'-o',color=color,label=f'System {system}')
                axes[1,1].plot(q,[100*int(r['Fallbacks'])/max(1,int(r['Reductions'])) for r in ss],'-o',color=color,label=f'System {system}')
    axes[0,0].set(ylabel='NMPIW',title='Output 1: validation intervals')
    axes[0,1].set(ylabel='NMPIW',title='Output 2: validation intervals')
    axes[1,0].set(ylabel='Maximum posterior generators',title='Observed complexity')
    axes[1,1].set(ylabel='Fallbacks / reductions (%)',title='Selection without a feasible candidate')
    for ax in axes.flat:
        ax.set_xlabel('q_max'); ax.grid(alpha=.2); ax.set_xticks(sorted({int(r['QMax']) for r in rows}))
    axes[0,0].legend(frameon=False)
    # Complexity curves can coincide exactly; say so rather than implying missing data.
    offsets={int(r['MaxPosteriorColumns'])-int(r['QMax']) for r in rows}
    if len(offsets)==1:
        axes[1,0].text(.04,.96,f'Curves coincide: maximum = q_max + {next(iter(offsets))}',
                       transform=axes[1,0].transAxes,va='top',fontsize=8.5)
    path=Path(path); fig.savefig(path,dpi=180); fig.savefig(path.with_suffix('.svg')); plt.close(fig)

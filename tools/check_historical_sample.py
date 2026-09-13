"""Check the previously flagged sample under the newly trained Python model."""
from pathlib import Path
import numpy as np
from manfiz import MANFIZ,NARXSpec
from manfiz.benchmarks import make_run
from manfiz.io import write_json


def main():
    import matplotlib.pyplot as plt
    spec=NARXSpec(); ps=926093308
    run=make_run(2,amplitude=1.3,noise_std=.02,phase_seed=ps,noise_seed=ps+100000)
    d=run.regression(spec); model=MANFIZ.load('results/fresh_training/system2/model.npz')
    p=model.predict_interval(d.X,mode='noise'); j=1739
    row=dict(System=2,RetainedZeroBasedIndex=j,RawZeroBasedIndex=int(d.sample_indices[j]),
             MatlabOneBasedIndex=int(d.sample_indices[j]+1),PhaseSeed=ps,NoiseSeed=ps+100000,
             Amplitude=1.3,NoiseStd=.02,Output=1,Measurement=float(d.y[j,0]),
             Center=float(p.center[j,0]),Lower=float(p.lower[j,0]),Upper=float(p.upper[j,0]),
             Slack=float(p.radius[j,0]-abs(d.y[j,0]-p.center[j,0])),
             AllTrajectoryViolations=int(np.sum(abs(d.y-p.center)>p.radius+1e-10)),
             Scope='Retrospective check of a known sample; separate from the 540 confirmation trajectories')
    assert abs(row['Measurement']-.59419143845)<1e-10 and row['Slack']>0
    write_json('results/historical_sample_check.json',row)
    sl=slice(j-25,min(len(d.y),j+26)); k=d.sample_indices[sl]+1
    fig,ax=plt.subplots(figsize=(7.1,2.6),layout='constrained')
    ax.fill_between(k,p.lower[sl,0],p.upper[sl,0],color='#d2e2ef',label='Intervalo con ruido')
    ax.plot(k,p.center[sl,0],color='#174c72',lw=1,label='Centro MANFIZ')
    ax.plot(k,d.y[sl,0],color='#b4522a',lw=1,label='Medición')
    ax.scatter([row['MatlabOneBasedIndex']],[row['Measurement']],color='#ae322c',s=30,zorder=5)
    ax.axvline(row['MatlabOneBasedIndex'],ls=':',lw=.8,color='#ae322c')
    ax.set(xlabel='Muestra original (índice MATLAB desde 1)',ylabel='y1 (unidades físicas)',
           title='Sistema 2: comprobación retrospectiva de la muestra 1770')
    ax.legend(frameon=False,loc='lower left',ncol=3,fontsize=8); ax.grid(alpha=.18)
    fig.savefig('results/historical_sample_check.png',dpi=200)
    fig.savefig('results/historical_sample_check.svg'); plt.close(fig)
    print(row)


if __name__=='__main__':
    main()

"""Refit zonotopes at several q_max values using one newly trained premise model.

This is a controlled consequent-budget study. It deliberately holds the new
nominal model fixed; it is not three additional premise-training experiments.
"""
import argparse
from pathlib import Path
import numpy as np
from manfiz import MANFIZ,NARXSpec
from manfiz.cli import metric_rows,write_csv
from manfiz.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/fresh_training'))
    parser.add_argument('--budgets',nargs='+',type=int,default=[800,1000,1200])
    parser.add_argument('--output',type=Path,default=Path('results/budget_sensitivity'))
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    spec=NARXSpec(); rows=[]
    for folder in sorted(args.results.glob('system[123]')):
        s=int(folder.name[-1])
        with np.load(folder/'datasets.npz',allow_pickle=False) as data:
            runs=[(data[f'u_{i}'],data[f'y_{i}']) for i in range(1,7)]
        train=spec.transform_runs(runs[:4]); val=spec.transform(*runs[4]); test=spec.transform(*runs[5])
        for budget in args.budgets:
            model=MANFIZ.load(folder/'model.npz'); model.config.reduction.q_max=budget
            # Every posterior is rebuilt from its training-derived initial prior.
            model.fit_zonotopes(train.X,train.y,groups=train.groups,sensor_bound=np.sqrt(3)*.01)
            model.calibrate_validation(val.X,val.y)
            p=model.predict_interval(test.X,mode='validation')
            logs=[log for o in model.outputs_ for log in o['history']]
            rows+=metric_rows(test.y,p.center,p.radius,System=s,QMax=budget,
                              AcceptedUpdates=len(logs),Reductions=sum(l['triggered'] for l in logs),
                              Fallbacks=sum(l['fallback'] for l in logs),
                              MaxPosteriorColumns=max(l['posterior_columns'] for l in logs),
                              ConsequentSeconds=model.fit_report_['consequent_seconds'])
            model.save(args.output/f'system{s}_q{budget}.npz')
            print(f'System {s}, q_max={budget}: complete',flush=True)
            write_csv(args.output/'metrics.csv',rows)
    write_json(args.output/'protocol.json',dict(premises='Held fixed at newly trained Python parameters',
               posterior='Reinitialized and fitted separately per budget',budgets=args.budgets,
               evaluation='Same held-out runs; validation mode; no test-driven budget selection'))


if __name__=='__main__':
    main()
